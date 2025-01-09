//===- SemaHLSL.cpp - Semantic Analysis for HLSL constructs ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This implements Semantic Analysis for HLSL constructs.
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaHLSL.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Attrs.inc"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <iterator>
#include <utility>

using namespace clang;
using RegisterType = HLSLResourceBindingAttr::RegisterType;

static RegisterType getRegisterType(ResourceClass RC) {
  switch (RC) {
  case ResourceClass::SRV:
    return RegisterType::SRV;
  case ResourceClass::UAV:
    return RegisterType::UAV;
  case ResourceClass::CBuffer:
    return RegisterType::CBuffer;
  case ResourceClass::Sampler:
    return RegisterType::Sampler;
  }
  llvm_unreachable("unexpected ResourceClass value");
}

// Converts the first letter of string Slot to RegisterType.
// Returns false if the letter does not correspond to a valid register type.
static bool convertToRegisterType(StringRef Slot, RegisterType *RT) {
  assert(RT != nullptr);
  switch (Slot[0]) {
  case 't':
  case 'T':
    *RT = RegisterType::SRV;
    return true;
  case 'u':
  case 'U':
    *RT = RegisterType::UAV;
    return true;
  case 'b':
  case 'B':
    *RT = RegisterType::CBuffer;
    return true;
  case 's':
  case 'S':
    *RT = RegisterType::Sampler;
    return true;
  case 'c':
  case 'C':
    *RT = RegisterType::C;
    return true;
  case 'i':
  case 'I':
    *RT = RegisterType::I;
    return true;
  default:
    return false;
  }
}

static ResourceClass getResourceClass(RegisterType RT) {
  switch (RT) {
  case RegisterType::SRV:
    return ResourceClass::SRV;
  case RegisterType::UAV:
    return ResourceClass::UAV;
  case RegisterType::CBuffer:
    return ResourceClass::CBuffer;
  case RegisterType::Sampler:
    return ResourceClass::Sampler;
  case RegisterType::C:
  case RegisterType::I:
    // Deliberately falling through to the unreachable below.
    break;
  }
  llvm_unreachable("unexpected RegisterType value");
}

DeclBindingInfo *ResourceBindings::addDeclBindingInfo(const VarDecl *VD,
                                                      ResourceClass ResClass) {
  assert(getDeclBindingInfo(VD, ResClass) == nullptr &&
         "DeclBindingInfo already added");
  assert(!hasBindingInfoForDecl(VD) || BindingsList.back().Decl == VD);
  // VarDecl may have multiple entries for different resource classes.
  // DeclToBindingListIndex stores the index of the first binding we saw
  // for this decl. If there are any additional ones then that index
  // shouldn't be updated.
  DeclToBindingListIndex.try_emplace(VD, BindingsList.size());
  return &BindingsList.emplace_back(VD, ResClass);
}

DeclBindingInfo *ResourceBindings::getDeclBindingInfo(const VarDecl *VD,
                                                      ResourceClass ResClass) {
  auto Entry = DeclToBindingListIndex.find(VD);
  if (Entry != DeclToBindingListIndex.end()) {
    for (unsigned Index = Entry->getSecond();
         Index < BindingsList.size() && BindingsList[Index].Decl == VD;
         ++Index) {
      if (BindingsList[Index].ResClass == ResClass)
        return &BindingsList[Index];
    }
  }
  return nullptr;
}

bool ResourceBindings::hasBindingInfoForDecl(const VarDecl *VD) const {
  return DeclToBindingListIndex.contains(VD);
}

SemaHLSL::SemaHLSL(Sema &S) : SemaBase(S) {}

Decl *SemaHLSL::ActOnStartBuffer(Scope *BufferScope, bool CBuffer,
                                 SourceLocation KwLoc, IdentifierInfo *Ident,
                                 SourceLocation IdentLoc,
                                 SourceLocation LBrace) {
  // For anonymous namespace, take the location of the left brace.
  DeclContext *LexicalParent = SemaRef.getCurLexicalContext();
  HLSLBufferDecl *Result = HLSLBufferDecl::Create(
      getASTContext(), LexicalParent, CBuffer, KwLoc, Ident, IdentLoc, LBrace);

  // if CBuffer is false, then it's a TBuffer
  auto RC = CBuffer ? llvm::hlsl::ResourceClass::CBuffer
                    : llvm::hlsl::ResourceClass::SRV;
  auto RK = CBuffer ? llvm::hlsl::ResourceKind::CBuffer
                    : llvm::hlsl::ResourceKind::TBuffer;
  Result->addAttr(HLSLResourceClassAttr::CreateImplicit(getASTContext(), RC));
  Result->addAttr(HLSLResourceAttr::CreateImplicit(getASTContext(), RK));

  SemaRef.PushOnScopeChains(Result, BufferScope);
  SemaRef.PushDeclContext(BufferScope, Result);

  return Result;
}

// Calculate the size of a legacy cbuffer type in bytes based on
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules
static unsigned calculateLegacyCbufferSize(const ASTContext &Context,
                                           QualType T) {
  unsigned Size = 0;
  constexpr unsigned CBufferAlign = 16;
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    for (const FieldDecl *Field : RD->fields()) {
      QualType Ty = Field->getType();
      unsigned FieldSize = calculateLegacyCbufferSize(Context, Ty);
      // FIXME: This is not the correct alignment, it does not work for 16-bit
      // types. See llvm/llvm-project#119641.
      unsigned FieldAlign = 4;
      if (Ty->isAggregateType())
        FieldAlign = CBufferAlign;
      Size = llvm::alignTo(Size, FieldAlign);
      Size += FieldSize;
    }
  } else if (const ConstantArrayType *AT = Context.getAsConstantArrayType(T)) {
    if (unsigned ElementCount = AT->getSize().getZExtValue()) {
      unsigned ElementSize =
          calculateLegacyCbufferSize(Context, AT->getElementType());
      unsigned AlignedElementSize = llvm::alignTo(ElementSize, CBufferAlign);
      Size = AlignedElementSize * (ElementCount - 1) + ElementSize;
    }
  } else if (const VectorType *VT = T->getAs<VectorType>()) {
    unsigned ElementCount = VT->getNumElements();
    unsigned ElementSize =
        calculateLegacyCbufferSize(Context, VT->getElementType());
    Size = ElementSize * ElementCount;
  } else {
    Size = Context.getTypeSize(T) / 8;
  }
  return Size;
}

// Validate packoffset:
// - if packoffset it used it must be set on all declarations inside the buffer
// - packoffset ranges must not overlap
static void validatePackoffset(Sema &S, HLSLBufferDecl *BufDecl) {
  llvm::SmallVector<std::pair<VarDecl *, HLSLPackOffsetAttr *>> PackOffsetVec;

  // Make sure the packoffset annotations are either on all declarations
  // or on none.
  bool HasPackOffset = false;
  bool HasNonPackOffset = false;
  for (auto *Field : BufDecl->decls()) {
    VarDecl *Var = dyn_cast<VarDecl>(Field);
    if (!Var)
      continue;
    if (Field->hasAttr<HLSLPackOffsetAttr>()) {
      PackOffsetVec.emplace_back(Var, Field->getAttr<HLSLPackOffsetAttr>());
      HasPackOffset = true;
    } else {
      HasNonPackOffset = true;
    }
  }

  if (!HasPackOffset)
    return;

  if (HasNonPackOffset)
    S.Diag(BufDecl->getLocation(), diag::warn_hlsl_packoffset_mix);

  // Make sure there is no overlap in packoffset - sort PackOffsetVec by offset
  // and compare adjacent values.
  ASTContext &Context = S.getASTContext();
  std::sort(PackOffsetVec.begin(), PackOffsetVec.end(),
            [](const std::pair<VarDecl *, HLSLPackOffsetAttr *> &LHS,
               const std::pair<VarDecl *, HLSLPackOffsetAttr *> &RHS) {
              return LHS.second->getOffsetInBytes() <
                     RHS.second->getOffsetInBytes();
            });
  for (unsigned i = 0; i < PackOffsetVec.size() - 1; i++) {
    VarDecl *Var = PackOffsetVec[i].first;
    HLSLPackOffsetAttr *Attr = PackOffsetVec[i].second;
    unsigned Size = calculateLegacyCbufferSize(Context, Var->getType());
    unsigned Begin = Attr->getOffsetInBytes();
    unsigned End = Begin + Size;
    unsigned NextBegin = PackOffsetVec[i + 1].second->getOffsetInBytes();
    if (End > NextBegin) {
      VarDecl *NextVar = PackOffsetVec[i + 1].first;
      S.Diag(NextVar->getLocation(), diag::err_hlsl_packoffset_overlap)
          << NextVar << Var;
    }
  }
}

void SemaHLSL::ActOnFinishBuffer(Decl *Dcl, SourceLocation RBrace) {
  auto *BufDecl = cast<HLSLBufferDecl>(Dcl);
  BufDecl->setRBraceLoc(RBrace);

  validatePackoffset(SemaRef, BufDecl);

  SemaRef.PopDeclContext();
}

HLSLNumThreadsAttr *SemaHLSL::mergeNumThreadsAttr(Decl *D,
                                                  const AttributeCommonInfo &AL,
                                                  int X, int Y, int Z) {
  if (HLSLNumThreadsAttr *NT = D->getAttr<HLSLNumThreadsAttr>()) {
    if (NT->getX() != X || NT->getY() != Y || NT->getZ() != Z) {
      Diag(NT->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }
  return ::new (getASTContext())
      HLSLNumThreadsAttr(getASTContext(), AL, X, Y, Z);
}

HLSLWaveSizeAttr *SemaHLSL::mergeWaveSizeAttr(Decl *D,
                                              const AttributeCommonInfo &AL,
                                              int Min, int Max, int Preferred,
                                              int SpelledArgsCount) {
  if (HLSLWaveSizeAttr *WS = D->getAttr<HLSLWaveSizeAttr>()) {
    if (WS->getMin() != Min || WS->getMax() != Max ||
        WS->getPreferred() != Preferred ||
        WS->getSpelledArgsCount() != SpelledArgsCount) {
      Diag(WS->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }
  HLSLWaveSizeAttr *Result = ::new (getASTContext())
      HLSLWaveSizeAttr(getASTContext(), AL, Min, Max, Preferred);
  Result->setSpelledArgsCount(SpelledArgsCount);
  return Result;
}

HLSLShaderAttr *
SemaHLSL::mergeShaderAttr(Decl *D, const AttributeCommonInfo &AL,
                          llvm::Triple::EnvironmentType ShaderType) {
  if (HLSLShaderAttr *NT = D->getAttr<HLSLShaderAttr>()) {
    if (NT->getType() != ShaderType) {
      Diag(NT->getLocation(), diag::err_hlsl_attribute_param_mismatch) << AL;
      Diag(AL.getLoc(), diag::note_conflicting_attribute);
    }
    return nullptr;
  }
  return HLSLShaderAttr::Create(getASTContext(), ShaderType, AL);
}

HLSLParamModifierAttr *
SemaHLSL::mergeParamModifierAttr(Decl *D, const AttributeCommonInfo &AL,
                                 HLSLParamModifierAttr::Spelling Spelling) {
  // We can only merge an `in` attribute with an `out` attribute. All other
  // combinations of duplicated attributes are ill-formed.
  if (HLSLParamModifierAttr *PA = D->getAttr<HLSLParamModifierAttr>()) {
    if ((PA->isIn() && Spelling == HLSLParamModifierAttr::Keyword_out) ||
        (PA->isOut() && Spelling == HLSLParamModifierAttr::Keyword_in)) {
      D->dropAttr<HLSLParamModifierAttr>();
      SourceRange AdjustedRange = {PA->getLocation(), AL.getRange().getEnd()};
      return HLSLParamModifierAttr::Create(
          getASTContext(), /*MergedSpelling=*/true, AdjustedRange,
          HLSLParamModifierAttr::Keyword_inout);
    }
    Diag(AL.getLoc(), diag::err_hlsl_duplicate_parameter_modifier) << AL;
    Diag(PA->getLocation(), diag::note_conflicting_attribute);
    return nullptr;
  }
  return HLSLParamModifierAttr::Create(getASTContext(), AL);
}

void SemaHLSL::ActOnTopLevelFunction(FunctionDecl *FD) {
  auto &TargetInfo = getASTContext().getTargetInfo();

  if (FD->getName() != TargetInfo.getTargetOpts().HLSLEntry)
    return;

  llvm::Triple::EnvironmentType Env = TargetInfo.getTriple().getEnvironment();
  if (HLSLShaderAttr::isValidShaderType(Env) && Env != llvm::Triple::Library) {
    if (const auto *Shader = FD->getAttr<HLSLShaderAttr>()) {
      // The entry point is already annotated - check that it matches the
      // triple.
      if (Shader->getType() != Env) {
        Diag(Shader->getLocation(), diag::err_hlsl_entry_shader_attr_mismatch)
            << Shader;
        FD->setInvalidDecl();
      }
    } else {
      // Implicitly add the shader attribute if the entry function isn't
      // explicitly annotated.
      FD->addAttr(HLSLShaderAttr::CreateImplicit(getASTContext(), Env,
                                                 FD->getBeginLoc()));
    }
  } else {
    switch (Env) {
    case llvm::Triple::UnknownEnvironment:
    case llvm::Triple::Library:
      break;
    default:
      llvm_unreachable("Unhandled environment in triple");
    }
  }
}

void SemaHLSL::CheckEntryPoint(FunctionDecl *FD) {
  const auto *ShaderAttr = FD->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "Entry point has no shader attribute");
  llvm::Triple::EnvironmentType ST = ShaderAttr->getType();
  auto &TargetInfo = getASTContext().getTargetInfo();
  VersionTuple Ver = TargetInfo.getTriple().getOSVersion();
  switch (ST) {
  case llvm::Triple::Pixel:
  case llvm::Triple::Vertex:
  case llvm::Triple::Geometry:
  case llvm::Triple::Hull:
  case llvm::Triple::Domain:
  case llvm::Triple::RayGeneration:
  case llvm::Triple::Intersection:
  case llvm::Triple::AnyHit:
  case llvm::Triple::ClosestHit:
  case llvm::Triple::Miss:
  case llvm::Triple::Callable:
    if (const auto *NT = FD->getAttr<HLSLNumThreadsAttr>()) {
      DiagnoseAttrStageMismatch(NT, ST,
                                {llvm::Triple::Compute,
                                 llvm::Triple::Amplification,
                                 llvm::Triple::Mesh});
      FD->setInvalidDecl();
    }
    if (const auto *WS = FD->getAttr<HLSLWaveSizeAttr>()) {
      DiagnoseAttrStageMismatch(WS, ST,
                                {llvm::Triple::Compute,
                                 llvm::Triple::Amplification,
                                 llvm::Triple::Mesh});
      FD->setInvalidDecl();
    }
    break;

  case llvm::Triple::Compute:
  case llvm::Triple::Amplification:
  case llvm::Triple::Mesh:
    if (!FD->hasAttr<HLSLNumThreadsAttr>()) {
      Diag(FD->getLocation(), diag::err_hlsl_missing_numthreads)
          << llvm::Triple::getEnvironmentTypeName(ST);
      FD->setInvalidDecl();
    }
    if (const auto *WS = FD->getAttr<HLSLWaveSizeAttr>()) {
      if (Ver < VersionTuple(6, 6)) {
        Diag(WS->getLocation(), diag::err_hlsl_attribute_in_wrong_shader_model)
            << WS << "6.6";
        FD->setInvalidDecl();
      } else if (WS->getSpelledArgsCount() > 1 && Ver < VersionTuple(6, 8)) {
        Diag(
            WS->getLocation(),
            diag::err_hlsl_attribute_number_arguments_insufficient_shader_model)
            << WS << WS->getSpelledArgsCount() << "6.8";
        FD->setInvalidDecl();
      }
    }
    break;
  default:
    llvm_unreachable("Unhandled environment in triple");
  }

  for (ParmVarDecl *Param : FD->parameters()) {
    if (const auto *AnnotationAttr = Param->getAttr<HLSLAnnotationAttr>()) {
      CheckSemanticAnnotation(FD, Param, AnnotationAttr);
    } else {
      // FIXME: Handle struct parameters where annotations are on struct fields.
      // See: https://github.com/llvm/llvm-project/issues/57875
      Diag(FD->getLocation(), diag::err_hlsl_missing_semantic_annotation);
      Diag(Param->getLocation(), diag::note_previous_decl) << Param;
      FD->setInvalidDecl();
    }
  }
  // FIXME: Verify return type semantic annotation.
}

void SemaHLSL::CheckSemanticAnnotation(
    FunctionDecl *EntryPoint, const Decl *Param,
    const HLSLAnnotationAttr *AnnotationAttr) {
  auto *ShaderAttr = EntryPoint->getAttr<HLSLShaderAttr>();
  assert(ShaderAttr && "Entry point has no shader attribute");
  llvm::Triple::EnvironmentType ST = ShaderAttr->getType();

  switch (AnnotationAttr->getKind()) {
  case attr::HLSLSV_DispatchThreadID:
  case attr::HLSLSV_GroupIndex:
  case attr::HLSLSV_GroupThreadID:
  case attr::HLSLSV_GroupID:
    if (ST == llvm::Triple::Compute)
      return;
    DiagnoseAttrStageMismatch(AnnotationAttr, ST, {llvm::Triple::Compute});
    break;
  default:
    llvm_unreachable("Unknown HLSLAnnotationAttr");
  }
}

void SemaHLSL::DiagnoseAttrStageMismatch(
    const Attr *A, llvm::Triple::EnvironmentType Stage,
    std::initializer_list<llvm::Triple::EnvironmentType> AllowedStages) {
  SmallVector<StringRef, 8> StageStrings;
  llvm::transform(AllowedStages, std::back_inserter(StageStrings),
                  [](llvm::Triple::EnvironmentType ST) {
                    return StringRef(
                        HLSLShaderAttr::ConvertEnvironmentTypeToStr(ST));
                  });
  Diag(A->getLoc(), diag::err_hlsl_attr_unsupported_in_stage)
      << A << llvm::Triple::getEnvironmentTypeName(Stage)
      << (AllowedStages.size() != 1) << join(StageStrings, ", ");
}

template <CastKind Kind>
static void castVector(Sema &S, ExprResult &E, QualType &Ty, unsigned Sz) {
  if (const auto *VTy = Ty->getAs<VectorType>())
    Ty = VTy->getElementType();
  Ty = S.getASTContext().getExtVectorType(Ty, Sz);
  E = S.ImpCastExprToType(E.get(), Ty, Kind);
}

template <CastKind Kind>
static QualType castElement(Sema &S, ExprResult &E, QualType Ty) {
  E = S.ImpCastExprToType(E.get(), Ty, Kind);
  return Ty;
}

static QualType handleFloatVectorBinOpConversion(
    Sema &SemaRef, ExprResult &LHS, ExprResult &RHS, QualType LHSType,
    QualType RHSType, QualType LElTy, QualType RElTy, bool IsCompAssign) {
  bool LHSFloat = LElTy->isRealFloatingType();
  bool RHSFloat = RElTy->isRealFloatingType();

  if (LHSFloat && RHSFloat) {
    if (IsCompAssign ||
        SemaRef.getASTContext().getFloatingTypeOrder(LElTy, RElTy) > 0)
      return castElement<CK_FloatingCast>(SemaRef, RHS, LHSType);

    return castElement<CK_FloatingCast>(SemaRef, LHS, RHSType);
  }

  if (LHSFloat)
    return castElement<CK_IntegralToFloating>(SemaRef, RHS, LHSType);

  assert(RHSFloat);
  if (IsCompAssign)
    return castElement<clang::CK_FloatingToIntegral>(SemaRef, RHS, LHSType);

  return castElement<CK_IntegralToFloating>(SemaRef, LHS, RHSType);
}

static QualType handleIntegerVectorBinOpConversion(
    Sema &SemaRef, ExprResult &LHS, ExprResult &RHS, QualType LHSType,
    QualType RHSType, QualType LElTy, QualType RElTy, bool IsCompAssign) {

  int IntOrder = SemaRef.Context.getIntegerTypeOrder(LElTy, RElTy);
  bool LHSSigned = LElTy->hasSignedIntegerRepresentation();
  bool RHSSigned = RElTy->hasSignedIntegerRepresentation();
  auto &Ctx = SemaRef.getASTContext();

  // If both types have the same signedness, use the higher ranked type.
  if (LHSSigned == RHSSigned) {
    if (IsCompAssign || IntOrder >= 0)
      return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);

    return castElement<CK_IntegralCast>(SemaRef, LHS, RHSType);
  }

  // If the unsigned type has greater than or equal rank of the signed type, use
  // the unsigned type.
  if (IntOrder != (LHSSigned ? 1 : -1)) {
    if (IsCompAssign || RHSSigned)
      return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);
    return castElement<CK_IntegralCast>(SemaRef, LHS, RHSType);
  }

  // At this point the signed type has higher rank than the unsigned type, which
  // means it will be the same size or bigger. If the signed type is bigger, it
  // can represent all the values of the unsigned type, so select it.
  if (Ctx.getIntWidth(LElTy) != Ctx.getIntWidth(RElTy)) {
    if (IsCompAssign || LHSSigned)
      return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);
    return castElement<CK_IntegralCast>(SemaRef, LHS, RHSType);
  }

  // This is a bit of an odd duck case in HLSL. It shouldn't happen, but can due
  // to C/C++ leaking through. The place this happens today is long vs long
  // long. When arguments are vector<unsigned long, N> and vector<long long, N>,
  // the long long has higher rank than long even though they are the same size.

  // If this is a compound assignment cast the right hand side to the left hand
  // side's type.
  if (IsCompAssign)
    return castElement<CK_IntegralCast>(SemaRef, RHS, LHSType);

  // If this isn't a compound assignment we convert to unsigned long long.
  QualType ElTy = Ctx.getCorrespondingUnsignedType(LHSSigned ? LElTy : RElTy);
  QualType NewTy = Ctx.getExtVectorType(
      ElTy, RHSType->castAs<VectorType>()->getNumElements());
  (void)castElement<CK_IntegralCast>(SemaRef, RHS, NewTy);

  return castElement<CK_IntegralCast>(SemaRef, LHS, NewTy);
}

static CastKind getScalarCastKind(ASTContext &Ctx, QualType DestTy,
                                  QualType SrcTy) {
  if (DestTy->isRealFloatingType() && SrcTy->isRealFloatingType())
    return CK_FloatingCast;
  if (DestTy->isIntegralType(Ctx) && SrcTy->isIntegralType(Ctx))
    return CK_IntegralCast;
  if (DestTy->isRealFloatingType())
    return CK_IntegralToFloating;
  assert(SrcTy->isRealFloatingType() && DestTy->isIntegralType(Ctx));
  return CK_FloatingToIntegral;
}

QualType SemaHLSL::handleVectorBinOpConversion(ExprResult &LHS, ExprResult &RHS,
                                               QualType LHSType,
                                               QualType RHSType,
                                               bool IsCompAssign) {
  const auto *LVecTy = LHSType->getAs<VectorType>();
  const auto *RVecTy = RHSType->getAs<VectorType>();
  auto &Ctx = getASTContext();

  // If the LHS is not a vector and this is a compound assignment, we truncate
  // the argument to a scalar then convert it to the LHS's type.
  if (!LVecTy && IsCompAssign) {
    QualType RElTy = RHSType->castAs<VectorType>()->getElementType();
    RHS = SemaRef.ImpCastExprToType(RHS.get(), RElTy, CK_HLSLVectorTruncation);
    RHSType = RHS.get()->getType();
    if (Ctx.hasSameUnqualifiedType(LHSType, RHSType))
      return LHSType;
    RHS = SemaRef.ImpCastExprToType(RHS.get(), LHSType,
                                    getScalarCastKind(Ctx, LHSType, RHSType));
    return LHSType;
  }

  unsigned EndSz = std::numeric_limits<unsigned>::max();
  unsigned LSz = 0;
  if (LVecTy)
    LSz = EndSz = LVecTy->getNumElements();
  if (RVecTy)
    EndSz = std::min(RVecTy->getNumElements(), EndSz);
  assert(EndSz != std::numeric_limits<unsigned>::max() &&
         "one of the above should have had a value");

  // In a compound assignment, the left operand does not change type, the right
  // operand is converted to the type of the left operand.
  if (IsCompAssign && LSz != EndSz) {
    Diag(LHS.get()->getBeginLoc(),
         diag::err_hlsl_vector_compound_assignment_truncation)
        << LHSType << RHSType;
    return QualType();
  }

  if (RVecTy && RVecTy->getNumElements() > EndSz)
    castVector<CK_HLSLVectorTruncation>(SemaRef, RHS, RHSType, EndSz);
  if (!IsCompAssign && LVecTy && LVecTy->getNumElements() > EndSz)
    castVector<CK_HLSLVectorTruncation>(SemaRef, LHS, LHSType, EndSz);

  if (!RVecTy)
    castVector<CK_VectorSplat>(SemaRef, RHS, RHSType, EndSz);
  if (!IsCompAssign && !LVecTy)
    castVector<CK_VectorSplat>(SemaRef, LHS, LHSType, EndSz);

  // If we're at the same type after resizing we can stop here.
  if (Ctx.hasSameUnqualifiedType(LHSType, RHSType))
    return Ctx.getCommonSugaredType(LHSType, RHSType);

  QualType LElTy = LHSType->castAs<VectorType>()->getElementType();
  QualType RElTy = RHSType->castAs<VectorType>()->getElementType();

  // Handle conversion for floating point vectors.
  if (LElTy->isRealFloatingType() || RElTy->isRealFloatingType())
    return handleFloatVectorBinOpConversion(SemaRef, LHS, RHS, LHSType, RHSType,
                                            LElTy, RElTy, IsCompAssign);

  assert(LElTy->isIntegralType(Ctx) && RElTy->isIntegralType(Ctx) &&
         "HLSL Vectors can only contain integer or floating point types");
  return handleIntegerVectorBinOpConversion(SemaRef, LHS, RHS, LHSType, RHSType,
                                            LElTy, RElTy, IsCompAssign);
}

void SemaHLSL::emitLogicalOperatorFixIt(Expr *LHS, Expr *RHS,
                                        BinaryOperatorKind Opc) {
  assert((Opc == BO_LOr || Opc == BO_LAnd) &&
         "Called with non-logical operator");
  llvm::SmallVector<char, 256> Buff;
  llvm::raw_svector_ostream OS(Buff);
  PrintingPolicy PP(SemaRef.getLangOpts());
  StringRef NewFnName = Opc == BO_LOr ? "or" : "and";
  OS << NewFnName << "(";
  LHS->printPretty(OS, nullptr, PP);
  OS << ", ";
  RHS->printPretty(OS, nullptr, PP);
  OS << ")";
  SourceRange FullRange = SourceRange(LHS->getBeginLoc(), RHS->getEndLoc());
  SemaRef.Diag(LHS->getBeginLoc(), diag::note_function_suggestion)
      << NewFnName << FixItHint::CreateReplacement(FullRange, OS.str());
}

void SemaHLSL::handleNumThreadsAttr(Decl *D, const ParsedAttr &AL) {
  llvm::VersionTuple SMVersion =
      getASTContext().getTargetInfo().getTriple().getOSVersion();
  uint32_t ZMax = 1024;
  uint32_t ThreadMax = 1024;
  if (SMVersion.getMajor() <= 4) {
    ZMax = 1;
    ThreadMax = 768;
  } else if (SMVersion.getMajor() == 5) {
    ZMax = 64;
    ThreadMax = 1024;
  }

  uint32_t X;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), X))
    return;
  if (X > 1024) {
    Diag(AL.getArgAsExpr(0)->getExprLoc(),
         diag::err_hlsl_numthreads_argument_oor)
        << 0 << 1024;
    return;
  }
  uint32_t Y;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Y))
    return;
  if (Y > 1024) {
    Diag(AL.getArgAsExpr(1)->getExprLoc(),
         diag::err_hlsl_numthreads_argument_oor)
        << 1 << 1024;
    return;
  }
  uint32_t Z;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(2), Z))
    return;
  if (Z > ZMax) {
    SemaRef.Diag(AL.getArgAsExpr(2)->getExprLoc(),
                 diag::err_hlsl_numthreads_argument_oor)
        << 2 << ZMax;
    return;
  }

  if (X * Y * Z > ThreadMax) {
    Diag(AL.getLoc(), diag::err_hlsl_numthreads_invalid) << ThreadMax;
    return;
  }

  HLSLNumThreadsAttr *NewAttr = mergeNumThreadsAttr(D, AL, X, Y, Z);
  if (NewAttr)
    D->addAttr(NewAttr);
}

static bool isValidWaveSizeValue(unsigned Value) {
  return llvm::isPowerOf2_32(Value) && Value >= 4 && Value <= 128;
}

void SemaHLSL::handleWaveSizeAttr(Decl *D, const ParsedAttr &AL) {
  // validate that the wavesize argument is a power of 2 between 4 and 128
  // inclusive
  unsigned SpelledArgsCount = AL.getNumArgs();
  if (SpelledArgsCount == 0 || SpelledArgsCount > 3)
    return;

  uint32_t Min;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), Min))
    return;

  uint32_t Max = 0;
  if (SpelledArgsCount > 1 &&
      !SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Max))
    return;

  uint32_t Preferred = 0;
  if (SpelledArgsCount > 2 &&
      !SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(2), Preferred))
    return;

  if (SpelledArgsCount > 2) {
    if (!isValidWaveSizeValue(Preferred)) {
      Diag(AL.getArgAsExpr(2)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << llvm::dxil::MinWaveSize << llvm::dxil::MaxWaveSize
          << Preferred;
      return;
    }
    // Preferred not in range.
    if (Preferred < Min || Preferred > Max) {
      Diag(AL.getArgAsExpr(2)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << Min << Max << Preferred;
      return;
    }
  } else if (SpelledArgsCount > 1) {
    if (!isValidWaveSizeValue(Max)) {
      Diag(AL.getArgAsExpr(1)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << llvm::dxil::MinWaveSize << llvm::dxil::MaxWaveSize << Max;
      return;
    }
    if (Max < Min) {
      Diag(AL.getLoc(), diag::err_attribute_argument_invalid) << AL << 1;
      return;
    } else if (Max == Min) {
      Diag(AL.getLoc(), diag::warn_attr_min_eq_max) << AL;
    }
  } else {
    if (!isValidWaveSizeValue(Min)) {
      Diag(AL.getArgAsExpr(0)->getExprLoc(),
           diag::err_attribute_power_of_two_in_range)
          << AL << llvm::dxil::MinWaveSize << llvm::dxil::MaxWaveSize << Min;
      return;
    }
  }

  HLSLWaveSizeAttr *NewAttr =
      mergeWaveSizeAttr(D, AL, Min, Max, Preferred, SpelledArgsCount);
  if (NewAttr)
    D->addAttr(NewAttr);
}

bool SemaHLSL::diagnoseInputIDType(QualType T, const ParsedAttr &AL) {
  const auto *VT = T->getAs<VectorType>();

  if (!T->hasUnsignedIntegerRepresentation() ||
      (VT && VT->getNumElements() > 3)) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_type)
        << AL << "uint/uint2/uint3";
    return false;
  }

  return true;
}

void SemaHLSL::handleSV_DispatchThreadIDAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnoseInputIDType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext())
                 HLSLSV_DispatchThreadIDAttr(getASTContext(), AL));
}

void SemaHLSL::handleSV_GroupThreadIDAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnoseInputIDType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext())
                 HLSLSV_GroupThreadIDAttr(getASTContext(), AL));
}

void SemaHLSL::handleSV_GroupIDAttr(Decl *D, const ParsedAttr &AL) {
  auto *VD = cast<ValueDecl>(D);
  if (!diagnoseInputIDType(VD->getType(), AL))
    return;

  D->addAttr(::new (getASTContext()) HLSLSV_GroupIDAttr(getASTContext(), AL));
}

void SemaHLSL::handlePackOffsetAttr(Decl *D, const ParsedAttr &AL) {
  if (!isa<VarDecl>(D) || !isa<HLSLBufferDecl>(D->getDeclContext())) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_ast_node)
        << AL << "shader constant in a constant buffer";
    return;
  }

  uint32_t SubComponent;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(0), SubComponent))
    return;
  uint32_t Component;
  if (!SemaRef.checkUInt32Argument(AL, AL.getArgAsExpr(1), Component))
    return;

  QualType T = cast<VarDecl>(D)->getType().getCanonicalType();
  // Check if T is an array or struct type.
  // TODO: mark matrix type as aggregate type.
  bool IsAggregateTy = (T->isArrayType() || T->isStructureType());

  // Check Component is valid for T.
  if (Component) {
    unsigned Size = getASTContext().getTypeSize(T);
    if (IsAggregateTy || Size > 128) {
      Diag(AL.getLoc(), diag::err_hlsl_packoffset_cross_reg_boundary);
      return;
    } else {
      // Make sure Component + sizeof(T) <= 4.
      if ((Component * 32 + Size) > 128) {
        Diag(AL.getLoc(), diag::err_hlsl_packoffset_cross_reg_boundary);
        return;
      }
      QualType EltTy = T;
      if (const auto *VT = T->getAs<VectorType>())
        EltTy = VT->getElementType();
      unsigned Align = getASTContext().getTypeAlign(EltTy);
      if (Align > 32 && Component == 1) {
        // NOTE: Component 3 will hit err_hlsl_packoffset_cross_reg_boundary.
        // So we only need to check Component 1 here.
        Diag(AL.getLoc(), diag::err_hlsl_packoffset_alignment_mismatch)
            << Align << EltTy;
        return;
      }
    }
  }

  D->addAttr(::new (getASTContext()) HLSLPackOffsetAttr(
      getASTContext(), AL, SubComponent, Component));
}

void SemaHLSL::handleShaderAttr(Decl *D, const ParsedAttr &AL) {
  StringRef Str;
  SourceLocation ArgLoc;
  if (!SemaRef.checkStringLiteralArgumentAttr(AL, 0, Str, &ArgLoc))
    return;

  llvm::Triple::EnvironmentType ShaderType;
  if (!HLSLShaderAttr::ConvertStrToEnvironmentType(Str, ShaderType)) {
    Diag(AL.getLoc(), diag::warn_attribute_type_not_supported)
        << AL << Str << ArgLoc;
    return;
  }

  // FIXME: check function match the shader stage.

  HLSLShaderAttr *NewAttr = mergeShaderAttr(D, AL, ShaderType);
  if (NewAttr)
    D->addAttr(NewAttr);
}

bool clang::CreateHLSLAttributedResourceType(
    Sema &S, QualType Wrapped, ArrayRef<const Attr *> AttrList,
    QualType &ResType, HLSLAttributedResourceLocInfo *LocInfo) {
  assert(AttrList.size() && "expected list of resource attributes");

  QualType ContainedTy = QualType();
  TypeSourceInfo *ContainedTyInfo = nullptr;
  SourceLocation LocBegin = AttrList[0]->getRange().getBegin();
  SourceLocation LocEnd = AttrList[0]->getRange().getEnd();

  HLSLAttributedResourceType::Attributes ResAttrs;

  bool HasResourceClass = false;
  for (const Attr *A : AttrList) {
    if (!A)
      continue;
    LocEnd = A->getRange().getEnd();
    switch (A->getKind()) {
    case attr::HLSLResourceClass: {
      ResourceClass RC = cast<HLSLResourceClassAttr>(A)->getResourceClass();
      if (HasResourceClass) {
        S.Diag(A->getLocation(), ResAttrs.ResourceClass == RC
                                     ? diag::warn_duplicate_attribute_exact
                                     : diag::warn_duplicate_attribute)
            << A;
        return false;
      }
      ResAttrs.ResourceClass = RC;
      HasResourceClass = true;
      break;
    }
    case attr::HLSLROV:
      if (ResAttrs.IsROV) {
        S.Diag(A->getLocation(), diag::warn_duplicate_attribute_exact) << A;
        return false;
      }
      ResAttrs.IsROV = true;
      break;
    case attr::HLSLRawBuffer:
      if (ResAttrs.RawBuffer) {
        S.Diag(A->getLocation(), diag::warn_duplicate_attribute_exact) << A;
        return false;
      }
      ResAttrs.RawBuffer = true;
      break;
    case attr::HLSLContainedType: {
      const HLSLContainedTypeAttr *CTAttr = cast<HLSLContainedTypeAttr>(A);
      QualType Ty = CTAttr->getType();
      if (!ContainedTy.isNull()) {
        S.Diag(A->getLocation(), ContainedTy == Ty
                                     ? diag::warn_duplicate_attribute_exact
                                     : diag::warn_duplicate_attribute)
            << A;
        return false;
      }
      ContainedTy = Ty;
      ContainedTyInfo = CTAttr->getTypeLoc();
      break;
    }
    default:
      llvm_unreachable("unhandled resource attribute type");
    }
  }

  if (!HasResourceClass) {
    S.Diag(AttrList.back()->getRange().getEnd(),
           diag::err_hlsl_missing_resource_class);
    return false;
  }

  ResType = S.getASTContext().getHLSLAttributedResourceType(
      Wrapped, ContainedTy, ResAttrs);

  if (LocInfo && ContainedTyInfo) {
    LocInfo->Range = SourceRange(LocBegin, LocEnd);
    LocInfo->ContainedTyInfo = ContainedTyInfo;
  }
  return true;
}

// Validates and creates an HLSL attribute that is applied as type attribute on
// HLSL resource. The attributes are collected in HLSLResourcesTypeAttrs and at
// the end of the declaration they are applied to the declaration type by
// wrapping it in HLSLAttributedResourceType.
bool SemaHLSL::handleResourceTypeAttr(QualType T, const ParsedAttr &AL) {
  // only allow resource type attributes on intangible types
  if (!T->isHLSLResourceType()) {
    Diag(AL.getLoc(), diag::err_hlsl_attribute_needs_intangible_type)
        << AL << getASTContext().HLSLResourceTy;
    return false;
  }

  // validate number of arguments
  if (!AL.checkExactlyNumArgs(SemaRef, AL.getMinArgs()))
    return false;

  Attr *A = nullptr;
  switch (AL.getKind()) {
  case ParsedAttr::AT_HLSLResourceClass: {
    if (!AL.isArgIdent(0)) {
      Diag(AL.getLoc(), diag::err_attribute_argument_type)
          << AL << AANT_ArgumentIdentifier;
      return false;
    }

    IdentifierLoc *Loc = AL.getArgAsIdent(0);
    StringRef Identifier = Loc->Ident->getName();
    SourceLocation ArgLoc = Loc->Loc;

    // Validate resource class value
    ResourceClass RC;
    if (!HLSLResourceClassAttr::ConvertStrToResourceClass(Identifier, RC)) {
      Diag(ArgLoc, diag::warn_attribute_type_not_supported)
          << "ResourceClass" << Identifier;
      return false;
    }
    A = HLSLResourceClassAttr::Create(getASTContext(), RC, AL.getLoc());
    break;
  }

  case ParsedAttr::AT_HLSLROV:
    A = HLSLROVAttr::Create(getASTContext(), AL.getLoc());
    break;

  case ParsedAttr::AT_HLSLRawBuffer:
    A = HLSLRawBufferAttr::Create(getASTContext(), AL.getLoc());
    break;

  case ParsedAttr::AT_HLSLContainedType: {
    if (AL.getNumArgs() != 1 && !AL.hasParsedType()) {
      Diag(AL.getLoc(), diag::err_attribute_wrong_number_arguments) << AL << 1;
      return false;
    }

    TypeSourceInfo *TSI = nullptr;
    QualType QT = SemaRef.GetTypeFromParser(AL.getTypeArg(), &TSI);
    assert(TSI && "no type source info for attribute argument");
    if (SemaRef.RequireCompleteType(TSI->getTypeLoc().getBeginLoc(), QT,
                                    diag::err_incomplete_type))
      return false;
    A = HLSLContainedTypeAttr::Create(getASTContext(), TSI, AL.getLoc());
    break;
  }

  default:
    llvm_unreachable("unhandled HLSL attribute");
  }

  HLSLResourcesTypeAttrs.emplace_back(A);
  return true;
}

// Combines all resource type attributes and creates HLSLAttributedResourceType.
QualType SemaHLSL::ProcessResourceTypeAttributes(QualType CurrentType) {
  if (!HLSLResourcesTypeAttrs.size())
    return CurrentType;

  QualType QT = CurrentType;
  HLSLAttributedResourceLocInfo LocInfo;
  if (CreateHLSLAttributedResourceType(SemaRef, CurrentType,
                                       HLSLResourcesTypeAttrs, QT, &LocInfo)) {
    const HLSLAttributedResourceType *RT =
        cast<HLSLAttributedResourceType>(QT.getTypePtr());

    // Temporarily store TypeLoc information for the new type.
    // It will be transferred to HLSLAttributesResourceTypeLoc
    // shortly after the type is created by TypeSpecLocFiller which
    // will call the TakeLocForHLSLAttribute method below.
    LocsForHLSLAttributedResources.insert(std::pair(RT, LocInfo));
  }
  HLSLResourcesTypeAttrs.clear();
  return QT;
}

// Returns source location for the HLSLAttributedResourceType
HLSLAttributedResourceLocInfo
SemaHLSL::TakeLocForHLSLAttribute(const HLSLAttributedResourceType *RT) {
  HLSLAttributedResourceLocInfo LocInfo = {};
  auto I = LocsForHLSLAttributedResources.find(RT);
  if (I != LocsForHLSLAttributedResources.end()) {
    LocInfo = I->second;
    LocsForHLSLAttributedResources.erase(I);
    return LocInfo;
  }
  LocInfo.Range = SourceRange();
  return LocInfo;
}

// Walks though the global variable declaration, collects all resource binding
// requirements and adds them to Bindings
void SemaHLSL::collectResourcesOnUserRecordDecl(const VarDecl *VD,
                                                const RecordType *RT) {
  const RecordDecl *RD = RT->getDecl();
  for (FieldDecl *FD : RD->fields()) {
    const Type *Ty = FD->getType()->getUnqualifiedDesugaredType();

    // Unwrap arrays
    // FIXME: Calculate array size while unwrapping
    assert(!Ty->isIncompleteArrayType() &&
           "incomplete arrays inside user defined types are not supported");
    while (Ty->isConstantArrayType()) {
      const ConstantArrayType *CAT = cast<ConstantArrayType>(Ty);
      Ty = CAT->getElementType()->getUnqualifiedDesugaredType();
    }

    if (!Ty->isRecordType())
      continue;

    if (const HLSLAttributedResourceType *AttrResType =
            HLSLAttributedResourceType::findHandleTypeOnResource(Ty)) {
      // Add a new DeclBindingInfo to Bindings if it does not already exist
      ResourceClass RC = AttrResType->getAttrs().ResourceClass;
      DeclBindingInfo *DBI = Bindings.getDeclBindingInfo(VD, RC);
      if (!DBI)
        Bindings.addDeclBindingInfo(VD, RC);
    } else if (const RecordType *RT = dyn_cast<RecordType>(Ty)) {
      // Recursively scan embedded struct or class; it would be nice to do this
      // without recursion, but tricky to correctly calculate the size of the
      // binding, which is something we are probably going to need to do later
      // on. Hopefully nesting of structs in structs too many levels is
      // unlikely.
      collectResourcesOnUserRecordDecl(VD, RT);
    }
  }
}

// Diagnore localized register binding errors for a single binding; does not
// diagnose resource binding on user record types, that will be done later
// in processResourceBindingOnDecl based on the information collected in
// collectResourcesOnVarDecl.
// Returns false if the register binding is not valid.
static bool DiagnoseLocalRegisterBinding(Sema &S, SourceLocation &ArgLoc,
                                         Decl *D, RegisterType RegType,
                                         bool SpecifiedSpace) {
  int RegTypeNum = static_cast<int>(RegType);

  // check if the decl type is groupshared
  if (D->hasAttr<HLSLGroupSharedAddressSpaceAttr>()) {
    S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
    return false;
  }

  // Cbuffers and Tbuffers are HLSLBufferDecl types
  if (HLSLBufferDecl *CBufferOrTBuffer = dyn_cast<HLSLBufferDecl>(D)) {
    ResourceClass RC = CBufferOrTBuffer->isCBuffer() ? ResourceClass::CBuffer
                                                     : ResourceClass::SRV;
    if (RegType == getRegisterType(RC))
      return true;

    S.Diag(D->getLocation(), diag::err_hlsl_binding_type_mismatch)
        << RegTypeNum;
    return false;
  }

  // Samplers, UAVs, and SRVs are VarDecl types
  assert(isa<VarDecl>(D) && "D is expected to be VarDecl or HLSLBufferDecl");
  VarDecl *VD = cast<VarDecl>(D);

  // Resource
  if (const HLSLAttributedResourceType *AttrResType =
          HLSLAttributedResourceType::findHandleTypeOnResource(
              VD->getType().getTypePtr())) {
    if (RegType == getRegisterType(AttrResType->getAttrs().ResourceClass))
      return true;

    S.Diag(D->getLocation(), diag::err_hlsl_binding_type_mismatch)
        << RegTypeNum;
    return false;
  }

  const clang::Type *Ty = VD->getType().getTypePtr();
  while (Ty->isArrayType())
    Ty = Ty->getArrayElementTypeNoTypeQual();

  // Basic types
  if (Ty->isArithmeticType()) {
    bool DeclaredInCOrTBuffer = isa<HLSLBufferDecl>(D->getDeclContext());
    if (SpecifiedSpace && !DeclaredInCOrTBuffer)
      S.Diag(ArgLoc, diag::err_hlsl_space_on_global_constant);

    if (!DeclaredInCOrTBuffer &&
        (Ty->isIntegralType(S.getASTContext()) || Ty->isFloatingType())) {
      // Default Globals
      if (RegType == RegisterType::CBuffer)
        S.Diag(ArgLoc, diag::warn_hlsl_deprecated_register_type_b);
      else if (RegType != RegisterType::C)
        S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
    } else {
      if (RegType == RegisterType::C)
        S.Diag(ArgLoc, diag::warn_hlsl_register_type_c_packoffset);
      else
        S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
    }
    return false;
  }
  if (Ty->isRecordType())
    // RecordTypes will be diagnosed in processResourceBindingOnDecl
    // that is called from ActOnVariableDeclarator
    return true;

  // Anything else is an error
  S.Diag(ArgLoc, diag::err_hlsl_binding_type_mismatch) << RegTypeNum;
  return false;
}

static bool ValidateMultipleRegisterAnnotations(Sema &S, Decl *TheDecl,
                                                RegisterType regType) {
  // make sure that there are no two register annotations
  // applied to the decl with the same register type
  bool RegisterTypesDetected[5] = {false};
  RegisterTypesDetected[static_cast<int>(regType)] = true;

  for (auto it = TheDecl->attr_begin(); it != TheDecl->attr_end(); ++it) {
    if (HLSLResourceBindingAttr *attr =
            dyn_cast<HLSLResourceBindingAttr>(*it)) {

      RegisterType otherRegType = attr->getRegisterType();
      if (RegisterTypesDetected[static_cast<int>(otherRegType)]) {
        int otherRegTypeNum = static_cast<int>(otherRegType);
        S.Diag(TheDecl->getLocation(),
               diag::err_hlsl_duplicate_register_annotation)
            << otherRegTypeNum;
        return false;
      }
      RegisterTypesDetected[static_cast<int>(otherRegType)] = true;
    }
  }
  return true;
}

static bool DiagnoseHLSLRegisterAttribute(Sema &S, SourceLocation &ArgLoc,
                                          Decl *D, RegisterType RegType,
                                          bool SpecifiedSpace) {

  // exactly one of these two types should be set
  assert(((isa<VarDecl>(D) && !isa<HLSLBufferDecl>(D)) ||
          (!isa<VarDecl>(D) && isa<HLSLBufferDecl>(D))) &&
         "expecting VarDecl or HLSLBufferDecl");

  // check if the declaration contains resource matching the register type
  if (!DiagnoseLocalRegisterBinding(S, ArgLoc, D, RegType, SpecifiedSpace))
    return false;

  // next, if multiple register annotations exist, check that none conflict.
  return ValidateMultipleRegisterAnnotations(S, D, RegType);
}

void SemaHLSL::handleResourceBindingAttr(Decl *TheDecl, const ParsedAttr &AL) {
  if (isa<VarDecl>(TheDecl)) {
    if (SemaRef.RequireCompleteType(TheDecl->getBeginLoc(),
                                    cast<ValueDecl>(TheDecl)->getType(),
                                    diag::err_incomplete_type))
      return;
  }
  StringRef Space = "space0";
  StringRef Slot = "";

  if (!AL.isArgIdent(0)) {
    Diag(AL.getLoc(), diag::err_attribute_argument_type)
        << AL << AANT_ArgumentIdentifier;
    return;
  }

  IdentifierLoc *Loc = AL.getArgAsIdent(0);
  StringRef Str = Loc->Ident->getName();
  SourceLocation ArgLoc = Loc->Loc;

  SourceLocation SpaceArgLoc;
  bool SpecifiedSpace = false;
  if (AL.getNumArgs() == 2) {
    SpecifiedSpace = true;
    Slot = Str;
    if (!AL.isArgIdent(1)) {
      Diag(AL.getLoc(), diag::err_attribute_argument_type)
          << AL << AANT_ArgumentIdentifier;
      return;
    }

    IdentifierLoc *Loc = AL.getArgAsIdent(1);
    Space = Loc->Ident->getName();
    SpaceArgLoc = Loc->Loc;
  } else {
    Slot = Str;
  }

  RegisterType RegType;
  unsigned SlotNum = 0;
  unsigned SpaceNum = 0;

  // Validate.
  if (!Slot.empty()) {
    if (!convertToRegisterType(Slot, &RegType)) {
      Diag(ArgLoc, diag::err_hlsl_binding_type_invalid) << Slot.substr(0, 1);
      return;
    }
    if (RegType == RegisterType::I) {
      Diag(ArgLoc, diag::warn_hlsl_deprecated_register_type_i);
      return;
    }

    StringRef SlotNumStr = Slot.substr(1);
    if (SlotNumStr.getAsInteger(10, SlotNum)) {
      Diag(ArgLoc, diag::err_hlsl_unsupported_register_number);
      return;
    }
  }

  if (!Space.starts_with("space")) {
    Diag(SpaceArgLoc, diag::err_hlsl_expected_space) << Space;
    return;
  }
  StringRef SpaceNumStr = Space.substr(5);
  if (SpaceNumStr.getAsInteger(10, SpaceNum)) {
    Diag(SpaceArgLoc, diag::err_hlsl_expected_space) << Space;
    return;
  }

  if (!DiagnoseHLSLRegisterAttribute(SemaRef, ArgLoc, TheDecl, RegType,
                                     SpecifiedSpace))
    return;

  HLSLResourceBindingAttr *NewAttr =
      HLSLResourceBindingAttr::Create(getASTContext(), Slot, Space, AL);
  if (NewAttr) {
    NewAttr->setBinding(RegType, SlotNum, SpaceNum);
    TheDecl->addAttr(NewAttr);
  }
}

void SemaHLSL::handleParamModifierAttr(Decl *D, const ParsedAttr &AL) {
  HLSLParamModifierAttr *NewAttr = mergeParamModifierAttr(
      D, AL,
      static_cast<HLSLParamModifierAttr::Spelling>(AL.getSemanticSpelling()));
  if (NewAttr)
    D->addAttr(NewAttr);
}

namespace {

/// This class implements HLSL availability diagnostics for default
/// and relaxed mode
///
/// The goal of this diagnostic is to emit an error or warning when an
/// unavailable API is found in code that is reachable from the shader
/// entry function or from an exported function (when compiling a shader
/// library).
///
/// This is done by traversing the AST of all shader entry point functions
/// and of all exported functions, and any functions that are referenced
/// from this AST. In other words, any functions that are reachable from
/// the entry points.
class DiagnoseHLSLAvailability : public DynamicRecursiveASTVisitor {
  Sema &SemaRef;

  // Stack of functions to be scaned
  llvm::SmallVector<const FunctionDecl *, 8> DeclsToScan;

  // Tracks which environments functions have been scanned in.
  //
  // Maps FunctionDecl to an unsigned number that represents the set of shader
  // environments the function has been scanned for.
  // The llvm::Triple::EnvironmentType enum values for shader stages guaranteed
  // to be numbered from llvm::Triple::Pixel to llvm::Triple::Amplification
  // (verified by static_asserts in Triple.cpp), we can use it to index
  // individual bits in the set, as long as we shift the values to start with 0
  // by subtracting the value of llvm::Triple::Pixel first.
  //
  // The N'th bit in the set will be set if the function has been scanned
  // in shader environment whose llvm::Triple::EnvironmentType integer value
  // equals (llvm::Triple::Pixel + N).
  //
  // For example, if a function has been scanned in compute and pixel stage
  // environment, the value will be 0x21 (100001 binary) because:
  //
  //   (int)(llvm::Triple::Pixel - llvm::Triple::Pixel) == 0
  //   (int)(llvm::Triple::Compute - llvm::Triple::Pixel) == 5
  //
  // A FunctionDecl is mapped to 0 (or not included in the map) if it has not
  // been scanned in any environment.
  llvm::DenseMap<const FunctionDecl *, unsigned> ScannedDecls;

  // Do not access these directly, use the get/set methods below to make
  // sure the values are in sync
  llvm::Triple::EnvironmentType CurrentShaderEnvironment;
  unsigned CurrentShaderStageBit;

  // True if scanning a function that was already scanned in a different
  // shader stage context, and therefore we should not report issues that
  // depend only on shader model version because they would be duplicate.
  bool ReportOnlyShaderStageIssues;

  // Helper methods for dealing with current stage context / environment
  void SetShaderStageContext(llvm::Triple::EnvironmentType ShaderType) {
    static_assert(sizeof(unsigned) >= 4);
    assert(HLSLShaderAttr::isValidShaderType(ShaderType));
    assert((unsigned)(ShaderType - llvm::Triple::Pixel) < 31 &&
           "ShaderType is too big for this bitmap"); // 31 is reserved for
                                                     // "unknown"

    unsigned bitmapIndex = ShaderType - llvm::Triple::Pixel;
    CurrentShaderEnvironment = ShaderType;
    CurrentShaderStageBit = (1 << bitmapIndex);
  }

  void SetUnknownShaderStageContext() {
    CurrentShaderEnvironment = llvm::Triple::UnknownEnvironment;
    CurrentShaderStageBit = (1 << 31);
  }

  llvm::Triple::EnvironmentType GetCurrentShaderEnvironment() const {
    return CurrentShaderEnvironment;
  }

  bool InUnknownShaderStageContext() const {
    return CurrentShaderEnvironment == llvm::Triple::UnknownEnvironment;
  }

  // Helper methods for dealing with shader stage bitmap
  void AddToScannedFunctions(const FunctionDecl *FD) {
    unsigned &ScannedStages = ScannedDecls[FD];
    ScannedStages |= CurrentShaderStageBit;
  }

  unsigned GetScannedStages(const FunctionDecl *FD) { return ScannedDecls[FD]; }

  bool WasAlreadyScannedInCurrentStage(const FunctionDecl *FD) {
    return WasAlreadyScannedInCurrentStage(GetScannedStages(FD));
  }

  bool WasAlreadyScannedInCurrentStage(unsigned ScannerStages) {
    return ScannerStages & CurrentShaderStageBit;
  }

  static bool NeverBeenScanned(unsigned ScannedStages) {
    return ScannedStages == 0;
  }

  // Scanning methods
  void HandleFunctionOrMethodRef(FunctionDecl *FD, Expr *RefExpr);
  void CheckDeclAvailability(NamedDecl *D, const AvailabilityAttr *AA,
                             SourceRange Range);
  const AvailabilityAttr *FindAvailabilityAttr(const Decl *D);
  bool HasMatchingEnvironmentOrNone(const AvailabilityAttr *AA);

public:
  DiagnoseHLSLAvailability(Sema &SemaRef)
      : SemaRef(SemaRef),
        CurrentShaderEnvironment(llvm::Triple::UnknownEnvironment),
        CurrentShaderStageBit(0), ReportOnlyShaderStageIssues(false) {}

  // AST traversal methods
  void RunOnTranslationUnit(const TranslationUnitDecl *TU);
  void RunOnFunction(const FunctionDecl *FD);

  bool VisitDeclRefExpr(DeclRefExpr *DRE) override {
    FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(DRE->getDecl());
    if (FD)
      HandleFunctionOrMethodRef(FD, DRE);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) override {
    FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(ME->getMemberDecl());
    if (FD)
      HandleFunctionOrMethodRef(FD, ME);
    return true;
  }
};

void DiagnoseHLSLAvailability::HandleFunctionOrMethodRef(FunctionDecl *FD,
                                                         Expr *RefExpr) {
  assert((isa<DeclRefExpr>(RefExpr) || isa<MemberExpr>(RefExpr)) &&
         "expected DeclRefExpr or MemberExpr");

  // has a definition -> add to stack to be scanned
  const FunctionDecl *FDWithBody = nullptr;
  if (FD->hasBody(FDWithBody)) {
    if (!WasAlreadyScannedInCurrentStage(FDWithBody))
      DeclsToScan.push_back(FDWithBody);
    return;
  }

  // no body -> diagnose availability
  const AvailabilityAttr *AA = FindAvailabilityAttr(FD);
  if (AA)
    CheckDeclAvailability(
        FD, AA, SourceRange(RefExpr->getBeginLoc(), RefExpr->getEndLoc()));
}

void DiagnoseHLSLAvailability::RunOnTranslationUnit(
    const TranslationUnitDecl *TU) {

  // Iterate over all shader entry functions and library exports, and for those
  // that have a body (definiton), run diag scan on each, setting appropriate
  // shader environment context based on whether it is a shader entry function
  // or an exported function. Exported functions can be in namespaces and in
  // export declarations so we need to scan those declaration contexts as well.
  llvm::SmallVector<const DeclContext *, 8> DeclContextsToScan;
  DeclContextsToScan.push_back(TU);

  while (!DeclContextsToScan.empty()) {
    const DeclContext *DC = DeclContextsToScan.pop_back_val();
    for (auto &D : DC->decls()) {
      // do not scan implicit declaration generated by the implementation
      if (D->isImplicit())
        continue;

      // for namespace or export declaration add the context to the list to be
      // scanned later
      if (llvm::dyn_cast<NamespaceDecl>(D) || llvm::dyn_cast<ExportDecl>(D)) {
        DeclContextsToScan.push_back(llvm::dyn_cast<DeclContext>(D));
        continue;
      }

      // skip over other decls or function decls without body
      const FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(D);
      if (!FD || !FD->isThisDeclarationADefinition())
        continue;

      // shader entry point
      if (HLSLShaderAttr *ShaderAttr = FD->getAttr<HLSLShaderAttr>()) {
        SetShaderStageContext(ShaderAttr->getType());
        RunOnFunction(FD);
        continue;
      }
      // exported library function
      // FIXME: replace this loop with external linkage check once issue #92071
      // is resolved
      bool isExport = FD->isInExportDeclContext();
      if (!isExport) {
        for (const auto *Redecl : FD->redecls()) {
          if (Redecl->isInExportDeclContext()) {
            isExport = true;
            break;
          }
        }
      }
      if (isExport) {
        SetUnknownShaderStageContext();
        RunOnFunction(FD);
        continue;
      }
    }
  }
}

void DiagnoseHLSLAvailability::RunOnFunction(const FunctionDecl *FD) {
  assert(DeclsToScan.empty() && "DeclsToScan should be empty");
  DeclsToScan.push_back(FD);

  while (!DeclsToScan.empty()) {
    // Take one decl from the stack and check it by traversing its AST.
    // For any CallExpr found during the traversal add it's callee to the top of
    // the stack to be processed next. Functions already processed are stored in
    // ScannedDecls.
    const FunctionDecl *FD = DeclsToScan.pop_back_val();

    // Decl was already scanned
    const unsigned ScannedStages = GetScannedStages(FD);
    if (WasAlreadyScannedInCurrentStage(ScannedStages))
      continue;

    ReportOnlyShaderStageIssues = !NeverBeenScanned(ScannedStages);

    AddToScannedFunctions(FD);
    TraverseStmt(FD->getBody());
  }
}

bool DiagnoseHLSLAvailability::HasMatchingEnvironmentOrNone(
    const AvailabilityAttr *AA) {
  IdentifierInfo *IIEnvironment = AA->getEnvironment();
  if (!IIEnvironment)
    return true;

  llvm::Triple::EnvironmentType CurrentEnv = GetCurrentShaderEnvironment();
  if (CurrentEnv == llvm::Triple::UnknownEnvironment)
    return false;

  llvm::Triple::EnvironmentType AttrEnv =
      AvailabilityAttr::getEnvironmentType(IIEnvironment->getName());

  return CurrentEnv == AttrEnv;
}

const AvailabilityAttr *
DiagnoseHLSLAvailability::FindAvailabilityAttr(const Decl *D) {
  AvailabilityAttr const *PartialMatch = nullptr;
  // Check each AvailabilityAttr to find the one for this platform.
  // For multiple attributes with the same platform try to find one for this
  // environment.
  for (const auto *A : D->attrs()) {
    if (const auto *Avail = dyn_cast<AvailabilityAttr>(A)) {
      StringRef AttrPlatform = Avail->getPlatform()->getName();
      StringRef TargetPlatform =
          SemaRef.getASTContext().getTargetInfo().getPlatformName();

      // Match the platform name.
      if (AttrPlatform == TargetPlatform) {
        // Find the best matching attribute for this environment
        if (HasMatchingEnvironmentOrNone(Avail))
          return Avail;
        PartialMatch = Avail;
      }
    }
  }
  return PartialMatch;
}

// Check availability against target shader model version and current shader
// stage and emit diagnostic
void DiagnoseHLSLAvailability::CheckDeclAvailability(NamedDecl *D,
                                                     const AvailabilityAttr *AA,
                                                     SourceRange Range) {

  IdentifierInfo *IIEnv = AA->getEnvironment();

  if (!IIEnv) {
    // The availability attribute does not have environment -> it depends only
    // on shader model version and not on specific the shader stage.

    // Skip emitting the diagnostics if the diagnostic mode is set to
    // strict (-fhlsl-strict-availability) because all relevant diagnostics
    // were already emitted in the DiagnoseUnguardedAvailability scan
    // (SemaAvailability.cpp).
    if (SemaRef.getLangOpts().HLSLStrictAvailability)
      return;

    // Do not report shader-stage-independent issues if scanning a function
    // that was already scanned in a different shader stage context (they would
    // be duplicate)
    if (ReportOnlyShaderStageIssues)
      return;

  } else {
    // The availability attribute has environment -> we need to know
    // the current stage context to property diagnose it.
    if (InUnknownShaderStageContext())
      return;
  }

  // Check introduced version and if environment matches
  bool EnvironmentMatches = HasMatchingEnvironmentOrNone(AA);
  VersionTuple Introduced = AA->getIntroduced();
  VersionTuple TargetVersion =
      SemaRef.Context.getTargetInfo().getPlatformMinVersion();

  if (TargetVersion >= Introduced && EnvironmentMatches)
    return;

  // Emit diagnostic message
  const TargetInfo &TI = SemaRef.getASTContext().getTargetInfo();
  llvm::StringRef PlatformName(
      AvailabilityAttr::getPrettyPlatformName(TI.getPlatformName()));

  llvm::StringRef CurrentEnvStr =
      llvm::Triple::getEnvironmentTypeName(GetCurrentShaderEnvironment());

  llvm::StringRef AttrEnvStr =
      AA->getEnvironment() ? AA->getEnvironment()->getName() : "";
  bool UseEnvironment = !AttrEnvStr.empty();

  if (EnvironmentMatches) {
    SemaRef.Diag(Range.getBegin(), diag::warn_hlsl_availability)
        << Range << D << PlatformName << Introduced.getAsString()
        << UseEnvironment << CurrentEnvStr;
  } else {
    SemaRef.Diag(Range.getBegin(), diag::warn_hlsl_availability_unavailable)
        << Range << D;
  }

  SemaRef.Diag(D->getLocation(), diag::note_partial_availability_specified_here)
      << D << PlatformName << Introduced.getAsString()
      << SemaRef.Context.getTargetInfo().getPlatformMinVersion().getAsString()
      << UseEnvironment << AttrEnvStr << CurrentEnvStr;
}

} // namespace

void SemaHLSL::DiagnoseAvailabilityViolations(TranslationUnitDecl *TU) {
  // Skip running the diagnostics scan if the diagnostic mode is
  // strict (-fhlsl-strict-availability) and the target shader stage is known
  // because all relevant diagnostics were already emitted in the
  // DiagnoseUnguardedAvailability scan (SemaAvailability.cpp).
  const TargetInfo &TI = SemaRef.getASTContext().getTargetInfo();
  if (SemaRef.getLangOpts().HLSLStrictAvailability &&
      TI.getTriple().getEnvironment() != llvm::Triple::EnvironmentType::Library)
    return;

  DiagnoseHLSLAvailability(SemaRef).RunOnTranslationUnit(TU);
}

// Helper function for CheckHLSLBuiltinFunctionCall
static bool CheckVectorElementCallArgs(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() > 1);
  ExprResult A = TheCall->getArg(0);

  QualType ArgTyA = A.get()->getType();

  auto *VecTyA = ArgTyA->getAs<VectorType>();
  SourceLocation BuiltinLoc = TheCall->getBeginLoc();

  for (unsigned i = 1; i < TheCall->getNumArgs(); ++i) {
    ExprResult B = TheCall->getArg(i);
    QualType ArgTyB = B.get()->getType();
    auto *VecTyB = ArgTyB->getAs<VectorType>();
    if (VecTyA == nullptr && VecTyB == nullptr)
      return false;

    if (VecTyA && VecTyB) {
      bool retValue = false;
      if (VecTyA->getElementType() != VecTyB->getElementType()) {
        // Note: type promotion is intended to be handeled via the intrinsics
        //  and not the builtin itself.
        S->Diag(TheCall->getBeginLoc(),
                diag::err_vec_builtin_incompatible_vector)
            << TheCall->getDirectCallee() << /*useAllTerminology*/ true
            << SourceRange(A.get()->getBeginLoc(), B.get()->getEndLoc());
        retValue = true;
      }
      if (VecTyA->getNumElements() != VecTyB->getNumElements()) {
        // You should only be hitting this case if you are calling the builtin
        // directly. HLSL intrinsics should avoid this case via a
        // HLSLVectorTruncation.
        S->Diag(BuiltinLoc, diag::err_vec_builtin_incompatible_vector)
            << TheCall->getDirectCallee() << /*useAllTerminology*/ true
            << SourceRange(TheCall->getArg(0)->getBeginLoc(),
                           TheCall->getArg(1)->getEndLoc());
        retValue = true;
      }
      return retValue;
    }
  }

  // Note: if we get here one of the args is a scalar which
  // requires a VectorSplat on Arg0 or Arg1
  S->Diag(BuiltinLoc, diag::err_vec_builtin_non_vector)
      << TheCall->getDirectCallee() << /*useAllTerminology*/ true
      << SourceRange(TheCall->getArg(0)->getBeginLoc(),
                     TheCall->getArg(1)->getEndLoc());
  return true;
}

static bool CheckArgTypeMatches(Sema *S, Expr *Arg, QualType ExpectedType) {
  QualType ArgType = Arg->getType();
  if (!S->getASTContext().hasSameUnqualifiedType(ArgType, ExpectedType)) {
    S->Diag(Arg->getBeginLoc(), diag::err_typecheck_convert_incompatible)
        << ArgType << ExpectedType << 1 << 0 << 0;
    return true;
  }
  return false;
}

static bool CheckArgTypeIsCorrect(
    Sema *S, Expr *Arg, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check) {
  QualType PassedType = Arg->getType();
  if (Check(PassedType)) {
    if (auto *VecTyA = PassedType->getAs<VectorType>())
      ExpectedType = S->Context.getVectorType(
          ExpectedType, VecTyA->getNumElements(), VecTyA->getVectorKind());
    S->Diag(Arg->getBeginLoc(), diag::err_typecheck_convert_incompatible)
        << PassedType << ExpectedType << 1 << 0 << 0;
    return true;
  }
  return false;
}

static bool CheckAllArgTypesAreCorrect(
    Sema *S, CallExpr *TheCall, QualType ExpectedType,
    llvm::function_ref<bool(clang::QualType PassedType)> Check) {
  for (unsigned i = 0; i < TheCall->getNumArgs(); ++i) {
    Expr *Arg = TheCall->getArg(i);
    if (CheckArgTypeIsCorrect(S, Arg, ExpectedType, Check)) {
      return true;
    }
  }
  return false;
}

static bool CheckAllArgsHaveFloatRepresentation(Sema *S, CallExpr *TheCall) {
  auto checkAllFloatTypes = [](clang::QualType PassedType) -> bool {
    return !PassedType->hasFloatingRepresentation();
  };
  return CheckAllArgTypesAreCorrect(S, TheCall, S->Context.FloatTy,
                                    checkAllFloatTypes);
}

static bool CheckFloatOrHalfRepresentations(Sema *S, CallExpr *TheCall) {
  auto checkFloatorHalf = [](clang::QualType PassedType) -> bool {
    clang::QualType BaseType =
        PassedType->isVectorType()
            ? PassedType->getAs<clang::VectorType>()->getElementType()
            : PassedType;
    return !BaseType->isHalfType() && !BaseType->isFloat32Type();
  };
  return CheckAllArgTypesAreCorrect(S, TheCall, S->Context.FloatTy,
                                    checkFloatorHalf);
}

static bool CheckModifiableLValue(Sema *S, CallExpr *TheCall,
                                  unsigned ArgIndex) {
  auto *Arg = TheCall->getArg(ArgIndex);
  SourceLocation OrigLoc = Arg->getExprLoc();
  if (Arg->IgnoreCasts()->isModifiableLvalue(S->Context, &OrigLoc) ==
      Expr::MLV_Valid)
    return false;
  S->Diag(OrigLoc, diag::error_hlsl_inout_lvalue) << Arg << 0;
  return true;
}

static bool CheckNoDoubleVectors(Sema *S, CallExpr *TheCall) {
  auto checkDoubleVector = [](clang::QualType PassedType) -> bool {
    if (const auto *VecTy = PassedType->getAs<VectorType>())
      return VecTy->getElementType()->isDoubleType();
    return false;
  };
  return CheckAllArgTypesAreCorrect(S, TheCall, S->Context.FloatTy,
                                    checkDoubleVector);
}
static bool CheckFloatingOrIntRepresentation(Sema *S, CallExpr *TheCall) {
  auto checkAllSignedTypes = [](clang::QualType PassedType) -> bool {
    return !PassedType->hasIntegerRepresentation() &&
           !PassedType->hasFloatingRepresentation();
  };
  return CheckAllArgTypesAreCorrect(S, TheCall, S->Context.IntTy,
                                    checkAllSignedTypes);
}

static bool CheckUnsignedIntRepresentation(Sema *S, CallExpr *TheCall) {
  auto checkAllUnsignedTypes = [](clang::QualType PassedType) -> bool {
    return !PassedType->hasUnsignedIntegerRepresentation();
  };
  return CheckAllArgTypesAreCorrect(S, TheCall, S->Context.UnsignedIntTy,
                                    checkAllUnsignedTypes);
}

static void SetElementTypeAsReturnType(Sema *S, CallExpr *TheCall,
                                       QualType ReturnType) {
  auto *VecTyA = TheCall->getArg(0)->getType()->getAs<VectorType>();
  if (VecTyA)
    ReturnType = S->Context.getVectorType(ReturnType, VecTyA->getNumElements(),
                                          VectorKind::Generic);
  TheCall->setType(ReturnType);
}

static bool CheckScalarOrVector(Sema *S, CallExpr *TheCall, QualType Scalar,
                                unsigned ArgIndex) {
  assert(TheCall->getNumArgs() >= ArgIndex);
  QualType ArgType = TheCall->getArg(ArgIndex)->getType();
  auto *VTy = ArgType->getAs<VectorType>();
  // not the scalar or vector<scalar>
  if (!(S->Context.hasSameUnqualifiedType(ArgType, Scalar) ||
        (VTy &&
         S->Context.hasSameUnqualifiedType(VTy->getElementType(), Scalar)))) {
    S->Diag(TheCall->getArg(0)->getBeginLoc(),
            diag::err_typecheck_expect_scalar_or_vector)
        << ArgType << Scalar;
    return true;
  }
  return false;
}

static bool CheckAnyScalarOrVector(Sema *S, CallExpr *TheCall,
                                   unsigned ArgIndex) {
  assert(TheCall->getNumArgs() >= ArgIndex);
  QualType ArgType = TheCall->getArg(ArgIndex)->getType();
  auto *VTy = ArgType->getAs<VectorType>();
  // not the scalar or vector<scalar>
  if (!(ArgType->isScalarType() ||
        (VTy && VTy->getElementType()->isScalarType()))) {
    S->Diag(TheCall->getArg(0)->getBeginLoc(),
            diag::err_typecheck_expect_any_scalar_or_vector)
        << ArgType;
    return true;
  }
  return false;
}

static bool CheckBoolSelect(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() == 3);
  Expr *Arg1 = TheCall->getArg(1);
  Expr *Arg2 = TheCall->getArg(2);
  if (!S->Context.hasSameUnqualifiedType(Arg1->getType(), Arg2->getType())) {
    S->Diag(TheCall->getBeginLoc(),
            diag::err_typecheck_call_different_arg_types)
        << Arg1->getType() << Arg2->getType() << Arg1->getSourceRange()
        << Arg2->getSourceRange();
    return true;
  }

  TheCall->setType(Arg1->getType());
  return false;
}

static bool CheckVectorSelect(Sema *S, CallExpr *TheCall) {
  assert(TheCall->getNumArgs() == 3);
  Expr *Arg1 = TheCall->getArg(1);
  Expr *Arg2 = TheCall->getArg(2);
  if (!Arg1->getType()->isVectorType()) {
    S->Diag(Arg1->getBeginLoc(), diag::err_builtin_non_vector_type)
        << "Second" << TheCall->getDirectCallee() << Arg1->getType()
        << Arg1->getSourceRange();
    return true;
  }

  if (!Arg2->getType()->isVectorType()) {
    S->Diag(Arg2->getBeginLoc(), diag::err_builtin_non_vector_type)
        << "Third" << TheCall->getDirectCallee() << Arg2->getType()
        << Arg2->getSourceRange();
    return true;
  }

  if (!S->Context.hasSameUnqualifiedType(Arg1->getType(), Arg2->getType())) {
    S->Diag(TheCall->getBeginLoc(),
            diag::err_typecheck_call_different_arg_types)
        << Arg1->getType() << Arg2->getType() << Arg1->getSourceRange()
        << Arg2->getSourceRange();
    return true;
  }

  // caller has checked that Arg0 is a vector.
  // check all three args have the same length.
  if (TheCall->getArg(0)->getType()->getAs<VectorType>()->getNumElements() !=
      Arg1->getType()->getAs<VectorType>()->getNumElements()) {
    S->Diag(TheCall->getBeginLoc(),
            diag::err_typecheck_vector_lengths_not_equal)
        << TheCall->getArg(0)->getType() << Arg1->getType()
        << TheCall->getArg(0)->getSourceRange() << Arg1->getSourceRange();
    return true;
  }
  TheCall->setType(Arg1->getType());
  return false;
}

static bool CheckResourceHandle(
    Sema *S, CallExpr *TheCall, unsigned ArgIndex,
    llvm::function_ref<bool(const HLSLAttributedResourceType *ResType)> Check =
        nullptr) {
  assert(TheCall->getNumArgs() >= ArgIndex);
  QualType ArgType = TheCall->getArg(ArgIndex)->getType();
  const HLSLAttributedResourceType *ResTy =
      ArgType.getTypePtr()->getAs<HLSLAttributedResourceType>();
  if (!ResTy) {
    S->Diag(TheCall->getArg(ArgIndex)->getBeginLoc(),
            diag::err_typecheck_expect_hlsl_resource)
        << ArgType;
    return true;
  }
  if (Check && Check(ResTy)) {
    S->Diag(TheCall->getArg(ArgIndex)->getExprLoc(),
            diag::err_invalid_hlsl_resource_type)
        << ArgType;
    return true;
  }
  return false;
}

// Note: returning true in this case results in CheckBuiltinFunctionCall
// returning an ExprError
bool SemaHLSL::CheckBuiltinFunctionCall(unsigned BuiltinID, CallExpr *TheCall) {
  switch (BuiltinID) {
  case Builtin::BI__builtin_hlsl_resource_getpointer: {
    if (SemaRef.checkArgCount(TheCall, 2) ||
        CheckResourceHandle(&SemaRef, TheCall, 0) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(1),
                            SemaRef.getASTContext().UnsignedIntTy))
      return true;

    auto *ResourceTy =
        TheCall->getArg(0)->getType()->castAs<HLSLAttributedResourceType>();
    QualType ContainedTy = ResourceTy->getContainedType();
    // TODO: Map to an hlsl_device address space.
    TheCall->setType(getASTContext().getPointerType(ContainedTy));
    TheCall->setValueKind(VK_LValue);

    break;
  }
  case Builtin::BI__builtin_hlsl_all:
  case Builtin::BI__builtin_hlsl_any: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_asdouble: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckUnsignedIntRepresentation(&SemaRef, TheCall))
      return true;

    SetElementTypeAsReturnType(&SemaRef, TheCall, getASTContext().DoubleTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_clamp: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;
    if (CheckVectorElementCallArgs(&SemaRef, TheCall))
      return true;
    if (SemaRef.BuiltinElementwiseTernaryMath(
            TheCall, /*CheckForFloatArgs*/
            TheCall->getArg(0)->getType()->hasFloatingRepresentation()))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_cross: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckVectorElementCallArgs(&SemaRef, TheCall))
      return true;
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    // ensure both args have 3 elements
    int NumElementsArg1 =
        TheCall->getArg(0)->getType()->castAs<VectorType>()->getNumElements();
    int NumElementsArg2 =
        TheCall->getArg(1)->getType()->castAs<VectorType>()->getNumElements();

    if (NumElementsArg1 != 3) {
      int LessOrMore = NumElementsArg1 > 3 ? 1 : 0;
      SemaRef.Diag(TheCall->getBeginLoc(),
                   diag::err_vector_incorrect_num_elements)
          << LessOrMore << 3 << NumElementsArg1 << /*operand*/ 1;
      return true;
    }
    if (NumElementsArg2 != 3) {
      int LessOrMore = NumElementsArg2 > 3 ? 1 : 0;

      SemaRef.Diag(TheCall->getBeginLoc(),
                   diag::err_vector_incorrect_num_elements)
          << LessOrMore << 3 << NumElementsArg2 << /*operand*/ 1;
      return true;
    }

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  case Builtin::BI__builtin_hlsl_dot: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckVectorElementCallArgs(&SemaRef, TheCall))
      return true;
    if (SemaRef.BuiltinVectorToScalarMath(TheCall))
      return true;
    if (CheckNoDoubleVectors(&SemaRef, TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_firstbithigh: {
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;

    const Expr *Arg = TheCall->getArg(0);
    QualType ArgTy = Arg->getType();
    QualType EltTy = ArgTy;

    QualType ResTy = SemaRef.Context.UnsignedIntTy;

    if (auto *VecTy = EltTy->getAs<VectorType>()) {
      EltTy = VecTy->getElementType();
      ResTy = SemaRef.Context.getVectorType(ResTy, VecTy->getNumElements(),
                                            VecTy->getVectorKind());
    }

    if (!EltTy->isIntegerType()) {
      Diag(Arg->getBeginLoc(), diag::err_builtin_invalid_arg_type)
          << 1 << /* integer ty */ 6 << ArgTy;
      return true;
    }

    TheCall->setType(ResTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_select: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;
    if (CheckScalarOrVector(&SemaRef, TheCall, getASTContext().BoolTy, 0))
      return true;
    QualType ArgTy = TheCall->getArg(0)->getType();
    if (ArgTy->isBooleanType() && CheckBoolSelect(&SemaRef, TheCall))
      return true;
    auto *VTy = ArgTy->getAs<VectorType>();
    if (VTy && VTy->getElementType()->isBooleanType() &&
        CheckVectorSelect(&SemaRef, TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_saturate:
  case Builtin::BI__builtin_hlsl_elementwise_rcp: {
    if (CheckAllArgsHaveFloatRepresentation(&SemaRef, TheCall))
      return true;
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_degrees:
  case Builtin::BI__builtin_hlsl_elementwise_radians:
  case Builtin::BI__builtin_hlsl_elementwise_rsqrt:
  case Builtin::BI__builtin_hlsl_elementwise_frac: {
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_isinf: {
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    SetElementTypeAsReturnType(&SemaRef, TheCall, getASTContext().BoolTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_lerp: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;
    if (CheckVectorElementCallArgs(&SemaRef, TheCall))
      return true;
    if (SemaRef.BuiltinElementwiseTernaryMath(TheCall))
      return true;
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_length: {
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    QualType RetTy;

    if (auto *VTy = ArgTyA->getAs<VectorType>())
      RetTy = VTy->getElementType();
    else
      RetTy = TheCall->getArg(0)->getType();

    TheCall->setType(RetTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_mad: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;
    if (CheckVectorElementCallArgs(&SemaRef, TheCall))
      return true;
    if (SemaRef.BuiltinElementwiseTernaryMath(
            TheCall, /*CheckForFloatArgs*/
            TheCall->getArg(0)->getType()->hasFloatingRepresentation()))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_normalize: {
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_sign: {
    if (CheckFloatingOrIntRepresentation(&SemaRef, TheCall))
      return true;
    if (SemaRef.PrepareBuiltinElementwiseMathOneArgCall(TheCall))
      return true;
    SetElementTypeAsReturnType(&SemaRef, TheCall, getASTContext().IntTy);
    break;
  }
  case Builtin::BI__builtin_hlsl_step: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;

    ExprResult A = TheCall->getArg(0);
    QualType ArgTyA = A.get()->getType();
    // return type is the same as the input type
    TheCall->setType(ArgTyA);
    break;
  }
  // Note these are llvm builtins that we want to catch invalid intrinsic
  // generation. Normal handling of these builitns will occur elsewhere.
  case Builtin::BI__builtin_elementwise_bitreverse: {
    if (CheckUnsignedIntRepresentation(&SemaRef, TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_wave_read_lane_at: {
    if (SemaRef.checkArgCount(TheCall, 2))
      return true;

    // Ensure index parameter type can be interpreted as a uint
    ExprResult Index = TheCall->getArg(1);
    QualType ArgTyIndex = Index.get()->getType();
    if (!ArgTyIndex->isIntegerType()) {
      SemaRef.Diag(TheCall->getArg(1)->getBeginLoc(),
                   diag::err_typecheck_convert_incompatible)
          << ArgTyIndex << SemaRef.Context.UnsignedIntTy << 1 << 0 << 0;
      return true;
    }

    // Ensure input expr type is a scalar/vector and the same as the return type
    if (CheckAnyScalarOrVector(&SemaRef, TheCall, 0))
      return true;

    ExprResult Expr = TheCall->getArg(0);
    QualType ArgTyExpr = Expr.get()->getType();
    TheCall->setType(ArgTyExpr);
    break;
  }
  case Builtin::BI__builtin_hlsl_wave_get_lane_index: {
    if (SemaRef.checkArgCount(TheCall, 0))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_splitdouble: {
    if (SemaRef.checkArgCount(TheCall, 3))
      return true;

    if (CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.DoubleTy, 0) ||
        CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.UnsignedIntTy,
                            1) ||
        CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.UnsignedIntTy,
                            2))
      return true;

    if (CheckModifiableLValue(&SemaRef, TheCall, 1) ||
        CheckModifiableLValue(&SemaRef, TheCall, 2))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_elementwise_clip: {
    if (SemaRef.checkArgCount(TheCall, 1))
      return true;

    if (CheckScalarOrVector(&SemaRef, TheCall, SemaRef.Context.FloatTy, 0))
      return true;
    break;
  }
  case Builtin::BI__builtin_elementwise_acos:
  case Builtin::BI__builtin_elementwise_asin:
  case Builtin::BI__builtin_elementwise_atan:
  case Builtin::BI__builtin_elementwise_atan2:
  case Builtin::BI__builtin_elementwise_ceil:
  case Builtin::BI__builtin_elementwise_cos:
  case Builtin::BI__builtin_elementwise_cosh:
  case Builtin::BI__builtin_elementwise_exp:
  case Builtin::BI__builtin_elementwise_exp2:
  case Builtin::BI__builtin_elementwise_floor:
  case Builtin::BI__builtin_elementwise_fmod:
  case Builtin::BI__builtin_elementwise_log:
  case Builtin::BI__builtin_elementwise_log2:
  case Builtin::BI__builtin_elementwise_log10:
  case Builtin::BI__builtin_elementwise_pow:
  case Builtin::BI__builtin_elementwise_roundeven:
  case Builtin::BI__builtin_elementwise_sin:
  case Builtin::BI__builtin_elementwise_sinh:
  case Builtin::BI__builtin_elementwise_sqrt:
  case Builtin::BI__builtin_elementwise_tan:
  case Builtin::BI__builtin_elementwise_tanh:
  case Builtin::BI__builtin_elementwise_trunc: {
    if (CheckFloatOrHalfRepresentations(&SemaRef, TheCall))
      return true;
    break;
  }
  case Builtin::BI__builtin_hlsl_buffer_update_counter: {
    auto checkResTy = [](const HLSLAttributedResourceType *ResTy) -> bool {
      return !(ResTy->getAttrs().ResourceClass == ResourceClass::UAV &&
               ResTy->getAttrs().RawBuffer && ResTy->hasContainedType());
    };
    if (SemaRef.checkArgCount(TheCall, 2) ||
        CheckResourceHandle(&SemaRef, TheCall, 0, checkResTy) ||
        CheckArgTypeMatches(&SemaRef, TheCall->getArg(1),
                            SemaRef.getASTContext().IntTy))
      return true;
    Expr *OffsetExpr = TheCall->getArg(1);
    std::optional<llvm::APSInt> Offset =
        OffsetExpr->getIntegerConstantExpr(SemaRef.getASTContext());
    if (!Offset.has_value() || std::abs(Offset->getExtValue()) != 1) {
      SemaRef.Diag(TheCall->getArg(1)->getBeginLoc(),
                   diag::err_hlsl_expect_arg_const_int_one_or_neg_one)
          << 1;
      return true;
    }
    break;
  }
  }
  return false;
}

static void BuildFlattenedTypeList(QualType BaseTy,
                                   llvm::SmallVectorImpl<QualType> &List) {
  llvm::SmallVector<QualType, 16> WorkList;
  WorkList.push_back(BaseTy);
  while (!WorkList.empty()) {
    QualType T = WorkList.pop_back_val();
    T = T.getCanonicalType().getUnqualifiedType();
    assert(!isa<MatrixType>(T) && "Matrix types not yet supported in HLSL");
    if (const auto *AT = dyn_cast<ConstantArrayType>(T)) {
      llvm::SmallVector<QualType, 16> ElementFields;
      // Generally I've avoided recursion in this algorithm, but arrays of
      // structs could be time-consuming to flatten and churn through on the
      // work list. Hopefully nesting arrays of structs containing arrays
      // of structs too many levels deep is unlikely.
      BuildFlattenedTypeList(AT->getElementType(), ElementFields);
      // Repeat the element's field list n times.
      for (uint64_t Ct = 0; Ct < AT->getZExtSize(); ++Ct)
        List.insert(List.end(), ElementFields.begin(), ElementFields.end());
      continue;
    }
    // Vectors can only have element types that are builtin types, so this can
    // add directly to the list instead of to the WorkList.
    if (const auto *VT = dyn_cast<VectorType>(T)) {
      List.insert(List.end(), VT->getNumElements(), VT->getElementType());
      continue;
    }
    if (const auto *RT = dyn_cast<RecordType>(T)) {
      const RecordDecl *RD = RT->getDecl();
      if (RD->isUnion()) {
        List.push_back(T);
        continue;
      }
      const CXXRecordDecl *CXXD = dyn_cast<CXXRecordDecl>(RD);

      llvm::SmallVector<QualType, 16> FieldTypes;
      if (CXXD && CXXD->isStandardLayout())
        RD = CXXD->getStandardLayoutBaseWithFields();

      for (const auto *FD : RD->fields())
        FieldTypes.push_back(FD->getType());
      // Reverse the newly added sub-range.
      std::reverse(FieldTypes.begin(), FieldTypes.end());
      WorkList.insert(WorkList.end(), FieldTypes.begin(), FieldTypes.end());

      // If this wasn't a standard layout type we may also have some base
      // classes to deal with.
      if (CXXD && !CXXD->isStandardLayout()) {
        FieldTypes.clear();
        for (const auto &Base : CXXD->bases())
          FieldTypes.push_back(Base.getType());
        std::reverse(FieldTypes.begin(), FieldTypes.end());
        WorkList.insert(WorkList.end(), FieldTypes.begin(), FieldTypes.end());
      }
      continue;
    }
    List.push_back(T);
  }
}

bool SemaHLSL::IsTypedResourceElementCompatible(clang::QualType QT) {
  // null and array types are not allowed.
  if (QT.isNull() || QT->isArrayType())
    return false;

  // UDT types are not allowed
  if (QT->isRecordType())
    return false;

  if (QT->isBooleanType() || QT->isEnumeralType())
    return false;

  // the only other valid builtin types are scalars or vectors
  if (QT->isArithmeticType()) {
    if (SemaRef.Context.getTypeSize(QT) / 8 > 16)
      return false;
    return true;
  }

  if (const VectorType *VT = QT->getAs<VectorType>()) {
    int ArraySize = VT->getNumElements();

    if (ArraySize > 4)
      return false;

    QualType ElTy = VT->getElementType();
    if (ElTy->isBooleanType())
      return false;

    if (SemaRef.Context.getTypeSize(QT) / 8 > 16)
      return false;
    return true;
  }

  return false;
}

bool SemaHLSL::IsScalarizedLayoutCompatible(QualType T1, QualType T2) const {
  if (T1.isNull() || T2.isNull())
    return false;

  T1 = T1.getCanonicalType().getUnqualifiedType();
  T2 = T2.getCanonicalType().getUnqualifiedType();

  // If both types are the same canonical type, they're obviously compatible.
  if (SemaRef.getASTContext().hasSameType(T1, T2))
    return true;

  llvm::SmallVector<QualType, 16> T1Types;
  BuildFlattenedTypeList(T1, T1Types);
  llvm::SmallVector<QualType, 16> T2Types;
  BuildFlattenedTypeList(T2, T2Types);

  // Check the flattened type list
  return llvm::equal(T1Types, T2Types,
                     [this](QualType LHS, QualType RHS) -> bool {
                       return SemaRef.IsLayoutCompatible(LHS, RHS);
                     });
}

bool SemaHLSL::CheckCompatibleParameterABI(FunctionDecl *New,
                                           FunctionDecl *Old) {
  if (New->getNumParams() != Old->getNumParams())
    return true;

  bool HadError = false;

  for (unsigned i = 0, e = New->getNumParams(); i != e; ++i) {
    ParmVarDecl *NewParam = New->getParamDecl(i);
    ParmVarDecl *OldParam = Old->getParamDecl(i);

    // HLSL parameter declarations for inout and out must match between
    // declarations. In HLSL inout and out are ambiguous at the call site,
    // but have different calling behavior, so you cannot overload a
    // method based on a difference between inout and out annotations.
    const auto *NDAttr = NewParam->getAttr<HLSLParamModifierAttr>();
    unsigned NSpellingIdx = (NDAttr ? NDAttr->getSpellingListIndex() : 0);
    const auto *ODAttr = OldParam->getAttr<HLSLParamModifierAttr>();
    unsigned OSpellingIdx = (ODAttr ? ODAttr->getSpellingListIndex() : 0);

    if (NSpellingIdx != OSpellingIdx) {
      SemaRef.Diag(NewParam->getLocation(),
                   diag::err_hlsl_param_qualifier_mismatch)
          << NDAttr << NewParam;
      SemaRef.Diag(OldParam->getLocation(), diag::note_previous_declaration_as)
          << ODAttr;
      HadError = true;
    }
  }
  return HadError;
}

ExprResult SemaHLSL::ActOnOutParamExpr(ParmVarDecl *Param, Expr *Arg) {
  assert(Param->hasAttr<HLSLParamModifierAttr>() &&
         "We should not get here without a parameter modifier expression");
  const auto *Attr = Param->getAttr<HLSLParamModifierAttr>();
  if (Attr->getABI() == ParameterABI::Ordinary)
    return ExprResult(Arg);

  bool IsInOut = Attr->getABI() == ParameterABI::HLSLInOut;
  if (!Arg->isLValue()) {
    SemaRef.Diag(Arg->getBeginLoc(), diag::error_hlsl_inout_lvalue)
        << Arg << (IsInOut ? 1 : 0);
    return ExprError();
  }

  ASTContext &Ctx = SemaRef.getASTContext();

  QualType Ty = Param->getType().getNonLValueExprType(Ctx);

  // HLSL allows implicit conversions from scalars to vectors, but not the
  // inverse, so we need to disallow `inout` with scalar->vector or
  // scalar->matrix conversions.
  if (Arg->getType()->isScalarType() != Ty->isScalarType()) {
    SemaRef.Diag(Arg->getBeginLoc(), diag::error_hlsl_inout_scalar_extension)
        << Arg << (IsInOut ? 1 : 0);
    return ExprError();
  }

  auto *ArgOpV = new (Ctx) OpaqueValueExpr(Param->getBeginLoc(), Arg->getType(),
                                           VK_LValue, OK_Ordinary, Arg);

  // Parameters are initialized via copy initialization. This allows for
  // overload resolution of argument constructors.
  InitializedEntity Entity =
      InitializedEntity::InitializeParameter(Ctx, Ty, false);
  ExprResult Res =
      SemaRef.PerformCopyInitialization(Entity, Param->getBeginLoc(), ArgOpV);
  if (Res.isInvalid())
    return ExprError();
  Expr *Base = Res.get();
  // After the cast, drop the reference type when creating the exprs.
  Ty = Ty.getNonLValueExprType(Ctx);
  auto *OpV = new (Ctx)
      OpaqueValueExpr(Param->getBeginLoc(), Ty, VK_LValue, OK_Ordinary, Base);

  // Writebacks are performed with `=` binary operator, which allows for
  // overload resolution on writeback result expressions.
  Res = SemaRef.ActOnBinOp(SemaRef.getCurScope(), Param->getBeginLoc(),
                           tok::equal, ArgOpV, OpV);

  if (Res.isInvalid())
    return ExprError();
  Expr *Writeback = Res.get();
  auto *OutExpr =
      HLSLOutArgExpr::Create(Ctx, Ty, ArgOpV, OpV, Writeback, IsInOut);

  return ExprResult(OutExpr);
}

QualType SemaHLSL::getInoutParameterType(QualType Ty) {
  // If HLSL gains support for references, all the cites that use this will need
  // to be updated with semantic checking to produce errors for
  // pointers/references.
  assert(!Ty->isReferenceType() &&
         "Pointer and reference types cannot be inout or out parameters");
  Ty = SemaRef.getASTContext().getLValueReferenceType(Ty);
  Ty.addRestrict();
  return Ty;
}

void SemaHLSL::ActOnVariableDeclarator(VarDecl *VD) {
  if (VD->hasGlobalStorage()) {
    // make sure the declaration has a complete type
    if (SemaRef.RequireCompleteType(
            VD->getLocation(),
            SemaRef.getASTContext().getBaseElementType(VD->getType()),
            diag::err_typecheck_decl_incomplete_type)) {
      VD->setInvalidDecl();
      return;
    }

    // find all resources on decl
    if (VD->getType()->isHLSLIntangibleType())
      collectResourcesOnVarDecl(VD);

    // process explicit bindings
    processExplicitBindingsOnDecl(VD);
  }
}

// Walks though the global variable declaration, collects all resource binding
// requirements and adds them to Bindings
void SemaHLSL::collectResourcesOnVarDecl(VarDecl *VD) {
  assert(VD->hasGlobalStorage() && VD->getType()->isHLSLIntangibleType() &&
         "expected global variable that contains HLSL resource");

  // Cbuffers and Tbuffers are HLSLBufferDecl types
  if (const HLSLBufferDecl *CBufferOrTBuffer = dyn_cast<HLSLBufferDecl>(VD)) {
    Bindings.addDeclBindingInfo(VD, CBufferOrTBuffer->isCBuffer()
                                        ? ResourceClass::CBuffer
                                        : ResourceClass::SRV);
    return;
  }

  // Unwrap arrays
  // FIXME: Calculate array size while unwrapping
  const Type *Ty = VD->getType()->getUnqualifiedDesugaredType();
  while (Ty->isConstantArrayType()) {
    const ConstantArrayType *CAT = cast<ConstantArrayType>(Ty);
    Ty = CAT->getElementType()->getUnqualifiedDesugaredType();
  }

  // Resource (or array of resources)
  if (const HLSLAttributedResourceType *AttrResType =
          HLSLAttributedResourceType::findHandleTypeOnResource(Ty)) {
    Bindings.addDeclBindingInfo(VD, AttrResType->getAttrs().ResourceClass);
    return;
  }

  // User defined record type
  if (const RecordType *RT = dyn_cast<RecordType>(Ty))
    collectResourcesOnUserRecordDecl(VD, RT);
}

// Walks though the explicit resource binding attributes on the declaration,
// and makes sure there is a resource that matched the binding and updates
// DeclBindingInfoLists
void SemaHLSL::processExplicitBindingsOnDecl(VarDecl *VD) {
  assert(VD->hasGlobalStorage() && "expected global variable");

  for (Attr *A : VD->attrs()) {
    HLSLResourceBindingAttr *RBA = dyn_cast<HLSLResourceBindingAttr>(A);
    if (!RBA)
      continue;

    RegisterType RT = RBA->getRegisterType();
    assert(RT != RegisterType::I && "invalid or obsolete register type should "
                                    "never have an attribute created");

    if (RT == RegisterType::C) {
      if (Bindings.hasBindingInfoForDecl(VD))
        SemaRef.Diag(VD->getLocation(),
                     diag::warn_hlsl_user_defined_type_missing_member)
            << static_cast<int>(RT);
      continue;
    }

    // Find DeclBindingInfo for this binding and update it, or report error
    // if it does not exist (user type does to contain resources with the
    // expected resource class).
    ResourceClass RC = getResourceClass(RT);
    if (DeclBindingInfo *BI = Bindings.getDeclBindingInfo(VD, RC)) {
      // update binding info
      BI->setBindingAttribute(RBA, BindingType::Explicit);
    } else {
      SemaRef.Diag(VD->getLocation(),
                   diag::warn_hlsl_user_defined_type_missing_member)
          << static_cast<int>(RT);
    }
  }
}
