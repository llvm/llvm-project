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
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/TargetParser/Triple.h"
#include <iterator>

using namespace clang;

SemaHLSL::SemaHLSL(Sema &S) : SemaBase(S) {}

Decl *SemaHLSL::ActOnStartBuffer(Scope *BufferScope, bool CBuffer,
                                 SourceLocation KwLoc, IdentifierInfo *Ident,
                                 SourceLocation IdentLoc,
                                 SourceLocation LBrace) {
  // For anonymous namespace, take the location of the left brace.
  DeclContext *LexicalParent = SemaRef.getCurLexicalContext();
  HLSLBufferDecl *Result = HLSLBufferDecl::Create(
      getASTContext(), LexicalParent, CBuffer, KwLoc, Ident, IdentLoc, LBrace);

  SemaRef.PushOnScopeChains(Result, BufferScope);
  SemaRef.PushDeclContext(BufferScope, Result);

  return Result;
}

// Calculate the size of a legacy cbuffer type based on
// https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-packing-rules
static unsigned calculateLegacyCbufferSize(const ASTContext &Context,
                                           QualType T) {
  unsigned Size = 0;
  constexpr unsigned CBufferAlign = 128;
  if (const RecordType *RT = T->getAs<RecordType>()) {
    const RecordDecl *RD = RT->getDecl();
    for (const FieldDecl *Field : RD->fields()) {
      QualType Ty = Field->getType();
      unsigned FieldSize = calculateLegacyCbufferSize(Context, Ty);
      unsigned FieldAlign = 32;
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
    Size = Context.getTypeSize(T);
  }
  return Size;
}

void SemaHLSL::ActOnFinishBuffer(Decl *Dcl, SourceLocation RBrace) {
  auto *BufDecl = cast<HLSLBufferDecl>(Dcl);
  BufDecl->setRBraceLoc(RBrace);

  // Validate packoffset.
  llvm::SmallVector<std::pair<VarDecl *, HLSLPackOffsetAttr *>> PackOffsetVec;
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

  if (HasPackOffset && HasNonPackOffset)
    Diag(BufDecl->getLocation(), diag::warn_hlsl_packoffset_mix);

  if (HasPackOffset) {
    ASTContext &Context = getASTContext();
    // Make sure no overlap in packoffset.
    // Sort PackOffsetVec by offset.
    std::sort(PackOffsetVec.begin(), PackOffsetVec.end(),
              [](const std::pair<VarDecl *, HLSLPackOffsetAttr *> &LHS,
                 const std::pair<VarDecl *, HLSLPackOffsetAttr *> &RHS) {
                return LHS.second->getOffset() < RHS.second->getOffset();
              });

    for (unsigned i = 0; i < PackOffsetVec.size() - 1; i++) {
      VarDecl *Var = PackOffsetVec[i].first;
      HLSLPackOffsetAttr *Attr = PackOffsetVec[i].second;
      unsigned Size = calculateLegacyCbufferSize(Context, Var->getType());
      unsigned Begin = Attr->getOffset() * 32;
      unsigned End = Begin + Size;
      unsigned NextBegin = PackOffsetVec[i + 1].second->getOffset() * 32;
      if (End > NextBegin) {
        VarDecl *NextVar = PackOffsetVec[i + 1].first;
        Diag(NextVar->getLocation(), diag::err_hlsl_packoffset_overlap)
            << NextVar << Var;
      }
    }
  }

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
    break;

  case llvm::Triple::Compute:
  case llvm::Triple::Amplification:
  case llvm::Triple::Mesh:
    if (!FD->hasAttr<HLSLNumThreadsAttr>()) {
      Diag(FD->getLocation(), diag::err_hlsl_missing_numthreads)
          << llvm::Triple::getEnvironmentTypeName(ST);
      FD->setInvalidDecl();
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

static bool isLegalTypeForHLSLSV_DispatchThreadID(QualType T) {
  if (!T->hasUnsignedIntegerRepresentation())
    return false;
  if (const auto *VT = T->getAs<VectorType>())
    return VT->getNumElements() <= 3;
  return true;
}

void SemaHLSL::handleSV_DispatchThreadIDAttr(Decl *D, const ParsedAttr &AL) {
  // FIXME: support semantic on field.
  // See https://github.com/llvm/llvm-project/issues/57889.
  if (isa<FieldDecl>(D)) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_ast_node)
        << AL << "parameter";
    return;
  }

  auto *VD = cast<ValueDecl>(D);
  if (!isLegalTypeForHLSLSV_DispatchThreadID(VD->getType())) {
    Diag(AL.getLoc(), diag::err_hlsl_attr_invalid_type)
        << AL << "uint/uint2/uint3";
    return;
  }

  D->addAttr(::new (getASTContext())
                 HLSLSV_DispatchThreadIDAttr(getASTContext(), AL));
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

void SemaHLSL::handleResourceBindingAttr(Decl *D, const ParsedAttr &AL) {
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
  if (AL.getNumArgs() == 2) {
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

  // Validate.
  if (!Slot.empty()) {
    switch (Slot[0]) {
    case 'u':
    case 'b':
    case 's':
    case 't':
      break;
    default:
      Diag(ArgLoc, diag::err_hlsl_unsupported_register_type)
          << Slot.substr(0, 1);
      return;
    }

    StringRef SlotNum = Slot.substr(1);
    unsigned Num = 0;
    if (SlotNum.getAsInteger(10, Num)) {
      Diag(ArgLoc, diag::err_hlsl_unsupported_register_number);
      return;
    }
  }

  if (!Space.starts_with("space")) {
    Diag(SpaceArgLoc, diag::err_hlsl_expected_space) << Space;
    return;
  }
  StringRef SpaceNum = Space.substr(5);
  unsigned Num = 0;
  if (SpaceNum.getAsInteger(10, Num)) {
    Diag(SpaceArgLoc, diag::err_hlsl_expected_space) << Space;
    return;
  }

  // FIXME: check reg type match decl. Issue
  // https://github.com/llvm/llvm-project/issues/57886.
  HLSLResourceBindingAttr *NewAttr =
      HLSLResourceBindingAttr::Create(getASTContext(), Slot, Space, AL);
  if (NewAttr)
    D->addAttr(NewAttr);
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
class DiagnoseHLSLAvailability
    : public RecursiveASTVisitor<DiagnoseHLSLAvailability> {

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
    unsigned &ScannedStages = ScannedDecls.getOrInsertDefault(FD);
    ScannedStages |= CurrentShaderStageBit;
  }

  unsigned GetScannedStages(const FunctionDecl *FD) {
    return ScannedDecls.getOrInsertDefault(FD);
  }

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
  DiagnoseHLSLAvailability(Sema &SemaRef) : SemaRef(SemaRef) {}

  // AST traversal methods
  void RunOnTranslationUnit(const TranslationUnitDecl *TU);
  void RunOnFunction(const FunctionDecl *FD);

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(DRE->getDecl());
    if (FD)
      HandleFunctionOrMethodRef(FD, DRE);
    return true;
  }

  bool VisitMemberExpr(MemberExpr *ME) {
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
  // or an exported function.
  for (auto &D : TU->decls()) {
    const FunctionDecl *FD = llvm::dyn_cast<FunctionDecl>(D);
    if (!FD || !FD->isThisDeclarationADefinition())
      continue;

    // shader entry point
    auto ShaderAttr = FD->getAttr<HLSLShaderAttr>();
    if (ShaderAttr) {
      SetShaderStageContext(ShaderAttr->getType());
      RunOnFunction(FD);
      continue;
    }
    // exported library function with definition
    // FIXME: tracking issue #92073
#if 0
    if (FD->getFormalLinkage() == Linkage::External) {
      SetUnknownShaderStageContext();
      RunOnFunction(FD);
    }
#endif
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
    const FunctionDecl *FD = DeclsToScan.back();
    DeclsToScan.pop_back();

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
