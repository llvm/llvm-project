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
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
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
                          HLSLShaderAttr::ShaderType ShaderType) {
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

  StringRef Env = TargetInfo.getTriple().getEnvironmentName();
  HLSLShaderAttr::ShaderType ShaderType;
  if (HLSLShaderAttr::ConvertStrToShaderType(Env, ShaderType)) {
    if (const auto *Shader = FD->getAttr<HLSLShaderAttr>()) {
      // The entry point is already annotated - check that it matches the
      // triple.
      if (Shader->getType() != ShaderType) {
        Diag(Shader->getLocation(), diag::err_hlsl_entry_shader_attr_mismatch)
            << Shader;
        FD->setInvalidDecl();
      }
    } else {
      // Implicitly add the shader attribute if the entry function isn't
      // explicitly annotated.
      FD->addAttr(HLSLShaderAttr::CreateImplicit(getASTContext(), ShaderType,
                                                 FD->getBeginLoc()));
    }
  } else {
    switch (TargetInfo.getTriple().getEnvironment()) {
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
  HLSLShaderAttr::ShaderType ST = ShaderAttr->getType();

  switch (ST) {
  case HLSLShaderAttr::Pixel:
  case HLSLShaderAttr::Vertex:
  case HLSLShaderAttr::Geometry:
  case HLSLShaderAttr::Hull:
  case HLSLShaderAttr::Domain:
  case HLSLShaderAttr::RayGeneration:
  case HLSLShaderAttr::Intersection:
  case HLSLShaderAttr::AnyHit:
  case HLSLShaderAttr::ClosestHit:
  case HLSLShaderAttr::Miss:
  case HLSLShaderAttr::Callable:
    if (const auto *NT = FD->getAttr<HLSLNumThreadsAttr>()) {
      DiagnoseAttrStageMismatch(NT, ST,
                                {HLSLShaderAttr::Compute,
                                 HLSLShaderAttr::Amplification,
                                 HLSLShaderAttr::Mesh});
      FD->setInvalidDecl();
    }
    break;

  case HLSLShaderAttr::Compute:
  case HLSLShaderAttr::Amplification:
  case HLSLShaderAttr::Mesh:
    if (!FD->hasAttr<HLSLNumThreadsAttr>()) {
      Diag(FD->getLocation(), diag::err_hlsl_missing_numthreads)
          << HLSLShaderAttr::ConvertShaderTypeToStr(ST);
      FD->setInvalidDecl();
    }
    break;
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
  HLSLShaderAttr::ShaderType ST = ShaderAttr->getType();

  switch (AnnotationAttr->getKind()) {
  case attr::HLSLSV_DispatchThreadID:
  case attr::HLSLSV_GroupIndex:
    if (ST == HLSLShaderAttr::Compute)
      return;
    DiagnoseAttrStageMismatch(AnnotationAttr, ST, {HLSLShaderAttr::Compute});
    break;
  default:
    llvm_unreachable("Unknown HLSLAnnotationAttr");
  }
}

void SemaHLSL::DiagnoseAttrStageMismatch(
    const Attr *A, HLSLShaderAttr::ShaderType Stage,
    std::initializer_list<HLSLShaderAttr::ShaderType> AllowedStages) {
  SmallVector<StringRef, 8> StageStrings;
  llvm::transform(AllowedStages, std::back_inserter(StageStrings),
                  [](HLSLShaderAttr::ShaderType ST) {
                    return StringRef(
                        HLSLShaderAttr::ConvertShaderTypeToStr(ST));
                  });
  Diag(A->getLoc(), diag::err_hlsl_attr_unsupported_in_stage)
      << A << HLSLShaderAttr::ConvertShaderTypeToStr(Stage)
      << (AllowedStages.size() != 1) << join(StageStrings, ", ");
}
