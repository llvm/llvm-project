//===--- SemaDeclSpec.cpp - Declaration Specifier Semantic Analysis -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for declaration specifiers.
//
//===----------------------------------------------------------------------===//

#include "clang/Parse/ParseDiagnostic.h" // FIXME: remove this back-dependency!
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/LocInfoType.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstring>
using namespace clang;


static DiagnosticBuilder Diag(DiagnosticsEngine &D, SourceLocation Loc,
                              unsigned DiagID) {
  return D.Report(Loc, DiagID);
}


void UnqualifiedId::setTemplateId(TemplateIdAnnotation *TemplateId) {
  assert(TemplateId && "NULL template-id annotation?");
  Kind = IK_TemplateId;
  this->TemplateId = TemplateId;
  StartLocation = TemplateId->TemplateNameLoc;
  EndLocation = TemplateId->RAngleLoc;
}

void UnqualifiedId::setConstructorTemplateId(TemplateIdAnnotation *TemplateId) {
  assert(TemplateId && "NULL template-id annotation?");
  Kind = IK_ConstructorTemplateId;
  this->TemplateId = TemplateId;
  StartLocation = TemplateId->TemplateNameLoc;
  EndLocation = TemplateId->RAngleLoc;
}

void CXXScopeSpec::Extend(ASTContext &Context, SourceLocation TemplateKWLoc, 
                          TypeLoc TL, SourceLocation ColonColonLoc) {
  Builder.Extend(Context, TemplateKWLoc, TL, ColonColonLoc);
  if (Range.getBegin().isInvalid())
    Range.setBegin(TL.getBeginLoc());
  Range.setEnd(ColonColonLoc);

  assert(Range == Builder.getSourceRange() &&
         "NestedNameSpecifierLoc range computation incorrect");
}

void CXXScopeSpec::Extend(ASTContext &Context, IdentifierInfo *Identifier,
                          SourceLocation IdentifierLoc, 
                          SourceLocation ColonColonLoc) {
  Builder.Extend(Context, Identifier, IdentifierLoc, ColonColonLoc);
  
  if (Range.getBegin().isInvalid())
    Range.setBegin(IdentifierLoc);
  Range.setEnd(ColonColonLoc);
  
  assert(Range == Builder.getSourceRange() &&
         "NestedNameSpecifierLoc range computation incorrect");
}

void CXXScopeSpec::Extend(ASTContext &Context, NamespaceDecl *Namespace,
                          SourceLocation NamespaceLoc, 
                          SourceLocation ColonColonLoc) {
  Builder.Extend(Context, Namespace, NamespaceLoc, ColonColonLoc);
  
  if (Range.getBegin().isInvalid())
    Range.setBegin(NamespaceLoc);
  Range.setEnd(ColonColonLoc);

  assert(Range == Builder.getSourceRange() &&
         "NestedNameSpecifierLoc range computation incorrect");
}

void CXXScopeSpec::Extend(ASTContext &Context, NamespaceAliasDecl *Alias,
                          SourceLocation AliasLoc, 
                          SourceLocation ColonColonLoc) {
  Builder.Extend(Context, Alias, AliasLoc, ColonColonLoc);
  
  if (Range.getBegin().isInvalid())
    Range.setBegin(AliasLoc);
  Range.setEnd(ColonColonLoc);

  assert(Range == Builder.getSourceRange() &&
         "NestedNameSpecifierLoc range computation incorrect");
}

void CXXScopeSpec::MakeGlobal(ASTContext &Context, 
                              SourceLocation ColonColonLoc) {
  Builder.MakeGlobal(Context, ColonColonLoc);
  
  Range = SourceRange(ColonColonLoc);
  
  assert(Range == Builder.getSourceRange() &&
         "NestedNameSpecifierLoc range computation incorrect");
}

void CXXScopeSpec::MakeTrivial(ASTContext &Context, 
                               NestedNameSpecifier *Qualifier, SourceRange R) {
  Builder.MakeTrivial(Context, Qualifier, R);
  Range = R;
}

void CXXScopeSpec::Adopt(NestedNameSpecifierLoc Other) {
  if (!Other) {
    Range = SourceRange();
    Builder.Clear();
    return;
  }

  Range = Other.getSourceRange();
  Builder.Adopt(Other);
}

SourceLocation CXXScopeSpec::getLastQualifierNameLoc() const {
  if (!Builder.getRepresentation())
    return SourceLocation();
  return Builder.getTemporary().getLocalBeginLoc();
}

NestedNameSpecifierLoc 
CXXScopeSpec::getWithLocInContext(ASTContext &Context) const {
  if (!Builder.getRepresentation())
    return NestedNameSpecifierLoc();
  
  return Builder.getWithLocInContext(Context);
}

/// DeclaratorChunk::getFunction - Return a DeclaratorChunk for a function.
/// "TheDeclarator" is the declarator that this will be added to.
DeclaratorChunk DeclaratorChunk::getFunction(bool hasProto, bool isVariadic,
                                             SourceLocation EllipsisLoc,
                                             ParamInfo *ArgInfo,
                                             unsigned NumArgs,
                                             unsigned TypeQuals,
                                             bool RefQualifierIsLvalueRef,
                                             SourceLocation RefQualifierLoc,
                                             SourceLocation ConstQualifierLoc,
                                             SourceLocation
                                                 VolatileQualifierLoc,
                                             SourceLocation MutableLoc,
                                             ExceptionSpecificationType
                                                 ESpecType,
                                             SourceLocation ESpecLoc,
                                             ParsedType *Exceptions,
                                             SourceRange *ExceptionRanges,
                                             unsigned NumExceptions,
                                             Expr *NoexceptExpr,
                                             SourceLocation LocalRangeBegin,
                                             SourceLocation LocalRangeEnd,
                                             Declarator &TheDeclarator,
                                             ParsedType TrailingReturnType) {
  DeclaratorChunk I;
  I.Kind                        = Function;
  I.Loc                         = LocalRangeBegin;
  I.EndLoc                      = LocalRangeEnd;
  I.Fun.AttrList                = 0;
  I.Fun.hasPrototype            = hasProto;
  I.Fun.isVariadic              = isVariadic;
  I.Fun.EllipsisLoc             = EllipsisLoc.getRawEncoding();
  I.Fun.DeleteArgInfo           = false;
  I.Fun.TypeQuals               = TypeQuals;
  I.Fun.NumArgs                 = NumArgs;
  I.Fun.ArgInfo                 = 0;
  I.Fun.RefQualifierIsLValueRef = RefQualifierIsLvalueRef;
  I.Fun.RefQualifierLoc         = RefQualifierLoc.getRawEncoding();
  I.Fun.ConstQualifierLoc       = ConstQualifierLoc.getRawEncoding();
  I.Fun.VolatileQualifierLoc    = VolatileQualifierLoc.getRawEncoding();
  I.Fun.MutableLoc              = MutableLoc.getRawEncoding();
  I.Fun.ExceptionSpecType       = ESpecType;
  I.Fun.ExceptionSpecLoc        = ESpecLoc.getRawEncoding();
  I.Fun.NumExceptions           = 0;
  I.Fun.Exceptions              = 0;
  I.Fun.NoexceptExpr            = 0;
  I.Fun.TrailingReturnType   = TrailingReturnType.getAsOpaquePtr();

  // new[] an argument array if needed.
  if (NumArgs) {
    // If the 'InlineParams' in Declarator is unused and big enough, put our
    // parameter list there (in an effort to avoid new/delete traffic).  If it
    // is already used (consider a function returning a function pointer) or too
    // small (function taking too many arguments), go to the heap.
    if (!TheDeclarator.InlineParamsUsed &&
        NumArgs <= llvm::array_lengthof(TheDeclarator.InlineParams)) {
      I.Fun.ArgInfo = TheDeclarator.InlineParams;
      I.Fun.DeleteArgInfo = false;
      TheDeclarator.InlineParamsUsed = true;
    } else {
      I.Fun.ArgInfo = new DeclaratorChunk::ParamInfo[NumArgs];
      I.Fun.DeleteArgInfo = true;
    }
    memcpy(I.Fun.ArgInfo, ArgInfo, sizeof(ArgInfo[0])*NumArgs);
  }

  // Check what exception specification information we should actually store.
  switch (ESpecType) {
  default: break; // By default, save nothing.
  case EST_Dynamic:
    // new[] an exception array if needed
    if (NumExceptions) {
      I.Fun.NumExceptions = NumExceptions;
      I.Fun.Exceptions = new DeclaratorChunk::TypeAndRange[NumExceptions];
      for (unsigned i = 0; i != NumExceptions; ++i) {
        I.Fun.Exceptions[i].Ty = Exceptions[i];
        I.Fun.Exceptions[i].Range = ExceptionRanges[i];
      }
    }
    break;

  case EST_ComputedNoexcept:
    I.Fun.NoexceptExpr = NoexceptExpr;
    break;
  }
  return I;
}

bool Declarator::isDeclarationOfFunction() const {
  for (unsigned i = 0, i_end = DeclTypeInfo.size(); i < i_end; ++i) {
    switch (DeclTypeInfo[i].Kind) {
    case DeclaratorChunk::Function:
      return true;
    case DeclaratorChunk::Paren:
      continue;
    case DeclaratorChunk::Pointer:
    case DeclaratorChunk::Reference:
    case DeclaratorChunk::Array:
    case DeclaratorChunk::BlockPointer:
    case DeclaratorChunk::MemberPointer:
      return false;
    }
    llvm_unreachable("Invalid type chunk");
  }
  
  switch (DS.getTypeSpecType()) {
    case TST_atomic:
    case TST_auto:
    case TST_bool:
    case TST_char:
    case TST_char16:
    case TST_char32:
    case TST_class:
    case TST_decimal128:
    case TST_decimal32:
    case TST_decimal64:
    case TST_double:
    case TST_enum:
    case TST_error:
    case TST_float:
    case TST_half:
    case TST_int:
    case TST_int128:
    case TST_struct:
    case TST_union:
    case TST_unknown_anytype:
    case TST_unspecified:
    case TST_void:
    case TST_wchar:
      return false;

    case TST_decltype:
    case TST_typeofExpr:
      if (Expr *E = DS.getRepAsExpr())
        return E->getType()->isFunctionType();
      return false;
     
    case TST_underlyingType:
    case TST_typename:
    case TST_typeofType: {
      QualType QT = DS.getRepAsType().get();
      if (QT.isNull())
        return false;
      
      if (const LocInfoType *LIT = dyn_cast<LocInfoType>(QT))
        QT = LIT->getType();

      if (QT.isNull())
        return false;
        
      return QT->isFunctionType();
    }
  }

  llvm_unreachable("Invalid TypeSpecType!");
}

/// getParsedSpecifiers - Return a bitmask of which flavors of specifiers this
/// declaration specifier includes.
///
unsigned DeclSpec::getParsedSpecifiers() const {
  unsigned Res = 0;
  if (StorageClassSpec != SCS_unspecified ||
      SCS_thread_specified)
    Res |= PQ_StorageClassSpecifier;

  if (TypeQualifiers != TQ_unspecified)
    Res |= PQ_TypeQualifier;

  if (hasTypeSpecifier())
    Res |= PQ_TypeSpecifier;

  if (FS_inline_specified || FS_virtual_specified || FS_explicit_specified)
    Res |= PQ_FunctionSpecifier;
  return Res;
}

template <class T> static bool BadSpecifier(T TNew, T TPrev,
                                            const char *&PrevSpec,
                                            unsigned &DiagID) {
  PrevSpec = DeclSpec::getSpecifierName(TPrev);
  DiagID = (TNew == TPrev ? diag::ext_duplicate_declspec
            : diag::err_invalid_decl_spec_combination);
  return true;
}

const char *DeclSpec::getSpecifierName(DeclSpec::SCS S) {
  switch (S) {
  case DeclSpec::SCS_unspecified: return "unspecified";
  case DeclSpec::SCS_typedef:     return "typedef";
  case DeclSpec::SCS_extern:      return "extern";
  case DeclSpec::SCS_static:      return "static";
  case DeclSpec::SCS_auto:        return "auto";
  case DeclSpec::SCS_register:    return "register";
  case DeclSpec::SCS_private_extern: return "__private_extern__";
  case DeclSpec::SCS_mutable:     return "mutable";
  }
  llvm_unreachable("Unknown typespec!");
}

const char *DeclSpec::getSpecifierName(TSW W) {
  switch (W) {
  case TSW_unspecified: return "unspecified";
  case TSW_short:       return "short";
  case TSW_long:        return "long";
  case TSW_longlong:    return "long long";
  }
  llvm_unreachable("Unknown typespec!");
}

const char *DeclSpec::getSpecifierName(TSC C) {
  switch (C) {
  case TSC_unspecified: return "unspecified";
  case TSC_imaginary:   return "imaginary";
  case TSC_complex:     return "complex";
  }
  llvm_unreachable("Unknown typespec!");
}


const char *DeclSpec::getSpecifierName(TSS S) {
  switch (S) {
  case TSS_unspecified: return "unspecified";
  case TSS_signed:      return "signed";
  case TSS_unsigned:    return "unsigned";
  }
  llvm_unreachable("Unknown typespec!");
}

const char *DeclSpec::getSpecifierName(DeclSpec::TST T) {
  switch (T) {
  case DeclSpec::TST_unspecified: return "unspecified";
  case DeclSpec::TST_void:        return "void";
  case DeclSpec::TST_char:        return "char";
  case DeclSpec::TST_wchar:       return "wchar_t";
  case DeclSpec::TST_char16:      return "char16_t";
  case DeclSpec::TST_char32:      return "char32_t";
  case DeclSpec::TST_int:         return "int";
  case DeclSpec::TST_int128:      return "__int128";
  case DeclSpec::TST_half:        return "half";
  case DeclSpec::TST_float:       return "float";
  case DeclSpec::TST_double:      return "double";
  case DeclSpec::TST_bool:        return "_Bool";
  case DeclSpec::TST_decimal32:   return "_Decimal32";
  case DeclSpec::TST_decimal64:   return "_Decimal64";
  case DeclSpec::TST_decimal128:  return "_Decimal128";
  case DeclSpec::TST_enum:        return "enum";
  case DeclSpec::TST_class:       return "class";
  case DeclSpec::TST_union:       return "union";
  case DeclSpec::TST_struct:      return "struct";
  case DeclSpec::TST_typename:    return "type-name";
  case DeclSpec::TST_typeofType:
  case DeclSpec::TST_typeofExpr:  return "typeof";
  case DeclSpec::TST_auto:        return "auto";
  case DeclSpec::TST_decltype:    return "(decltype)";
  case DeclSpec::TST_underlyingType: return "__underlying_type";
  case DeclSpec::TST_unknown_anytype: return "__unknown_anytype";
  case DeclSpec::TST_atomic: return "_Atomic";
  case DeclSpec::TST_error:       return "(error)";
  }
  llvm_unreachable("Unknown typespec!");
}

const char *DeclSpec::getSpecifierName(TQ T) {
  switch (T) {
  case DeclSpec::TQ_unspecified: return "unspecified";
  case DeclSpec::TQ_const:       return "const";
  case DeclSpec::TQ_restrict:    return "restrict";
  case DeclSpec::TQ_volatile:    return "volatile";
  }
  llvm_unreachable("Unknown typespec!");
}

bool DeclSpec::SetStorageClassSpec(Sema &S, SCS SC, SourceLocation Loc,
                                   const char *&PrevSpec,
                                   unsigned &DiagID) {
  // OpenCL 1.1 6.8g: "The extern, static, auto and register storage-class
  // specifiers are not supported."
  // It seems sensible to prohibit private_extern too
  // The cl_clang_storage_class_specifiers extension enables support for
  // these storage-class specifiers.
  if (S.getLangOpts().OpenCL &&
      !S.getOpenCLOptions().cl_clang_storage_class_specifiers) {
    switch (SC) {
    case SCS_extern:
    case SCS_private_extern:
    case SCS_auto:
    case SCS_register:
    case SCS_static:
      DiagID   = diag::err_not_opencl_storage_class_specifier;
      PrevSpec = getSpecifierName(SC);
      return true;
    default:
      break;
    }
  }

  if (StorageClassSpec != SCS_unspecified) {
    // Maybe this is an attempt to use C++0x 'auto' outside of C++0x mode.
    bool isInvalid = true;
    if (TypeSpecType == TST_unspecified && S.getLangOpts().CPlusPlus) {
      if (SC == SCS_auto)
        return SetTypeSpecType(TST_auto, Loc, PrevSpec, DiagID);
      if (StorageClassSpec == SCS_auto) {
        isInvalid = SetTypeSpecType(TST_auto, StorageClassSpecLoc,
                                    PrevSpec, DiagID);
        assert(!isInvalid && "auto SCS -> TST recovery failed");
      }
    }

    // Changing storage class is allowed only if the previous one
    // was the 'extern' that is part of a linkage specification and
    // the new storage class is 'typedef'.
    if (isInvalid &&
        !(SCS_extern_in_linkage_spec &&
          StorageClassSpec == SCS_extern &&
          SC == SCS_typedef))
      return BadSpecifier(SC, (SCS)StorageClassSpec, PrevSpec, DiagID);
  }
  StorageClassSpec = SC;
  StorageClassSpecLoc = Loc;
  assert((unsigned)SC == StorageClassSpec && "SCS constants overflow bitfield");
  return false;
}

bool DeclSpec::SetStorageClassSpecThread(SourceLocation Loc,
                                         const char *&PrevSpec,
                                         unsigned &DiagID) {
  if (SCS_thread_specified) {
    PrevSpec = "__thread";
    DiagID = diag::ext_duplicate_declspec;
    return true;
  }
  SCS_thread_specified = true;
  SCS_threadLoc = Loc;
  return false;
}

/// These methods set the specified attribute of the DeclSpec, but return true
/// and ignore the request if invalid (e.g. "extern" then "auto" is
/// specified).
bool DeclSpec::SetTypeSpecWidth(TSW W, SourceLocation Loc,
                                const char *&PrevSpec,
                                unsigned &DiagID) {
  // Overwrite TSWLoc only if TypeSpecWidth was unspecified, so that
  // for 'long long' we will keep the source location of the first 'long'.
  if (TypeSpecWidth == TSW_unspecified)
    TSWLoc = Loc;
  // Allow turning long -> long long.
  else if (W != TSW_longlong || TypeSpecWidth != TSW_long)
    return BadSpecifier(W, (TSW)TypeSpecWidth, PrevSpec, DiagID);
  TypeSpecWidth = W;
  if (TypeAltiVecVector && !TypeAltiVecBool &&
      ((TypeSpecWidth == TSW_long) || (TypeSpecWidth == TSW_longlong))) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::warn_vector_long_decl_spec_combination;
    return true;
  }
  return false;
}

bool DeclSpec::SetTypeSpecComplex(TSC C, SourceLocation Loc,
                                  const char *&PrevSpec,
                                  unsigned &DiagID) {
  if (TypeSpecComplex != TSC_unspecified)
    return BadSpecifier(C, (TSC)TypeSpecComplex, PrevSpec, DiagID);
  TypeSpecComplex = C;
  TSCLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecSign(TSS S, SourceLocation Loc,
                               const char *&PrevSpec,
                               unsigned &DiagID) {
  if (TypeSpecSign != TSS_unspecified)
    return BadSpecifier(S, (TSS)TypeSpecSign, PrevSpec, DiagID);
  TypeSpecSign = S;
  TSSLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec,
                               unsigned &DiagID,
                               ParsedType Rep) {
  return SetTypeSpecType(T, Loc, Loc, PrevSpec, DiagID, Rep);
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation TagKwLoc,
                               SourceLocation TagNameLoc,
                               const char *&PrevSpec,
                               unsigned &DiagID,
                               ParsedType Rep) {
  assert(isTypeRep(T) && "T does not store a type");
  assert(Rep && "no type provided!");
  if (TypeSpecType != TST_unspecified) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_decl_spec_combination;
    return true;
  }
  TypeSpecType = T;
  TypeRep = Rep;
  TSTLoc = TagKwLoc;
  TSTNameLoc = TagNameLoc;
  TypeSpecOwned = false;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec,
                               unsigned &DiagID,
                               Expr *Rep) {
  assert(isExprRep(T) && "T does not store an expr");
  assert(Rep && "no expression provided!");
  if (TypeSpecType != TST_unspecified) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_decl_spec_combination;
    return true;
  }
  TypeSpecType = T;
  ExprRep = Rep;
  TSTLoc = Loc;
  TSTNameLoc = Loc;
  TypeSpecOwned = false;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec,
                               unsigned &DiagID,
                               Decl *Rep, bool Owned) {
  return SetTypeSpecType(T, Loc, Loc, PrevSpec, DiagID, Rep, Owned);
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation TagKwLoc,
                               SourceLocation TagNameLoc,
                               const char *&PrevSpec,
                               unsigned &DiagID,
                               Decl *Rep, bool Owned) {
  assert(isDeclRep(T) && "T does not store a decl");
  // Unlike the other cases, we don't assert that we actually get a decl.

  if (TypeSpecType != TST_unspecified) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_decl_spec_combination;
    return true;
  }
  TypeSpecType = T;
  DeclRep = Rep;
  TSTLoc = TagKwLoc;
  TSTNameLoc = TagNameLoc;
  TypeSpecOwned = Owned;
  return false;
}

bool DeclSpec::SetTypeSpecType(TST T, SourceLocation Loc,
                               const char *&PrevSpec,
                               unsigned &DiagID) {
  assert(!isDeclRep(T) && !isTypeRep(T) && !isExprRep(T) &&
         "rep required for these type-spec kinds!");
  if (TypeSpecType != TST_unspecified) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_decl_spec_combination;
    return true;
  }
  TSTLoc = Loc;
  TSTNameLoc = Loc;
  if (TypeAltiVecVector && (T == TST_bool) && !TypeAltiVecBool) {
    TypeAltiVecBool = true;
    return false;
  }
  TypeSpecType = T;
  TypeSpecOwned = false;
  if (TypeAltiVecVector && !TypeAltiVecBool && (TypeSpecType == TST_double)) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_vector_decl_spec;
    return true;
  }
  return false;
}

bool DeclSpec::SetTypeAltiVecVector(bool isAltiVecVector, SourceLocation Loc,
                          const char *&PrevSpec, unsigned &DiagID) {
  if (TypeSpecType != TST_unspecified) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_vector_decl_spec_combination;
    return true;
  }
  TypeAltiVecVector = isAltiVecVector;
  AltiVecLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeAltiVecPixel(bool isAltiVecPixel, SourceLocation Loc,
                          const char *&PrevSpec, unsigned &DiagID) {
  if (!TypeAltiVecVector || TypeAltiVecPixel ||
      (TypeSpecType != TST_unspecified)) {
    PrevSpec = DeclSpec::getSpecifierName((TST) TypeSpecType);
    DiagID = diag::err_invalid_pixel_decl_spec_combination;
    return true;
  }
  TypeAltiVecPixel = isAltiVecPixel;
  TSTLoc = Loc;
  TSTNameLoc = Loc;
  return false;
}

bool DeclSpec::SetTypeSpecError() {
  TypeSpecType = TST_error;
  TypeSpecOwned = false;
  TSTLoc = SourceLocation();
  TSTNameLoc = SourceLocation();
  return false;
}

bool DeclSpec::SetTypeQual(TQ T, SourceLocation Loc, const char *&PrevSpec,
                           unsigned &DiagID, const LangOptions &Lang) {
  // Duplicates turn into warnings pre-C99.
  if ((TypeQualifiers & T) && !Lang.C99)
    return BadSpecifier(T, T, PrevSpec, DiagID);
  TypeQualifiers |= T;

  switch (T) {
  default: llvm_unreachable("Unknown type qualifier!");
  case TQ_const:    TQ_constLoc = Loc; break;
  case TQ_restrict: TQ_restrictLoc = Loc; break;
  case TQ_volatile: TQ_volatileLoc = Loc; break;
  }
  return false;
}

bool DeclSpec::SetFunctionSpecInline(SourceLocation Loc, const char *&PrevSpec,
                                     unsigned &DiagID) {
  // 'inline inline' is ok.
  FS_inline_specified = true;
  FS_inlineLoc = Loc;
  return false;
}

bool DeclSpec::SetFunctionSpecVirtual(SourceLocation Loc, const char *&PrevSpec,
                                      unsigned &DiagID) {
  // 'virtual virtual' is ok.
  FS_virtual_specified = true;
  FS_virtualLoc = Loc;
  return false;
}

bool DeclSpec::SetFunctionSpecExplicit(SourceLocation Loc, const char *&PrevSpec,
                                       unsigned &DiagID) {
  // 'explicit explicit' is ok.
  FS_explicit_specified = true;
  FS_explicitLoc = Loc;
  return false;
}

bool DeclSpec::SetFriendSpec(SourceLocation Loc, const char *&PrevSpec,
                             unsigned &DiagID) {
  if (Friend_specified) {
    PrevSpec = "friend";
    DiagID = diag::ext_duplicate_declspec;
    return true;
  }

  Friend_specified = true;
  FriendLoc = Loc;
  return false;
}

bool DeclSpec::setModulePrivateSpec(SourceLocation Loc, const char *&PrevSpec,
                                    unsigned &DiagID) {
  if (isModulePrivateSpecified()) {
    PrevSpec = "__module_private__";
    DiagID = diag::ext_duplicate_declspec;
    return true;
  }
  
  ModulePrivateLoc = Loc;
  return false;
}

bool DeclSpec::SetConstexprSpec(SourceLocation Loc, const char *&PrevSpec,
                                unsigned &DiagID) {
  // 'constexpr constexpr' is ok.
  Constexpr_specified = true;
  ConstexprLoc = Loc;
  return false;
}

void DeclSpec::setProtocolQualifiers(Decl * const *Protos,
                                     unsigned NP,
                                     SourceLocation *ProtoLocs,
                                     SourceLocation LAngleLoc) {
  if (NP == 0) return;
  ProtocolQualifiers = new Decl*[NP];
  ProtocolLocs = new SourceLocation[NP];
  memcpy((void*)ProtocolQualifiers, Protos, sizeof(Decl*)*NP);
  memcpy(ProtocolLocs, ProtoLocs, sizeof(SourceLocation)*NP);
  NumProtocolQualifiers = NP;
  ProtocolLAngleLoc = LAngleLoc;
}

void DeclSpec::SaveWrittenBuiltinSpecs() {
  writtenBS.Sign = getTypeSpecSign();
  writtenBS.Width = getTypeSpecWidth();
  writtenBS.Type = getTypeSpecType();
  // Search the list of attributes for the presence of a mode attribute.
  writtenBS.ModeAttr = false;
  AttributeList* attrs = getAttributes().getList();
  while (attrs) {
    if (attrs->getKind() == AttributeList::AT_mode) {
      writtenBS.ModeAttr = true;
      break;
    }
    attrs = attrs->getNext();
  }
}

void DeclSpec::SaveStorageSpecifierAsWritten() {
  if (SCS_extern_in_linkage_spec && StorageClassSpec == SCS_extern)
    // If 'extern' is part of a linkage specification,
    // then it is not a storage class "as written".
    StorageClassSpecAsWritten = SCS_unspecified;
  else
    StorageClassSpecAsWritten = StorageClassSpec;
}

/// Finish - This does final analysis of the declspec, rejecting things like
/// "_Imaginary" (lacking an FP type).  This returns a diagnostic to issue or
/// diag::NUM_DIAGNOSTICS if there is no error.  After calling this method,
/// DeclSpec is guaranteed self-consistent, even if an error occurred.
void DeclSpec::Finish(DiagnosticsEngine &D, Preprocessor &PP) {
  // Before possibly changing their values, save specs as written.
  SaveWrittenBuiltinSpecs();
  SaveStorageSpecifierAsWritten();

  // Check the type specifier components first.

  // Validate and finalize AltiVec vector declspec.
  if (TypeAltiVecVector) {
    if (TypeAltiVecBool) {
      // Sign specifiers are not allowed with vector bool. (PIM 2.1)
      if (TypeSpecSign != TSS_unspecified) {
        Diag(D, TSSLoc, diag::err_invalid_vector_bool_decl_spec)
          << getSpecifierName((TSS)TypeSpecSign);
      }

      // Only char/int are valid with vector bool. (PIM 2.1)
      if (((TypeSpecType != TST_unspecified) && (TypeSpecType != TST_char) &&
           (TypeSpecType != TST_int)) || TypeAltiVecPixel) {
        Diag(D, TSTLoc, diag::err_invalid_vector_bool_decl_spec)
          << (TypeAltiVecPixel ? "__pixel" :
                                 getSpecifierName((TST)TypeSpecType));
      }

      // Only 'short' is valid with vector bool. (PIM 2.1)
      if ((TypeSpecWidth != TSW_unspecified) && (TypeSpecWidth != TSW_short))
        Diag(D, TSWLoc, diag::err_invalid_vector_bool_decl_spec)
          << getSpecifierName((TSW)TypeSpecWidth);

      // Elements of vector bool are interpreted as unsigned. (PIM 2.1)
      if ((TypeSpecType == TST_char) || (TypeSpecType == TST_int) ||
          (TypeSpecWidth != TSW_unspecified))
        TypeSpecSign = TSS_unsigned;
    }

    if (TypeAltiVecPixel) {
      //TODO: perform validation
      TypeSpecType = TST_int;
      TypeSpecSign = TSS_unsigned;
      TypeSpecWidth = TSW_short;
      TypeSpecOwned = false;
    }
  }

  // signed/unsigned are only valid with int/char/wchar_t.
  if (TypeSpecSign != TSS_unspecified) {
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // unsigned -> unsigned int, signed -> signed int.
    else if (TypeSpecType != TST_int  && TypeSpecType != TST_int128 &&
             TypeSpecType != TST_char && TypeSpecType != TST_wchar) {
      Diag(D, TSSLoc, diag::err_invalid_sign_spec)
        << getSpecifierName((TST)TypeSpecType);
      // signed double -> double.
      TypeSpecSign = TSS_unspecified;
    }
  }

  // Validate the width of the type.
  switch (TypeSpecWidth) {
  case TSW_unspecified: break;
  case TSW_short:    // short int
  case TSW_longlong: // long long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int; // short -> short int, long long -> long long int.
    else if (TypeSpecType != TST_int) {
      Diag(D, TSWLoc,
           TypeSpecWidth == TSW_short ? diag::err_invalid_short_spec
                                      : diag::err_invalid_longlong_spec)
        <<  getSpecifierName((TST)TypeSpecType);
      TypeSpecType = TST_int;
      TypeSpecOwned = false;
    }
    break;
  case TSW_long:  // long double, long int
    if (TypeSpecType == TST_unspecified)
      TypeSpecType = TST_int;  // long -> long int.
    else if (TypeSpecType != TST_int && TypeSpecType != TST_double) {
      Diag(D, TSWLoc, diag::err_invalid_long_spec)
        << getSpecifierName((TST)TypeSpecType);
      TypeSpecType = TST_int;
      TypeSpecOwned = false;
    }
    break;
  }

  // TODO: if the implementation does not implement _Complex or _Imaginary,
  // disallow their use.  Need information about the backend.
  if (TypeSpecComplex != TSC_unspecified) {
    if (TypeSpecType == TST_unspecified) {
      Diag(D, TSCLoc, diag::ext_plain_complex)
        << FixItHint::CreateInsertion(
                              PP.getLocForEndOfToken(getTypeSpecComplexLoc()),
                                                 " double");
      TypeSpecType = TST_double;   // _Complex -> _Complex double.
    } else if (TypeSpecType == TST_int || TypeSpecType == TST_char) {
      // Note that this intentionally doesn't include _Complex _Bool.
      if (!PP.getLangOpts().CPlusPlus)
        Diag(D, TSTLoc, diag::ext_integer_complex);
    } else if (TypeSpecType != TST_float && TypeSpecType != TST_double) {
      Diag(D, TSCLoc, diag::err_invalid_complex_spec)
        << getSpecifierName((TST)TypeSpecType);
      TypeSpecComplex = TSC_unspecified;
    }
  }

  // If no type specifier was provided and we're parsing a language where
  // the type specifier is not optional, but we got 'auto' as a storage
  // class specifier, then assume this is an attempt to use C++0x's 'auto'
  // type specifier.
  // FIXME: Does Microsoft really support implicit int in C++?
  if (PP.getLangOpts().CPlusPlus && !PP.getLangOpts().MicrosoftExt &&
      TypeSpecType == TST_unspecified && StorageClassSpec == SCS_auto) {
    TypeSpecType = TST_auto;
    StorageClassSpec = StorageClassSpecAsWritten = SCS_unspecified;
    TSTLoc = TSTNameLoc = StorageClassSpecLoc;
    StorageClassSpecLoc = SourceLocation();
  }
  // Diagnose if we've recovered from an ill-formed 'auto' storage class
  // specifier in a pre-C++0x dialect of C++.
  if (!PP.getLangOpts().CPlusPlus0x && TypeSpecType == TST_auto)
    Diag(D, TSTLoc, diag::ext_auto_type_specifier);
  if (PP.getLangOpts().CPlusPlus && !PP.getLangOpts().CPlusPlus0x &&
      StorageClassSpec == SCS_auto)
    Diag(D, StorageClassSpecLoc, diag::warn_auto_storage_class)
      << FixItHint::CreateRemoval(StorageClassSpecLoc);
  if (TypeSpecType == TST_char16 || TypeSpecType == TST_char32)
    Diag(D, TSTLoc, diag::warn_cxx98_compat_unicode_type)
      << (TypeSpecType == TST_char16 ? "char16_t" : "char32_t");
  if (Constexpr_specified)
    Diag(D, ConstexprLoc, diag::warn_cxx98_compat_constexpr);

  // C++ [class.friend]p6:
  //   No storage-class-specifier shall appear in the decl-specifier-seq
  //   of a friend declaration.
  if (isFriendSpecified() && getStorageClassSpec()) {
    DeclSpec::SCS SC = getStorageClassSpec();
    const char *SpecName = getSpecifierName(SC);

    SourceLocation SCLoc = getStorageClassSpecLoc();
    SourceLocation SCEndLoc = SCLoc.getLocWithOffset(strlen(SpecName));

    Diag(D, SCLoc, diag::err_friend_storage_spec)
      << SpecName
      << FixItHint::CreateRemoval(SourceRange(SCLoc, SCEndLoc));

    ClearStorageClassSpecs();
  }

  assert(!TypeSpecOwned || isDeclRep((TST) TypeSpecType));
 
  // Okay, now we can infer the real type.

  // TODO: return "auto function" and other bad things based on the real type.

  // 'data definition has no type or storage class'?
}

bool DeclSpec::isMissingDeclaratorOk() {
  TST tst = getTypeSpecType();
  return isDeclRep(tst) && getRepAsDecl() != 0 &&
    StorageClassSpec != DeclSpec::SCS_typedef;
}

void UnqualifiedId::clear() {
  Kind = IK_Identifier;
  Identifier = 0;
  StartLocation = SourceLocation();
  EndLocation = SourceLocation();
}

void UnqualifiedId::setOperatorFunctionId(SourceLocation OperatorLoc, 
                                          OverloadedOperatorKind Op,
                                          SourceLocation SymbolLocations[3]) {
  Kind = IK_OperatorFunctionId;
  StartLocation = OperatorLoc;
  EndLocation = OperatorLoc;
  OperatorFunctionId.Operator = Op;
  for (unsigned I = 0; I != 3; ++I) {
    OperatorFunctionId.SymbolLocations[I] = SymbolLocations[I].getRawEncoding();
    
    if (SymbolLocations[I].isValid())
      EndLocation = SymbolLocations[I];
  }
}

bool VirtSpecifiers::SetSpecifier(Specifier VS, SourceLocation Loc,
                                  const char *&PrevSpec) {
  LastLocation = Loc;
  
  if (Specifiers & VS) {
    PrevSpec = getSpecifierName(VS);
    return true;
  }

  Specifiers |= VS;

  switch (VS) {
  default: llvm_unreachable("Unknown specifier!");
  case VS_Override: VS_overrideLoc = Loc; break;
  case VS_Final:    VS_finalLoc = Loc; break;
  }

  return false;
}

const char *VirtSpecifiers::getSpecifierName(Specifier VS) {
  switch (VS) {
  default: llvm_unreachable("Unknown specifier");
  case VS_Override: return "override";
  case VS_Final: return "final";
  }
}
