//===--- TypePrinter.cpp - Pretty-Print Clang Types -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code to print types from Clang's type system.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"
using namespace clang;

namespace {
  /// \brief RAII object that enables printing of the ARC __strong lifetime
  /// qualifier.
  class IncludeStrongLifetimeRAII {
    PrintingPolicy &Policy;
    bool Old;
    
  public:
    explicit IncludeStrongLifetimeRAII(PrintingPolicy &Policy) 
      : Policy(Policy), Old(Policy.SuppressStrongLifetime) {
        if (!Policy.SuppressLifetimeQualifiers)
          Policy.SuppressStrongLifetime = false;
    }
    
    ~IncludeStrongLifetimeRAII() {
      Policy.SuppressStrongLifetime = Old;
    }
  };

  class ParamPolicyRAII {
    PrintingPolicy &Policy;
    bool Old;
    
  public:
    explicit ParamPolicyRAII(PrintingPolicy &Policy) 
      : Policy(Policy), Old(Policy.SuppressSpecifiers) {
      Policy.SuppressSpecifiers = false;
    }
    
    ~ParamPolicyRAII() {
      Policy.SuppressSpecifiers = Old;
    }
  };

  class ElaboratedTypePolicyRAII {
    PrintingPolicy &Policy;
    bool SuppressTagKeyword;
    bool SuppressScope;
    
  public:
    explicit ElaboratedTypePolicyRAII(PrintingPolicy &Policy) : Policy(Policy) {
      SuppressTagKeyword = Policy.SuppressTagKeyword;
      SuppressScope = Policy.SuppressScope;
      Policy.SuppressTagKeyword = true;
      Policy.SuppressScope = true;
    }
    
    ~ElaboratedTypePolicyRAII() {
      Policy.SuppressTagKeyword = SuppressTagKeyword;
      Policy.SuppressScope = SuppressScope;
    }
  };
  
  class TypePrinter {
    PrintingPolicy Policy;
    unsigned Indentation;
    bool HasEmptyPlaceHolder;
    bool InsideCCAttribute;

  public:
    explicit TypePrinter(const PrintingPolicy &Policy, unsigned Indentation = 0)
      : Policy(Policy), Indentation(Indentation),
        HasEmptyPlaceHolder(false), InsideCCAttribute(false) { }

    void print(const Type *ty, Qualifiers qs, raw_ostream &OS,
               StringRef PlaceHolder);
    void print(QualType T, raw_ostream &OS, StringRef PlaceHolder);

    static bool canPrefixQualifiers(const Type *T, bool &NeedARCStrongQualifier);
    void spaceBeforePlaceHolder(raw_ostream &OS);
    void printTypeSpec(NamedDecl *D, raw_ostream &OS);

    void printBefore(const Type *ty, Qualifiers qs, raw_ostream &OS);
    void printBefore(QualType T, raw_ostream &OS);
    void printAfter(const Type *ty, Qualifiers qs, raw_ostream &OS);
    void printAfter(QualType T, raw_ostream &OS);
    void AppendScope(DeclContext *DC, raw_ostream &OS);
    void printTag(TagDecl *T, raw_ostream &OS);
    void printFunctionAfter(const FunctionType::ExtInfo &Info, raw_ostream &OS);
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) \
    void print##CLASS##Before(const CLASS##Type *T, raw_ostream &OS); \
    void print##CLASS##After(const CLASS##Type *T, raw_ostream &OS);
#include "clang/AST/TypeNodes.def"
  };
}

static void AppendTypeQualList(raw_ostream &OS, unsigned TypeQuals,
                               bool HasRestrictKeyword) {
  bool appendSpace = false;
  if (TypeQuals & Qualifiers::Const) {
    OS << "const";
    appendSpace = true;
  }
  if (TypeQuals & Qualifiers::Volatile) {
    if (appendSpace) OS << ' ';
    OS << "volatile";
    appendSpace = true;
  }
  if (TypeQuals & Qualifiers::Restrict) {
    if (appendSpace) OS << ' ';
    if (HasRestrictKeyword) {
      OS << "restrict";
    } else {
      OS << "__restrict";
    }
  }
}

void TypePrinter::spaceBeforePlaceHolder(raw_ostream &OS) {
  if (!HasEmptyPlaceHolder)
    OS << ' ';
}

void TypePrinter::print(QualType t, raw_ostream &OS, StringRef PlaceHolder) {
  SplitQualType split = t.split();
  print(split.Ty, split.Quals, OS, PlaceHolder);
}

void TypePrinter::print(const Type *T, Qualifiers Quals, raw_ostream &OS,
                        StringRef PlaceHolder) {
  if (!T) {
    OS << "NULL TYPE";
    return;
  }

  SaveAndRestore<bool> PHVal(HasEmptyPlaceHolder, PlaceHolder.empty());

  printBefore(T, Quals, OS);
  OS << PlaceHolder;
  printAfter(T, Quals, OS);
}

bool TypePrinter::canPrefixQualifiers(const Type *T,
                                      bool &NeedARCStrongQualifier) {
  // CanPrefixQualifiers - We prefer to print type qualifiers before the type,
  // so that we get "const int" instead of "int const", but we can't do this if
  // the type is complex.  For example if the type is "int*", we *must* print
  // "int * const", printing "const int *" is different.  Only do this when the
  // type expands to a simple string.
  bool CanPrefixQualifiers = false;
  NeedARCStrongQualifier = false;
  Type::TypeClass TC = T->getTypeClass();
  if (const AutoType *AT = dyn_cast<AutoType>(T))
    TC = AT->desugar()->getTypeClass();
  if (const SubstTemplateTypeParmType *Subst
                                      = dyn_cast<SubstTemplateTypeParmType>(T))
    TC = Subst->getReplacementType()->getTypeClass();
  
  switch (TC) {
    case Type::Auto:
    case Type::Builtin:
    case Type::Complex:
    case Type::UnresolvedUsing:
    case Type::Typedef:
    case Type::TypeOfExpr:
    case Type::TypeOf:
    case Type::Decltype:
    case Type::UnaryTransform:
    case Type::Record:
    case Type::Enum:
    case Type::Elaborated:
    case Type::TemplateTypeParm:
    case Type::SubstTemplateTypeParmPack:
    case Type::DeducedTemplateSpecialization:
    case Type::TemplateSpecialization:
    case Type::InjectedClassName:
    case Type::DependentName:
    case Type::DependentTemplateSpecialization:
    case Type::ObjCObject:
    case Type::ObjCTypeParam:
    case Type::ObjCInterface:
    case Type::Atomic:
    case Type::Pipe:
      CanPrefixQualifiers = true;
      break;
      
    case Type::ObjCObjectPointer:
      CanPrefixQualifiers = T->isObjCIdType() || T->isObjCClassType() ||
        T->isObjCQualifiedIdType() || T->isObjCQualifiedClassType();
      break;
      
    case Type::ConstantArray:
    case Type::IncompleteArray:
    case Type::VariableArray:
    case Type::DependentSizedArray:
      NeedARCStrongQualifier = true;
      // Fall through
      
    case Type::Adjusted:
    case Type::Decayed:
    case Type::Pointer:
    case Type::BlockPointer:
    case Type::LValueReference:
    case Type::RValueReference:
    case Type::MemberPointer:
    case Type::DependentAddressSpace:
    case Type::DependentSizedExtVector:
    case Type::Vector:
    case Type::ExtVector:
    case Type::FunctionProto:
    case Type::FunctionNoProto:
    case Type::Paren:
    case Type::Attributed:
    case Type::PackExpansion:
    case Type::SubstTemplateTypeParm:
      CanPrefixQualifiers = false;
      break;
  }

  return CanPrefixQualifiers;
}

void TypePrinter::printBefore(QualType T, raw_ostream &OS) {
  SplitQualType Split = T.split();

  // If we have cv1 T, where T is substituted for cv2 U, only print cv1 - cv2
  // at this level.
  Qualifiers Quals = Split.Quals;
  if (const SubstTemplateTypeParmType *Subst =
        dyn_cast<SubstTemplateTypeParmType>(Split.Ty))
    Quals -= QualType(Subst, 0).getQualifiers();

  printBefore(Split.Ty, Quals, OS);
}

/// \brief Prints the part of the type string before an identifier, e.g. for
/// "int foo[10]" it prints "int ".
void TypePrinter::printBefore(const Type *T,Qualifiers Quals, raw_ostream &OS) {
  if (Policy.SuppressSpecifiers && T->isSpecifierType())
    return;

  SaveAndRestore<bool> PrevPHIsEmpty(HasEmptyPlaceHolder);

  // Print qualifiers as appropriate.

  bool CanPrefixQualifiers = false;
  bool NeedARCStrongQualifier = false;
  CanPrefixQualifiers = canPrefixQualifiers(T, NeedARCStrongQualifier);

  if (CanPrefixQualifiers && !Quals.empty()) {
    if (NeedARCStrongQualifier) {
      IncludeStrongLifetimeRAII Strong(Policy);
      Quals.print(OS, Policy, /*appendSpaceIfNonEmpty=*/true);
    } else {
      Quals.print(OS, Policy, /*appendSpaceIfNonEmpty=*/true);
    }
  }

  bool hasAfterQuals = false;
  if (!CanPrefixQualifiers && !Quals.empty()) {
    hasAfterQuals = !Quals.isEmptyWhenPrinted(Policy);
    if (hasAfterQuals)
      HasEmptyPlaceHolder = false;
  }

  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) case Type::CLASS: \
    print##CLASS##Before(cast<CLASS##Type>(T), OS); \
    break;
#include "clang/AST/TypeNodes.def"
  }

  if (hasAfterQuals) {
    if (NeedARCStrongQualifier) {
      IncludeStrongLifetimeRAII Strong(Policy);
      Quals.print(OS, Policy, /*appendSpaceIfNonEmpty=*/!PrevPHIsEmpty.get());
    } else {
      Quals.print(OS, Policy, /*appendSpaceIfNonEmpty=*/!PrevPHIsEmpty.get());
    }
  }
}

void TypePrinter::printAfter(QualType t, raw_ostream &OS) {
  SplitQualType split = t.split();
  printAfter(split.Ty, split.Quals, OS);
}

/// \brief Prints the part of the type string after an identifier, e.g. for
/// "int foo[10]" it prints "[10]".
void TypePrinter::printAfter(const Type *T, Qualifiers Quals, raw_ostream &OS) {
  switch (T->getTypeClass()) {
#define ABSTRACT_TYPE(CLASS, PARENT)
#define TYPE(CLASS, PARENT) case Type::CLASS: \
    print##CLASS##After(cast<CLASS##Type>(T), OS); \
    break;
#include "clang/AST/TypeNodes.def"
  }
}

void TypePrinter::printBuiltinBefore(const BuiltinType *T, raw_ostream &OS) {
  OS << T->getName(Policy);
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printBuiltinAfter(const BuiltinType *T, raw_ostream &OS) { }

void TypePrinter::printComplexBefore(const ComplexType *T, raw_ostream &OS) {
  OS << "_Complex ";
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printComplexAfter(const ComplexType *T, raw_ostream &OS) {
  printAfter(T->getElementType(), OS);
}

void TypePrinter::printPointerBefore(const PointerType *T, raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getPointeeType(), OS);
  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeType()))
    OS << '(';
  OS << '*';
}
void TypePrinter::printPointerAfter(const PointerType *T, raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  // Handle things like 'int (*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeType()))
    OS << ')';
  printAfter(T->getPointeeType(), OS);
}

void TypePrinter::printBlockPointerBefore(const BlockPointerType *T,
                                          raw_ostream &OS) {
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getPointeeType(), OS);
  OS << '^';
}
void TypePrinter::printBlockPointerAfter(const BlockPointerType *T,
                                          raw_ostream &OS) {
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printAfter(T->getPointeeType(), OS);
}

void TypePrinter::printLValueReferenceBefore(const LValueReferenceType *T,
                                             raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getPointeeTypeAsWritten(), OS);
  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeTypeAsWritten()))
    OS << '(';
  OS << '&';
}
void TypePrinter::printLValueReferenceAfter(const LValueReferenceType *T,
                                            raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  // Handle things like 'int (&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeTypeAsWritten()))
    OS << ')';
  printAfter(T->getPointeeTypeAsWritten(), OS);
}

void TypePrinter::printRValueReferenceBefore(const RValueReferenceType *T,
                                             raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getPointeeTypeAsWritten(), OS);
  // Handle things like 'int (&&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeTypeAsWritten()))
    OS << '(';
  OS << "&&";
}
void TypePrinter::printRValueReferenceAfter(const RValueReferenceType *T,
                                            raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  // Handle things like 'int (&&A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeTypeAsWritten()))
    OS << ')';
  printAfter(T->getPointeeTypeAsWritten(), OS);
}

void TypePrinter::printMemberPointerBefore(const MemberPointerType *T, 
                                           raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getPointeeType(), OS);
  // Handle things like 'int (Cls::*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeType()))
    OS << '(';

  PrintingPolicy InnerPolicy(Policy);
  InnerPolicy.IncludeTagDefinition = false;
  TypePrinter(InnerPolicy).print(QualType(T->getClass(), 0), OS, StringRef());

  OS << "::*";
}
void TypePrinter::printMemberPointerAfter(const MemberPointerType *T, 
                                          raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  // Handle things like 'int (Cls::*A)[4];' correctly.
  // FIXME: this should include vectors, but vectors use attributes I guess.
  if (isa<ArrayType>(T->getPointeeType()))
    OS << ')';
  printAfter(T->getPointeeType(), OS);
}

void TypePrinter::printConstantArrayBefore(const ConstantArrayType *T, 
                                           raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printConstantArrayAfter(const ConstantArrayType *T, 
                                          raw_ostream &OS) {
  OS << '[';
  if (T->getIndexTypeQualifiers().hasQualifiers()) {
    AppendTypeQualList(OS, T->getIndexTypeCVRQualifiers(),
                       Policy.Restrict);
    OS << ' ';
  }

  if (T->getSizeModifier() == ArrayType::Static)
    OS << "static ";

  OS << T->getSize().getZExtValue() << ']';
  printAfter(T->getElementType(), OS);
}

void TypePrinter::printIncompleteArrayBefore(const IncompleteArrayType *T, 
                                             raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printIncompleteArrayAfter(const IncompleteArrayType *T, 
                                            raw_ostream &OS) {
  OS << "[]";
  printAfter(T->getElementType(), OS);
}

void TypePrinter::printVariableArrayBefore(const VariableArrayType *T, 
                                           raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printVariableArrayAfter(const VariableArrayType *T, 
                                          raw_ostream &OS) {
  OS << '[';
  if (T->getIndexTypeQualifiers().hasQualifiers()) {
    AppendTypeQualList(OS, T->getIndexTypeCVRQualifiers(), Policy.Restrict);
    OS << ' ';
  }

  if (T->getSizeModifier() == VariableArrayType::Static)
    OS << "static ";
  else if (T->getSizeModifier() == VariableArrayType::Star)
    OS << '*';

  if (T->getSizeExpr())
    T->getSizeExpr()->printPretty(OS, nullptr, Policy);
  OS << ']';

  printAfter(T->getElementType(), OS);
}

void TypePrinter::printAdjustedBefore(const AdjustedType *T, raw_ostream &OS) {
  // Print the adjusted representation, otherwise the adjustment will be
  // invisible.
  printBefore(T->getAdjustedType(), OS);
}
void TypePrinter::printAdjustedAfter(const AdjustedType *T, raw_ostream &OS) {
  printAfter(T->getAdjustedType(), OS);
}

void TypePrinter::printDecayedBefore(const DecayedType *T, raw_ostream &OS) {
  // Print as though it's a pointer.
  printAdjustedBefore(T, OS);
}
void TypePrinter::printDecayedAfter(const DecayedType *T, raw_ostream &OS) {
  printAdjustedAfter(T, OS);
}

void TypePrinter::printDependentSizedArrayBefore(
                                               const DependentSizedArrayType *T, 
                                               raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printDependentSizedArrayAfter(
                                               const DependentSizedArrayType *T, 
                                               raw_ostream &OS) {
  OS << '[';
  if (T->getSizeExpr())
    T->getSizeExpr()->printPretty(OS, nullptr, Policy);
  OS << ']';
  printAfter(T->getElementType(), OS);
}

void TypePrinter::printDependentAddressSpaceBefore(
    const DependentAddressSpaceType *T, raw_ostream &OS) {
  printBefore(T->getPointeeType(), OS);
}
void TypePrinter::printDependentAddressSpaceAfter(
    const DependentAddressSpaceType *T, raw_ostream &OS) {
  OS << " __attribute__((address_space(";
  if (T->getAddrSpaceExpr())
    T->getAddrSpaceExpr()->printPretty(OS, nullptr, Policy);
  OS << ")))";
  printAfter(T->getPointeeType(), OS);
}

void TypePrinter::printDependentSizedExtVectorBefore(
                                          const DependentSizedExtVectorType *T, 
                                          raw_ostream &OS) { 
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printDependentSizedExtVectorAfter(
                                          const DependentSizedExtVectorType *T, 
                                          raw_ostream &OS) { 
  OS << " __attribute__((ext_vector_type(";
  if (T->getSizeExpr())
    T->getSizeExpr()->printPretty(OS, nullptr, Policy);
  OS << ")))";  
  printAfter(T->getElementType(), OS);
}

void TypePrinter::printVectorBefore(const VectorType *T, raw_ostream &OS) { 
  switch (T->getVectorKind()) {
  case VectorType::AltiVecPixel:
    OS << "__vector __pixel ";
    break;
  case VectorType::AltiVecBool:
    OS << "__vector __bool ";
    printBefore(T->getElementType(), OS);
    break;
  case VectorType::AltiVecVector:
    OS << "__vector ";
    printBefore(T->getElementType(), OS);
    break;
  case VectorType::NeonVector:
    OS << "__attribute__((neon_vector_type("
       << T->getNumElements() << "))) ";
    printBefore(T->getElementType(), OS);
    break;
  case VectorType::NeonPolyVector:
    OS << "__attribute__((neon_polyvector_type(" <<
          T->getNumElements() << "))) ";
    printBefore(T->getElementType(), OS);
    break;
  case VectorType::GenericVector: {
    // FIXME: We prefer to print the size directly here, but have no way
    // to get the size of the type.
    OS << "__attribute__((__vector_size__("
       << T->getNumElements()
       << " * sizeof(";
    print(T->getElementType(), OS, StringRef());
    OS << ")))) "; 
    printBefore(T->getElementType(), OS);
    break;
  }
  }
}
void TypePrinter::printVectorAfter(const VectorType *T, raw_ostream &OS) {
  printAfter(T->getElementType(), OS);
} 

void TypePrinter::printExtVectorBefore(const ExtVectorType *T,
                                       raw_ostream &OS) { 
  printBefore(T->getElementType(), OS);
}
void TypePrinter::printExtVectorAfter(const ExtVectorType *T, raw_ostream &OS) { 
  printAfter(T->getElementType(), OS);
  OS << " __attribute__((ext_vector_type(";
  OS << T->getNumElements();
  OS << ")))";
}

void 
FunctionProtoType::printExceptionSpecification(raw_ostream &OS, 
                                               const PrintingPolicy &Policy)
                                                                         const {
  
  if (hasDynamicExceptionSpec()) {
    OS << " throw(";
    if (getExceptionSpecType() == EST_MSAny)
      OS << "...";
    else
      for (unsigned I = 0, N = getNumExceptions(); I != N; ++I) {
        if (I)
          OS << ", ";
        
        OS << getExceptionType(I).stream(Policy);
      }
    OS << ')';
  } else if (isNoexceptExceptionSpec(getExceptionSpecType())) {
    OS << " noexcept";
    if (getExceptionSpecType() == EST_ComputedNoexcept) {
      OS << '(';
      if (getNoexceptExpr())
        getNoexceptExpr()->printPretty(OS, nullptr, Policy);
      OS << ')';
    }
  }
}

void TypePrinter::printFunctionProtoBefore(const FunctionProtoType *T, 
                                           raw_ostream &OS) {
  if (T->hasTrailingReturn()) {
    OS << "auto ";
    if (!HasEmptyPlaceHolder)
      OS << '(';
  } else {
    // If needed for precedence reasons, wrap the inner part in grouping parens.
    SaveAndRestore<bool> PrevPHIsEmpty(HasEmptyPlaceHolder, false);
    printBefore(T->getReturnType(), OS);
    if (!PrevPHIsEmpty.get())
      OS << '(';
  }
}

llvm::StringRef clang::getParameterABISpelling(ParameterABI ABI) {
  switch (ABI) {
  case ParameterABI::Ordinary:
    llvm_unreachable("asking for spelling of ordinary parameter ABI");
  case ParameterABI::SwiftContext:
    return "swift_context";
  case ParameterABI::SwiftErrorResult:
    return "swift_error_result";
  case ParameterABI::SwiftIndirectResult:
    return "swift_indirect_result";
  }
  llvm_unreachable("bad parameter ABI kind");
}

void TypePrinter::printFunctionProtoAfter(const FunctionProtoType *T, 
                                          raw_ostream &OS) { 
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!HasEmptyPlaceHolder)
    OS << ')';
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);

  OS << '(';
  {
    ParamPolicyRAII ParamPolicy(Policy);
    for (unsigned i = 0, e = T->getNumParams(); i != e; ++i) {
      if (i) OS << ", ";

      auto EPI = T->getExtParameterInfo(i);
      if (EPI.isConsumed()) OS << "__attribute__((ns_consumed)) ";
      if (EPI.isNoEscape())
        OS << "__attribute__((noescape)) ";
      auto ABI = EPI.getABI();
      if (ABI != ParameterABI::Ordinary)
        OS << "__attribute__((" << getParameterABISpelling(ABI) << ")) ";

      print(T->getParamType(i), OS, StringRef());
    }
  }
  
  if (T->isVariadic()) {
    if (T->getNumParams())
      OS << ", ";
    OS << "...";
  } else if (T->getNumParams() == 0 && Policy.UseVoidForZeroParams) {
    // Do not emit int() if we have a proto, emit 'int(void)'.
    OS << "void";
  }
  
  OS << ')';

  FunctionType::ExtInfo Info = T->getExtInfo();

  printFunctionAfter(Info, OS);

  if (unsigned quals = T->getTypeQuals()) {
    OS << ' ';
    AppendTypeQualList(OS, quals, Policy.Restrict);
  }

  switch (T->getRefQualifier()) {
  case RQ_None:
    break;

  case RQ_LValue:
    OS << " &";
    break;

  case RQ_RValue:
    OS << " &&";
    break;
  }
  T->printExceptionSpecification(OS, Policy);

  if (T->hasTrailingReturn()) {
    OS << " -> ";
    print(T->getReturnType(), OS, StringRef());
  } else
    printAfter(T->getReturnType(), OS);
}

void TypePrinter::printFunctionAfter(const FunctionType::ExtInfo &Info,
                                     raw_ostream &OS) {
  if (!InsideCCAttribute) {
    switch (Info.getCC()) {
    case CC_C:
      // The C calling convention is the default on the vast majority of platforms
      // we support.  If the user wrote it explicitly, it will usually be printed
      // while traversing the AttributedType.  If the type has been desugared, let
      // the canonical spelling be the implicit calling convention.
      // FIXME: It would be better to be explicit in certain contexts, such as a
      // cdecl function typedef used to declare a member function with the
      // Microsoft C++ ABI.
      break;
    case CC_X86StdCall:
      OS << " __attribute__((stdcall))";
      break;
    case CC_X86FastCall:
      OS << " __attribute__((fastcall))";
      break;
    case CC_X86ThisCall:
      OS << " __attribute__((thiscall))";
      break;
    case CC_X86VectorCall:
      OS << " __attribute__((vectorcall))";
      break;
    case CC_X86Pascal:
      OS << " __attribute__((pascal))";
      break;
    case CC_AAPCS:
      OS << " __attribute__((pcs(\"aapcs\")))";
      break;
    case CC_AAPCS_VFP:
      OS << " __attribute__((pcs(\"aapcs-vfp\")))";
      break;
    case CC_IntelOclBicc:
      OS << " __attribute__((intel_ocl_bicc))";
      break;
    case CC_Win64:
      OS << " __attribute__((ms_abi))";
      break;
    case CC_X86_64SysV:
      OS << " __attribute__((sysv_abi))";
      break;
    case CC_X86RegCall:
      OS << " __attribute__((regcall))";
      break;
    case CC_SpirFunction:
    case CC_OpenCLKernel:
      // Do nothing. These CCs are not available as attributes.
      break;
    case CC_Swift:
      OS << " __attribute__((swiftcall))";
      break;
    case CC_PreserveMost:
      OS << " __attribute__((preserve_most))";
      break;
    case CC_PreserveAll:
      OS << " __attribute__((preserve_all))";
      break;
    }
  }

  if (Info.getNoReturn())
    OS << " __attribute__((noreturn))";
  if (Info.getProducesResult())
    OS << " __attribute__((ns_returns_retained))";
  if (Info.getRegParm())
    OS << " __attribute__((regparm ("
       << Info.getRegParm() << ")))";
  if (Info.getNoCallerSavedRegs())
    OS << " __attribute__((no_caller_saved_registers))";
}

void TypePrinter::printFunctionNoProtoBefore(const FunctionNoProtoType *T, 
                                             raw_ostream &OS) { 
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  SaveAndRestore<bool> PrevPHIsEmpty(HasEmptyPlaceHolder, false);
  printBefore(T->getReturnType(), OS);
  if (!PrevPHIsEmpty.get())
    OS << '(';
}
void TypePrinter::printFunctionNoProtoAfter(const FunctionNoProtoType *T, 
                                            raw_ostream &OS) {
  // If needed for precedence reasons, wrap the inner part in grouping parens.
  if (!HasEmptyPlaceHolder)
    OS << ')';
  SaveAndRestore<bool> NonEmptyPH(HasEmptyPlaceHolder, false);
  
  OS << "()";
  printFunctionAfter(T->getExtInfo(), OS);
  printAfter(T->getReturnType(), OS);
}

void TypePrinter::printTypeSpec(NamedDecl *D, raw_ostream &OS) {

  // Compute the full nested-name-specifier for this type.
  // In C, this will always be empty except when the type
  // being printed is anonymous within other Record.
  if (!Policy.SuppressScope)
    AppendScope(D->getDeclContext(), OS);

  IdentifierInfo *II = D->getIdentifier();
  OS << II->getName();
  spaceBeforePlaceHolder(OS);
}

void TypePrinter::printUnresolvedUsingBefore(const UnresolvedUsingType *T,
                                             raw_ostream &OS) {
  printTypeSpec(T->getDecl(), OS);
}
void TypePrinter::printUnresolvedUsingAfter(const UnresolvedUsingType *T,
                                             raw_ostream &OS) { }

void TypePrinter::printTypedefBefore(const TypedefType *T, raw_ostream &OS) { 
  printTypeSpec(T->getDecl(), OS);
}
void TypePrinter::printTypedefAfter(const TypedefType *T, raw_ostream &OS) { } 

void TypePrinter::printTypeOfExprBefore(const TypeOfExprType *T,
                                        raw_ostream &OS) {
  OS << "typeof ";
  if (T->getUnderlyingExpr())
    T->getUnderlyingExpr()->printPretty(OS, nullptr, Policy);
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printTypeOfExprAfter(const TypeOfExprType *T,
                                       raw_ostream &OS) { }

void TypePrinter::printTypeOfBefore(const TypeOfType *T, raw_ostream &OS) { 
  OS << "typeof(";
  print(T->getUnderlyingType(), OS, StringRef());
  OS << ')';
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printTypeOfAfter(const TypeOfType *T, raw_ostream &OS) { } 

void TypePrinter::printDecltypeBefore(const DecltypeType *T, raw_ostream &OS) { 
  OS << "decltype(";
  if (T->getUnderlyingExpr())
    T->getUnderlyingExpr()->printPretty(OS, nullptr, Policy);
  OS << ')';
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printDecltypeAfter(const DecltypeType *T, raw_ostream &OS) { } 

void TypePrinter::printUnaryTransformBefore(const UnaryTransformType *T,
                                            raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);

  switch (T->getUTTKind()) {
    case UnaryTransformType::EnumUnderlyingType:
      OS << "__underlying_type(";
      print(T->getBaseType(), OS, StringRef());
      OS << ')';
      spaceBeforePlaceHolder(OS);
      return;
  }

  printBefore(T->getBaseType(), OS);
}
void TypePrinter::printUnaryTransformAfter(const UnaryTransformType *T,
                                           raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);

  switch (T->getUTTKind()) {
    case UnaryTransformType::EnumUnderlyingType:
      return;
  }

  printAfter(T->getBaseType(), OS);
}

void TypePrinter::printAutoBefore(const AutoType *T, raw_ostream &OS) { 
  // If the type has been deduced, do not print 'auto'.
  if (!T->getDeducedType().isNull()) {
    printBefore(T->getDeducedType(), OS);
  } else {
    switch (T->getKeyword()) {
    case AutoTypeKeyword::Auto: OS << "auto"; break;
    case AutoTypeKeyword::DecltypeAuto: OS << "decltype(auto)"; break;
    case AutoTypeKeyword::GNUAutoType: OS << "__auto_type"; break;
    }
    spaceBeforePlaceHolder(OS);
  }
}
void TypePrinter::printAutoAfter(const AutoType *T, raw_ostream &OS) { 
  // If the type has been deduced, do not print 'auto'.
  if (!T->getDeducedType().isNull())
    printAfter(T->getDeducedType(), OS);
}

void TypePrinter::printDeducedTemplateSpecializationBefore(
    const DeducedTemplateSpecializationType *T, raw_ostream &OS) {
  // If the type has been deduced, print the deduced type.
  if (!T->getDeducedType().isNull()) {
    printBefore(T->getDeducedType(), OS);
  } else {
    IncludeStrongLifetimeRAII Strong(Policy);
    T->getTemplateName().print(OS, Policy);
    spaceBeforePlaceHolder(OS);
  }
}
void TypePrinter::printDeducedTemplateSpecializationAfter(
    const DeducedTemplateSpecializationType *T, raw_ostream &OS) {
  // If the type has been deduced, print the deduced type.
  if (!T->getDeducedType().isNull())
    printAfter(T->getDeducedType(), OS);
}

void TypePrinter::printAtomicBefore(const AtomicType *T, raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);

  OS << "_Atomic(";
  print(T->getValueType(), OS, StringRef());
  OS << ')';
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printAtomicAfter(const AtomicType *T, raw_ostream &OS) { }

void TypePrinter::printPipeBefore(const PipeType *T, raw_ostream &OS) {
  IncludeStrongLifetimeRAII Strong(Policy);

  if (T->isReadOnly())
    OS << "read_only ";
  else
    OS << "write_only ";
  OS << "pipe ";
  print(T->getElementType(), OS, StringRef());
  spaceBeforePlaceHolder(OS);
}

void TypePrinter::printPipeAfter(const PipeType *T, raw_ostream &OS) {
}
/// Appends the given scope to the end of a string.
void TypePrinter::AppendScope(DeclContext *DC, raw_ostream &OS) {
  if (DC->isTranslationUnit()) return;
  if (DC->isFunctionOrMethod()) return;
  AppendScope(DC->getParent(), OS);

  if (NamespaceDecl *NS = dyn_cast<NamespaceDecl>(DC)) {
    if (Policy.SuppressUnwrittenScope && 
        (NS->isAnonymousNamespace() || NS->isInline()))
      return;
    if (NS->getIdentifier())
      OS << NS->getName() << "::";
    else
      OS << "(anonymous namespace)::";
  } else if (ClassTemplateSpecializationDecl *Spec
               = dyn_cast<ClassTemplateSpecializationDecl>(DC)) {
    IncludeStrongLifetimeRAII Strong(Policy);
    OS << Spec->getIdentifier()->getName();
    const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
    TemplateSpecializationType::PrintTemplateArgumentList(
        OS, TemplateArgs.asArray(), Policy);
    OS << "::";
  } else if (TagDecl *Tag = dyn_cast<TagDecl>(DC)) {
    if (TypedefNameDecl *Typedef = Tag->getTypedefNameForAnonDecl())
      OS << Typedef->getIdentifier()->getName() << "::";
    else if (Tag->getIdentifier())
      OS << Tag->getIdentifier()->getName() << "::";
    else
      return;
  }
}

void TypePrinter::printTag(TagDecl *D, raw_ostream &OS) {
  if (Policy.IncludeTagDefinition) {
    PrintingPolicy SubPolicy = Policy;
    SubPolicy.IncludeTagDefinition = false;
    D->print(OS, SubPolicy, Indentation);
    spaceBeforePlaceHolder(OS);
    return;
  }

  bool HasKindDecoration = false;

  // We don't print tags unless this is an elaborated type.
  // In C, we just assume every RecordType is an elaborated type.
  if (!Policy.SuppressTagKeyword && !D->getTypedefNameForAnonDecl()) {
    HasKindDecoration = true;
    OS << D->getKindName();
    OS << ' ';
  }

  // Compute the full nested-name-specifier for this type.
  // In C, this will always be empty except when the type
  // being printed is anonymous within other Record.
  if (!Policy.SuppressScope)
    AppendScope(D->getDeclContext(), OS);

  if (const IdentifierInfo *II = D->getIdentifier())
    OS << II->getName();
  else if (TypedefNameDecl *Typedef = D->getTypedefNameForAnonDecl()) {
    assert(Typedef->getIdentifier() && "Typedef without identifier?");
    OS << Typedef->getIdentifier()->getName();
  } else {
    // Make an unambiguous representation for anonymous types, e.g.
    //   (anonymous enum at /usr/include/string.h:120:9)
    OS << (Policy.MSVCFormatting ? '`' : '(');

    if (isa<CXXRecordDecl>(D) && cast<CXXRecordDecl>(D)->isLambda()) {
      OS << "lambda";
      HasKindDecoration = true;
    } else {
      OS << "anonymous";
    }
    
    if (Policy.AnonymousTagLocations) {
      // Suppress the redundant tag keyword if we just printed one.
      // We don't have to worry about ElaboratedTypes here because you can't
      // refer to an anonymous type with one.
      if (!HasKindDecoration)
        OS << " " << D->getKindName();

      PresumedLoc PLoc = D->getASTContext().getSourceManager().getPresumedLoc(
          D->getLocation());
      if (PLoc.isValid()) {
        OS << " at " << PLoc.getFilename()
           << ':' << PLoc.getLine()
           << ':' << PLoc.getColumn();
      }
    }

    OS << (Policy.MSVCFormatting ? '\'' : ')');
  }

  // If this is a class template specialization, print the template
  // arguments.
  if (ClassTemplateSpecializationDecl *Spec
        = dyn_cast<ClassTemplateSpecializationDecl>(D)) {
    ArrayRef<TemplateArgument> Args;
    if (TypeSourceInfo *TAW = Spec->getTypeAsWritten()) {
      const TemplateSpecializationType *TST =
        cast<TemplateSpecializationType>(TAW->getType());
      Args = TST->template_arguments();
    } else {
      const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
      Args = TemplateArgs.asArray();
    }
    IncludeStrongLifetimeRAII Strong(Policy);
    TemplateSpecializationType::PrintTemplateArgumentList(OS, Args, Policy);
  }

  spaceBeforePlaceHolder(OS);
}

void TypePrinter::printRecordBefore(const RecordType *T, raw_ostream &OS) {
  printTag(T->getDecl(), OS);
}
void TypePrinter::printRecordAfter(const RecordType *T, raw_ostream &OS) { }

void TypePrinter::printEnumBefore(const EnumType *T, raw_ostream &OS) { 
  printTag(T->getDecl(), OS);
}
void TypePrinter::printEnumAfter(const EnumType *T, raw_ostream &OS) { }

void TypePrinter::printTemplateTypeParmBefore(const TemplateTypeParmType *T, 
                                              raw_ostream &OS) { 
  if (IdentifierInfo *Id = T->getIdentifier())
    OS << Id->getName();
  else
    OS << "type-parameter-" << T->getDepth() << '-' << T->getIndex();
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printTemplateTypeParmAfter(const TemplateTypeParmType *T, 
                                             raw_ostream &OS) { } 

void TypePrinter::printSubstTemplateTypeParmBefore(
                                             const SubstTemplateTypeParmType *T, 
                                             raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  printBefore(T->getReplacementType(), OS);
}
void TypePrinter::printSubstTemplateTypeParmAfter(
                                             const SubstTemplateTypeParmType *T, 
                                             raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  printAfter(T->getReplacementType(), OS);
}

void TypePrinter::printSubstTemplateTypeParmPackBefore(
                                        const SubstTemplateTypeParmPackType *T, 
                                        raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  printTemplateTypeParmBefore(T->getReplacedParameter(), OS);
}
void TypePrinter::printSubstTemplateTypeParmPackAfter(
                                        const SubstTemplateTypeParmPackType *T, 
                                        raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  printTemplateTypeParmAfter(T->getReplacedParameter(), OS);
}

void TypePrinter::printTemplateSpecializationBefore(
                                            const TemplateSpecializationType *T, 
                                            raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  T->getTemplateName().print(OS, Policy);

  TemplateSpecializationType::PrintTemplateArgumentList(
      OS, T->template_arguments(), Policy);
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printTemplateSpecializationAfter(
                                            const TemplateSpecializationType *T, 
                                            raw_ostream &OS) { } 

void TypePrinter::printInjectedClassNameBefore(const InjectedClassNameType *T,
                                               raw_ostream &OS) {
  printTemplateSpecializationBefore(T->getInjectedTST(), OS);
}
void TypePrinter::printInjectedClassNameAfter(const InjectedClassNameType *T,
                                               raw_ostream &OS) { }

void TypePrinter::printElaboratedBefore(const ElaboratedType *T,
                                        raw_ostream &OS) {
  // The tag definition will take care of these.
  if (!Policy.IncludeTagDefinition)
  {
    OS << TypeWithKeyword::getKeywordName(T->getKeyword());
    if (T->getKeyword() != ETK_None)
      OS << " ";
    NestedNameSpecifier* Qualifier = T->getQualifier();
    if (Qualifier)
      Qualifier->print(OS, Policy);
  }
  
  ElaboratedTypePolicyRAII PolicyRAII(Policy);
  printBefore(T->getNamedType(), OS);
}
void TypePrinter::printElaboratedAfter(const ElaboratedType *T,
                                        raw_ostream &OS) {
  ElaboratedTypePolicyRAII PolicyRAII(Policy);
  printAfter(T->getNamedType(), OS);
}

void TypePrinter::printParenBefore(const ParenType *T, raw_ostream &OS) {
  if (!HasEmptyPlaceHolder && !isa<FunctionType>(T->getInnerType())) {
    printBefore(T->getInnerType(), OS);
    OS << '(';
  } else
    printBefore(T->getInnerType(), OS);
}
void TypePrinter::printParenAfter(const ParenType *T, raw_ostream &OS) {
  if (!HasEmptyPlaceHolder && !isa<FunctionType>(T->getInnerType())) {
    OS << ')';
    printAfter(T->getInnerType(), OS);
  } else
    printAfter(T->getInnerType(), OS);
}

void TypePrinter::printDependentNameBefore(const DependentNameType *T,
                                           raw_ostream &OS) { 
  OS << TypeWithKeyword::getKeywordName(T->getKeyword());
  if (T->getKeyword() != ETK_None)
    OS << " ";
  
  T->getQualifier()->print(OS, Policy);
  
  OS << T->getIdentifier()->getName();
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printDependentNameAfter(const DependentNameType *T,
                                          raw_ostream &OS) { } 

void TypePrinter::printDependentTemplateSpecializationBefore(
        const DependentTemplateSpecializationType *T, raw_ostream &OS) { 
  IncludeStrongLifetimeRAII Strong(Policy);
  
  OS << TypeWithKeyword::getKeywordName(T->getKeyword());
  if (T->getKeyword() != ETK_None)
    OS << " ";
  
  if (T->getQualifier())
    T->getQualifier()->print(OS, Policy);    
  OS << T->getIdentifier()->getName();
  TemplateSpecializationType::PrintTemplateArgumentList(OS,
                                                        T->template_arguments(),
                                                        Policy);
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printDependentTemplateSpecializationAfter(
        const DependentTemplateSpecializationType *T, raw_ostream &OS) { } 

void TypePrinter::printPackExpansionBefore(const PackExpansionType *T, 
                                           raw_ostream &OS) {
  printBefore(T->getPattern(), OS);
}
void TypePrinter::printPackExpansionAfter(const PackExpansionType *T, 
                                          raw_ostream &OS) {
  printAfter(T->getPattern(), OS);
  OS << "...";
}

void TypePrinter::printAttributedBefore(const AttributedType *T,
                                        raw_ostream &OS) {
  // Prefer the macro forms of the GC and ownership qualifiers.
  if (T->getAttrKind() == AttributedType::attr_objc_gc ||
      T->getAttrKind() == AttributedType::attr_objc_ownership)
    return printBefore(T->getEquivalentType(), OS);

  if (T->getAttrKind() == AttributedType::attr_objc_kindof)
    OS << "__kindof ";

  printBefore(T->getModifiedType(), OS);

  if (T->isMSTypeSpec()) {
    switch (T->getAttrKind()) {
    default: return;
    case AttributedType::attr_ptr32: OS << " __ptr32"; break;
    case AttributedType::attr_ptr64: OS << " __ptr64"; break;
    case AttributedType::attr_sptr: OS << " __sptr"; break;
    case AttributedType::attr_uptr: OS << " __uptr"; break;
    }
    spaceBeforePlaceHolder(OS);
  }

  // Print nullability type specifiers.
  if (T->getAttrKind() == AttributedType::attr_nonnull ||
      T->getAttrKind() == AttributedType::attr_nullable ||
      T->getAttrKind() == AttributedType::attr_null_unspecified) {
    if (T->getAttrKind() == AttributedType::attr_nonnull)
      OS << " _Nonnull";
    else if (T->getAttrKind() == AttributedType::attr_nullable)
      OS << " _Nullable";
    else if (T->getAttrKind() == AttributedType::attr_null_unspecified)
      OS << " _Null_unspecified";
    else
      llvm_unreachable("unhandled nullability");
    spaceBeforePlaceHolder(OS);
  }
}

void TypePrinter::printAttributedAfter(const AttributedType *T,
                                       raw_ostream &OS) {
  // Prefer the macro forms of the GC and ownership qualifiers.
  if (T->getAttrKind() == AttributedType::attr_objc_gc ||
      T->getAttrKind() == AttributedType::attr_objc_ownership)
    return printAfter(T->getEquivalentType(), OS);

  if (T->getAttrKind() == AttributedType::attr_objc_kindof)
    return;

  // TODO: not all attributes are GCC-style attributes.
  if (T->isMSTypeSpec())
    return;

  // Nothing to print after.
  if (T->getAttrKind() == AttributedType::attr_nonnull ||
      T->getAttrKind() == AttributedType::attr_nullable ||
      T->getAttrKind() == AttributedType::attr_null_unspecified)
    return printAfter(T->getModifiedType(), OS);

  // If this is a calling convention attribute, don't print the implicit CC from
  // the modified type.
  SaveAndRestore<bool> MaybeSuppressCC(InsideCCAttribute, T->isCallingConv());

  printAfter(T->getModifiedType(), OS);

  // Don't print the inert __unsafe_unretained attribute at all.
  if (T->getAttrKind() == AttributedType::attr_objc_inert_unsafe_unretained)
    return;

  // Don't print ns_returns_retained unless it had an effect.
  if (T->getAttrKind() == AttributedType::attr_ns_returns_retained &&
      !T->getEquivalentType()->castAs<FunctionType>()
                             ->getExtInfo().getProducesResult())
    return;

  // Print nullability type specifiers that occur after
  if (T->getAttrKind() == AttributedType::attr_nonnull ||
      T->getAttrKind() == AttributedType::attr_nullable ||
      T->getAttrKind() == AttributedType::attr_null_unspecified) {
    if (T->getAttrKind() == AttributedType::attr_nonnull)
      OS << " _Nonnull";
    else if (T->getAttrKind() == AttributedType::attr_nullable)
      OS << " _Nullable";
    else if (T->getAttrKind() == AttributedType::attr_null_unspecified)
      OS << " _Null_unspecified";
    else
      llvm_unreachable("unhandled nullability");

    return;
  }

  OS << " __attribute__((";
  switch (T->getAttrKind()) {
  default: llvm_unreachable("This attribute should have been handled already");
  case AttributedType::attr_address_space:
    OS << "address_space(";
    OS << T->getEquivalentType().getAddressSpace();
    OS << ')';
    break;

  case AttributedType::attr_vector_size: {
    OS << "__vector_size__(";
    if (const VectorType *vector =T->getEquivalentType()->getAs<VectorType>()) {
      OS << vector->getNumElements();
      OS << " * sizeof(";
      print(vector->getElementType(), OS, StringRef());
      OS << ')';
    }
    OS << ')';
    break;
  }

  case AttributedType::attr_neon_vector_type:
  case AttributedType::attr_neon_polyvector_type: {
    if (T->getAttrKind() == AttributedType::attr_neon_vector_type)
      OS << "neon_vector_type(";
    else
      OS << "neon_polyvector_type(";
    const VectorType *vector = T->getEquivalentType()->getAs<VectorType>();
    OS << vector->getNumElements();
    OS << ')';
    break;
  }

  case AttributedType::attr_regparm: {
    // FIXME: When Sema learns to form this AttributedType, avoid printing the
    // attribute again in printFunctionProtoAfter.
    OS << "regparm(";
    QualType t = T->getEquivalentType();
    while (!t->isFunctionType())
      t = t->getPointeeType();
    OS << t->getAs<FunctionType>()->getRegParmType();
    OS << ')';
    break;
  }

  case AttributedType::attr_objc_gc: {
    OS << "objc_gc(";

    QualType tmp = T->getEquivalentType();
    while (tmp.getObjCGCAttr() == Qualifiers::GCNone) {
      QualType next = tmp->getPointeeType();
      if (next == tmp) break;
      tmp = next;
    }

    if (tmp.isObjCGCWeak())
      OS << "weak";
    else
      OS << "strong";
    OS << ')';
    break;
  }

  case AttributedType::attr_objc_ownership:
    OS << "objc_ownership(";
    switch (T->getEquivalentType().getObjCLifetime()) {
    case Qualifiers::OCL_None: llvm_unreachable("no ownership!");
    case Qualifiers::OCL_ExplicitNone: OS << "none"; break;
    case Qualifiers::OCL_Strong: OS << "strong"; break;
    case Qualifiers::OCL_Weak: OS << "weak"; break;
    case Qualifiers::OCL_Autoreleasing: OS << "autoreleasing"; break;
    }
    OS << ')';
    break;

  case AttributedType::attr_ns_returns_retained:
    OS << "ns_returns_retained";
    break;

  // FIXME: When Sema learns to form this AttributedType, avoid printing the
  // attribute again in printFunctionProtoAfter.
  case AttributedType::attr_noreturn: OS << "noreturn"; break;

  case AttributedType::attr_cdecl: OS << "cdecl"; break;
  case AttributedType::attr_fastcall: OS << "fastcall"; break;
  case AttributedType::attr_stdcall: OS << "stdcall"; break;
  case AttributedType::attr_thiscall: OS << "thiscall"; break;
  case AttributedType::attr_swiftcall: OS << "swiftcall"; break;
  case AttributedType::attr_vectorcall: OS << "vectorcall"; break;
  case AttributedType::attr_pascal: OS << "pascal"; break;
  case AttributedType::attr_ms_abi: OS << "ms_abi"; break;
  case AttributedType::attr_sysv_abi: OS << "sysv_abi"; break;
  case AttributedType::attr_regcall: OS << "regcall"; break;
  case AttributedType::attr_pcs:
  case AttributedType::attr_pcs_vfp: {
    OS << "pcs(";
   QualType t = T->getEquivalentType();
   while (!t->isFunctionType())
     t = t->getPointeeType();
   OS << (t->getAs<FunctionType>()->getCallConv() == CC_AAPCS ?
         "\"aapcs\"" : "\"aapcs-vfp\"");
   OS << ')';
   break;
  }
  case AttributedType::attr_inteloclbicc: OS << "inteloclbicc"; break;
  case AttributedType::attr_preserve_most:
    OS << "preserve_most";
    break;
  case AttributedType::attr_preserve_all:
    OS << "preserve_all";
    break;
  }
  OS << "))";
}

void TypePrinter::printObjCInterfaceBefore(const ObjCInterfaceType *T, 
                                           raw_ostream &OS) { 
  OS << T->getDecl()->getName();
  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printObjCInterfaceAfter(const ObjCInterfaceType *T, 
                                          raw_ostream &OS) { } 

void TypePrinter::printObjCTypeParamBefore(const ObjCTypeParamType *T,
                                          raw_ostream &OS) {
  OS << T->getDecl()->getName();
  if (!T->qual_empty()) {
    bool isFirst = true;
    OS << '<';
    for (const auto *I : T->quals()) {
      if (isFirst)
        isFirst = false;
      else
        OS << ',';
      OS << I->getName();
    }
    OS << '>';
  }

  spaceBeforePlaceHolder(OS);
}

void TypePrinter::printObjCTypeParamAfter(const ObjCTypeParamType *T,
                                          raw_ostream &OS) { }

void TypePrinter::printObjCObjectBefore(const ObjCObjectType *T,
                                        raw_ostream &OS) {
  if (T->qual_empty() && T->isUnspecializedAsWritten() &&
      !T->isKindOfTypeAsWritten())
    return printBefore(T->getBaseType(), OS);

  if (T->isKindOfTypeAsWritten())
    OS << "__kindof ";

  print(T->getBaseType(), OS, StringRef());

  if (T->isSpecializedAsWritten()) {
    bool isFirst = true;
    OS << '<';
    for (auto typeArg : T->getTypeArgsAsWritten()) {
      if (isFirst)
        isFirst = false;
      else
        OS << ",";

      print(typeArg, OS, StringRef());
    }
    OS << '>';
  }

  if (!T->qual_empty()) {
    bool isFirst = true;
    OS << '<';
    for (const auto *I : T->quals()) {
      if (isFirst)
        isFirst = false;
      else
        OS << ',';
      OS << I->getName();
    }
    OS << '>';
  }

  spaceBeforePlaceHolder(OS);
}
void TypePrinter::printObjCObjectAfter(const ObjCObjectType *T,
                                        raw_ostream &OS) {
  if (T->qual_empty() && T->isUnspecializedAsWritten() &&
      !T->isKindOfTypeAsWritten())
    return printAfter(T->getBaseType(), OS);
}

void TypePrinter::printObjCObjectPointerBefore(const ObjCObjectPointerType *T, 
                                               raw_ostream &OS) {
  printBefore(T->getPointeeType(), OS);

  // If we need to print the pointer, print it now.
  if (!T->isObjCIdType() && !T->isObjCQualifiedIdType() &&
      !T->isObjCClassType() && !T->isObjCQualifiedClassType()) {
    if (HasEmptyPlaceHolder)
      OS << ' ';
    OS << '*';
  }
}
void TypePrinter::printObjCObjectPointerAfter(const ObjCObjectPointerType *T, 
                                              raw_ostream &OS) { }

void TemplateSpecializationType::
  PrintTemplateArgumentList(raw_ostream &OS,
                            const TemplateArgumentListInfo &Args,
                            const PrintingPolicy &Policy) {
  return PrintTemplateArgumentList(OS,
                                   Args.arguments(),
                                   Policy);
}

void TemplateSpecializationType::PrintTemplateArgumentList(
    raw_ostream &OS, ArrayRef<TemplateArgument> Args,
    const PrintingPolicy &Policy, bool SkipBrackets) {
  const char *Comma = Policy.MSVCFormatting ? "," : ", ";
  if (!SkipBrackets)
    OS << '<';

  bool needSpace = false;
  bool FirstArg = true;
  for (const TemplateArgument &Arg : Args) {
    // Print the argument into a string.
    SmallString<128> Buf;
    llvm::raw_svector_ostream ArgOS(Buf);
    if (Arg.getKind() == TemplateArgument::Pack) {
      if (Arg.pack_size() && !FirstArg)
        OS << Comma;
      PrintTemplateArgumentList(ArgOS,
                                Arg.getPackAsArray(),
                                Policy, true);
    } else {
      if (!FirstArg)
        OS << Comma;
      Arg.print(Policy, ArgOS);
    }
    StringRef ArgString = ArgOS.str();

    // If this is the first argument and its string representation
    // begins with the global scope specifier ('::foo'), add a space
    // to avoid printing the diagraph '<:'.
    if (FirstArg && !ArgString.empty() && ArgString[0] == ':')
      OS << ' ';

    OS << ArgString;

    needSpace = (!ArgString.empty() && ArgString.back() == '>');
    FirstArg = false;
  }

  // If the last character of our string is '>', add another space to
  // keep the two '>''s separate tokens. We don't *have* to do this in
  // C++0x, but it's still good hygiene.
  if (needSpace)
    OS << ' ';

  if (!SkipBrackets)
    OS << '>';
}

// Sadly, repeat all that with TemplateArgLoc.
void TemplateSpecializationType::
PrintTemplateArgumentList(raw_ostream &OS,
                          ArrayRef<TemplateArgumentLoc> Args,
                          const PrintingPolicy &Policy) {
  OS << '<';
  const char *Comma = Policy.MSVCFormatting ? "," : ", ";

  bool needSpace = false;
  bool FirstArg = true;
  for (const TemplateArgumentLoc &Arg : Args) {
    if (!FirstArg)
      OS << Comma;

    // Print the argument into a string.
    SmallString<128> Buf;
    llvm::raw_svector_ostream ArgOS(Buf);
    if (Arg.getArgument().getKind() == TemplateArgument::Pack) {
      PrintTemplateArgumentList(ArgOS,
                                Arg.getArgument().getPackAsArray(),
                                Policy, true);
    } else {
      Arg.getArgument().print(Policy, ArgOS);
    }
    StringRef ArgString = ArgOS.str();

    // If this is the first argument and its string representation
    // begins with the global scope specifier ('::foo'), add a space
    // to avoid printing the diagraph '<:'.
    if (FirstArg && !ArgString.empty() && ArgString[0] == ':')
      OS << ' ';

    OS << ArgString;

    needSpace = (!ArgString.empty() && ArgString.back() == '>');
    FirstArg = false;
  }

  // If the last character of our string is '>', add another space to
  // keep the two '>''s separate tokens. We don't *have* to do this in
  // C++0x, but it's still good hygiene.
  if (needSpace)
    OS << ' ';

  OS << '>';
}

std::string Qualifiers::getAsString() const {
  LangOptions LO;
  return getAsString(PrintingPolicy(LO));
}

// Appends qualifiers to the given string, separated by spaces.  Will
// prefix a space if the string is non-empty.  Will not append a final
// space.
std::string Qualifiers::getAsString(const PrintingPolicy &Policy) const {
  SmallString<64> Buf;
  llvm::raw_svector_ostream StrOS(Buf);
  print(StrOS, Policy);
  return StrOS.str();
}

bool Qualifiers::isEmptyWhenPrinted(const PrintingPolicy &Policy) const {
  if (getCVRQualifiers())
    return false;

  if (getAddressSpace())
    return false;

  if (getObjCGCAttr())
    return false;

  if (Qualifiers::ObjCLifetime lifetime = getObjCLifetime())
    if (!(lifetime == Qualifiers::OCL_Strong && Policy.SuppressStrongLifetime))
      return false;

  return true;
}

// Appends qualifiers to the given string, separated by spaces.  Will
// prefix a space if the string is non-empty.  Will not append a final
// space.
void Qualifiers::print(raw_ostream &OS, const PrintingPolicy& Policy,
                       bool appendSpaceIfNonEmpty) const {
  bool addSpace = false;

  unsigned quals = getCVRQualifiers();
  if (quals) {
    AppendTypeQualList(OS, quals, Policy.Restrict);
    addSpace = true;
  }
  if (hasUnaligned()) {
    if (addSpace)
      OS << ' ';
    OS << "__unaligned";
    addSpace = true;
  }
  if (unsigned addrspace = getAddressSpace()) {
    if (addrspace != LangAS::opencl_private) {
      if (addSpace)
        OS << ' ';
      addSpace = true;
      switch (addrspace) {
      case LangAS::opencl_global:
        OS << "__global";
        break;
      case LangAS::opencl_local:
        OS << "__local";
        break;
      case LangAS::opencl_private:
        break;
      case LangAS::opencl_constant:
      case LangAS::cuda_constant:
        OS << "__constant";
        break;
      case LangAS::opencl_generic:
        OS << "__generic";
        break;
      case LangAS::cuda_device:
        OS << "__device";
        break;
      case LangAS::cuda_shared:
        OS << "__shared";
        break;
      default:
        assert(addrspace >= LangAS::FirstTargetAddressSpace);
        OS << "__attribute__((address_space(";
        OS << addrspace - LangAS::FirstTargetAddressSpace;
        OS << ")))";
      }
    }
  }
  if (Qualifiers::GC gc = getObjCGCAttr()) {
    if (addSpace)
      OS << ' ';
    addSpace = true;
    if (gc == Qualifiers::Weak)
      OS << "__weak";
    else
      OS << "__strong";
  }
  if (Qualifiers::ObjCLifetime lifetime = getObjCLifetime()) {
    if (!(lifetime == Qualifiers::OCL_Strong && Policy.SuppressStrongLifetime)){
      if (addSpace)
        OS << ' ';
      addSpace = true;
    }

    switch (lifetime) {
    case Qualifiers::OCL_None: llvm_unreachable("none but true");
    case Qualifiers::OCL_ExplicitNone: OS << "__unsafe_unretained"; break;
    case Qualifiers::OCL_Strong: 
      if (!Policy.SuppressStrongLifetime)
        OS << "__strong"; 
      break;
        
    case Qualifiers::OCL_Weak: OS << "__weak"; break;
    case Qualifiers::OCL_Autoreleasing: OS << "__autoreleasing"; break;
    }
  }

  if (appendSpaceIfNonEmpty && addSpace)
    OS << ' ';
}

std::string QualType::getAsString(const PrintingPolicy &Policy) const {
  std::string S;
  getAsStringInternal(S, Policy);
  return S;
}

std::string QualType::getAsString(const Type *ty, Qualifiers qs) {
  std::string buffer;
  LangOptions options;
  getAsStringInternal(ty, qs, buffer, PrintingPolicy(options));
  return buffer;
}

void QualType::print(const Type *ty, Qualifiers qs,
                     raw_ostream &OS, const PrintingPolicy &policy,
                     const Twine &PlaceHolder, unsigned Indentation) {
  SmallString<128> PHBuf;
  StringRef PH = PlaceHolder.toStringRef(PHBuf);

  TypePrinter(policy, Indentation).print(ty, qs, OS, PH);
}

void QualType::getAsStringInternal(const Type *ty, Qualifiers qs,
                                   std::string &buffer,
                                   const PrintingPolicy &policy) {
  SmallString<256> Buf;
  llvm::raw_svector_ostream StrOS(Buf);
  TypePrinter(policy).print(ty, qs, StrOS, buffer);
  std::string str = StrOS.str();
  buffer.swap(str);
}
