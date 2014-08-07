//===--- ASTWriter.cpp - AST File Writer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTWriter class, which writes AST files.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/ASTWriter.h"
#include "ASTCommon.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemStatCache.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/VersionTuple.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTReader.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/EndianStream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/OnDiskHashTable.h"
#include "llvm/Support/Path.h"
#include <algorithm>
#include <cstdio>
#include <string.h>
#include <utility>
using namespace clang;
using namespace clang::serialization;

template <typename T, typename Allocator>
static StringRef data(const std::vector<T, Allocator> &v) {
  if (v.empty()) return StringRef();
  return StringRef(reinterpret_cast<const char*>(&v[0]),
                         sizeof(T) * v.size());
}

template <typename T>
static StringRef data(const SmallVectorImpl<T> &v) {
  return StringRef(reinterpret_cast<const char*>(v.data()),
                         sizeof(T) * v.size());
}

//===----------------------------------------------------------------------===//
// Type serialization
//===----------------------------------------------------------------------===//

namespace {
  class ASTTypeWriter {
    ASTWriter &Writer;
    ASTWriter::RecordDataImpl &Record;

  public:
    /// \brief Type code that corresponds to the record generated.
    TypeCode Code;

    ASTTypeWriter(ASTWriter &Writer, ASTWriter::RecordDataImpl &Record)
      : Writer(Writer), Record(Record), Code(TYPE_EXT_QUAL) { }

    void VisitArrayType(const ArrayType *T);
    void VisitFunctionType(const FunctionType *T);
    void VisitTagType(const TagType *T);

#define TYPE(Class, Base) void Visit##Class##Type(const Class##Type *T);
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  };
}

void ASTTypeWriter::VisitBuiltinType(const BuiltinType *T) {
  llvm_unreachable("Built-in types are never serialized");
}

void ASTTypeWriter::VisitComplexType(const ComplexType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Code = TYPE_COMPLEX;
}

void ASTTypeWriter::VisitPointerType(const PointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = TYPE_POINTER;
}

void ASTTypeWriter::VisitDecayedType(const DecayedType *T) {
  Writer.AddTypeRef(T->getOriginalType(), Record);
  Code = TYPE_DECAYED;
}

void ASTTypeWriter::VisitAdjustedType(const AdjustedType *T) {
  Writer.AddTypeRef(T->getOriginalType(), Record);
  Writer.AddTypeRef(T->getAdjustedType(), Record);
  Code = TYPE_ADJUSTED;
}

void ASTTypeWriter::VisitBlockPointerType(const BlockPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = TYPE_BLOCK_POINTER;
}

void ASTTypeWriter::VisitLValueReferenceType(const LValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeTypeAsWritten(), Record);
  Record.push_back(T->isSpelledAsLValue());
  Code = TYPE_LVALUE_REFERENCE;
}

void ASTTypeWriter::VisitRValueReferenceType(const RValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeTypeAsWritten(), Record);
  Code = TYPE_RVALUE_REFERENCE;
}

void ASTTypeWriter::VisitMemberPointerType(const MemberPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Writer.AddTypeRef(QualType(T->getClass(), 0), Record);
  Code = TYPE_MEMBER_POINTER;
}

void ASTTypeWriter::VisitArrayType(const ArrayType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getSizeModifier()); // FIXME: stable values
  Record.push_back(T->getIndexTypeCVRQualifiers()); // FIXME: stable values
}

void ASTTypeWriter::VisitConstantArrayType(const ConstantArrayType *T) {
  VisitArrayType(T);
  Writer.AddAPInt(T->getSize(), Record);
  Code = TYPE_CONSTANT_ARRAY;
}

void ASTTypeWriter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  VisitArrayType(T);
  Code = TYPE_INCOMPLETE_ARRAY;
}

void ASTTypeWriter::VisitVariableArrayType(const VariableArrayType *T) {
  VisitArrayType(T);
  Writer.AddSourceLocation(T->getLBracketLoc(), Record);
  Writer.AddSourceLocation(T->getRBracketLoc(), Record);
  Writer.AddStmt(T->getSizeExpr());
  Code = TYPE_VARIABLE_ARRAY;
}

void ASTTypeWriter::VisitVectorType(const VectorType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getNumElements());
  Record.push_back(T->getVectorKind());
  Code = TYPE_VECTOR;
}

void ASTTypeWriter::VisitExtVectorType(const ExtVectorType *T) {
  VisitVectorType(T);
  Code = TYPE_EXT_VECTOR;
}

void ASTTypeWriter::VisitFunctionType(const FunctionType *T) {
  Writer.AddTypeRef(T->getReturnType(), Record);
  FunctionType::ExtInfo C = T->getExtInfo();
  Record.push_back(C.getNoReturn());
  Record.push_back(C.getHasRegParm());
  Record.push_back(C.getRegParm());
  // FIXME: need to stabilize encoding of calling convention...
  Record.push_back(C.getCC());
  Record.push_back(C.getProducesResult());
}

void ASTTypeWriter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  VisitFunctionType(T);
  Code = TYPE_FUNCTION_NO_PROTO;
}

static void addExceptionSpec(ASTWriter &Writer, const FunctionProtoType *T,
                             ASTWriter::RecordDataImpl &Record) {
  Record.push_back(T->getExceptionSpecType());
  if (T->getExceptionSpecType() == EST_Dynamic) {
    Record.push_back(T->getNumExceptions());
    for (unsigned I = 0, N = T->getNumExceptions(); I != N; ++I)
      Writer.AddTypeRef(T->getExceptionType(I), Record);
  } else if (T->getExceptionSpecType() == EST_ComputedNoexcept) {
    Writer.AddStmt(T->getNoexceptExpr());
  } else if (T->getExceptionSpecType() == EST_Uninstantiated) {
    Writer.AddDeclRef(T->getExceptionSpecDecl(), Record);
    Writer.AddDeclRef(T->getExceptionSpecTemplate(), Record);
  } else if (T->getExceptionSpecType() == EST_Unevaluated) {
    Writer.AddDeclRef(T->getExceptionSpecDecl(), Record);
  }
}

void ASTTypeWriter::VisitFunctionProtoType(const FunctionProtoType *T) {
  VisitFunctionType(T);
  Record.push_back(T->getNumParams());
  for (unsigned I = 0, N = T->getNumParams(); I != N; ++I)
    Writer.AddTypeRef(T->getParamType(I), Record);
  Record.push_back(T->isVariadic());
  Record.push_back(T->hasTrailingReturn());
  Record.push_back(T->getTypeQuals());
  Record.push_back(static_cast<unsigned>(T->getRefQualifier()));
  addExceptionSpec(Writer, T, Record);
  Code = TYPE_FUNCTION_PROTO;
}

void ASTTypeWriter::VisitUnresolvedUsingType(const UnresolvedUsingType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = TYPE_UNRESOLVED_USING;
}

void ASTTypeWriter::VisitTypedefType(const TypedefType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  assert(!T->isCanonicalUnqualified() && "Invalid typedef ?");
  Writer.AddTypeRef(T->getCanonicalTypeInternal(), Record);
  Code = TYPE_TYPEDEF;
}

void ASTTypeWriter::VisitTypeOfExprType(const TypeOfExprType *T) {
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = TYPE_TYPEOF_EXPR;
}

void ASTTypeWriter::VisitTypeOfType(const TypeOfType *T) {
  Writer.AddTypeRef(T->getUnderlyingType(), Record);
  Code = TYPE_TYPEOF;
}

void ASTTypeWriter::VisitDecltypeType(const DecltypeType *T) {
  Writer.AddTypeRef(T->getUnderlyingType(), Record);
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = TYPE_DECLTYPE;
}

void ASTTypeWriter::VisitUnaryTransformType(const UnaryTransformType *T) {
  Writer.AddTypeRef(T->getBaseType(), Record);
  Writer.AddTypeRef(T->getUnderlyingType(), Record);
  Record.push_back(T->getUTTKind());
  Code = TYPE_UNARY_TRANSFORM;
}

void ASTTypeWriter::VisitAutoType(const AutoType *T) {
  Writer.AddTypeRef(T->getDeducedType(), Record);
  Record.push_back(T->isDecltypeAuto());
  if (T->getDeducedType().isNull())
    Record.push_back(T->isDependentType());
  Code = TYPE_AUTO;
}

void ASTTypeWriter::VisitTagType(const TagType *T) {
  Record.push_back(T->isDependentType());
  Writer.AddDeclRef(T->getDecl()->getCanonicalDecl(), Record);
  assert(!T->isBeingDefined() &&
         "Cannot serialize in the middle of a type definition");
}

void ASTTypeWriter::VisitRecordType(const RecordType *T) {
  VisitTagType(T);
  Code = TYPE_RECORD;
}

void ASTTypeWriter::VisitEnumType(const EnumType *T) {
  VisitTagType(T);
  Code = TYPE_ENUM;
}

void ASTTypeWriter::VisitAttributedType(const AttributedType *T) {
  Writer.AddTypeRef(T->getModifiedType(), Record);
  Writer.AddTypeRef(T->getEquivalentType(), Record);
  Record.push_back(T->getAttrKind());
  Code = TYPE_ATTRIBUTED;
}

void
ASTTypeWriter::VisitSubstTemplateTypeParmType(
                                        const SubstTemplateTypeParmType *T) {
  Writer.AddTypeRef(QualType(T->getReplacedParameter(), 0), Record);
  Writer.AddTypeRef(T->getReplacementType(), Record);
  Code = TYPE_SUBST_TEMPLATE_TYPE_PARM;
}

void
ASTTypeWriter::VisitSubstTemplateTypeParmPackType(
                                      const SubstTemplateTypeParmPackType *T) {
  Writer.AddTypeRef(QualType(T->getReplacedParameter(), 0), Record);
  Writer.AddTemplateArgument(T->getArgumentPack(), Record);
  Code = TYPE_SUBST_TEMPLATE_TYPE_PARM_PACK;
}

void
ASTTypeWriter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  Record.push_back(T->isDependentType());
  Writer.AddTemplateName(T->getTemplateName(), Record);
  Record.push_back(T->getNumArgs());
  for (TemplateSpecializationType::iterator ArgI = T->begin(), ArgE = T->end();
         ArgI != ArgE; ++ArgI)
    Writer.AddTemplateArgument(*ArgI, Record);
  Writer.AddTypeRef(T->isTypeAlias() ? T->getAliasedType() :
                    T->isCanonicalUnqualified() ? QualType()
                                                : T->getCanonicalTypeInternal(),
                    Record);
  Code = TYPE_TEMPLATE_SPECIALIZATION;
}

void
ASTTypeWriter::VisitDependentSizedArrayType(const DependentSizedArrayType *T) {
  VisitArrayType(T);
  Writer.AddStmt(T->getSizeExpr());
  Writer.AddSourceRange(T->getBracketsRange(), Record);
  Code = TYPE_DEPENDENT_SIZED_ARRAY;
}

void
ASTTypeWriter::VisitDependentSizedExtVectorType(
                                        const DependentSizedExtVectorType *T) {
  // FIXME: Serialize this type (C++ only)
  llvm_unreachable("Cannot serialize dependent sized extended vector types");
}

void
ASTTypeWriter::VisitTemplateTypeParmType(const TemplateTypeParmType *T) {
  Record.push_back(T->getDepth());
  Record.push_back(T->getIndex());
  Record.push_back(T->isParameterPack());
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = TYPE_TEMPLATE_TYPE_PARM;
}

void
ASTTypeWriter::VisitDependentNameType(const DependentNameType *T) {
  Record.push_back(T->getKeyword());
  Writer.AddNestedNameSpecifier(T->getQualifier(), Record);
  Writer.AddIdentifierRef(T->getIdentifier(), Record);
  Writer.AddTypeRef(T->isCanonicalUnqualified() ? QualType()
                                                : T->getCanonicalTypeInternal(),
                    Record);
  Code = TYPE_DEPENDENT_NAME;
}

void
ASTTypeWriter::VisitDependentTemplateSpecializationType(
                                const DependentTemplateSpecializationType *T) {
  Record.push_back(T->getKeyword());
  Writer.AddNestedNameSpecifier(T->getQualifier(), Record);
  Writer.AddIdentifierRef(T->getIdentifier(), Record);
  Record.push_back(T->getNumArgs());
  for (DependentTemplateSpecializationType::iterator
         I = T->begin(), E = T->end(); I != E; ++I)
    Writer.AddTemplateArgument(*I, Record);
  Code = TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION;
}

void ASTTypeWriter::VisitPackExpansionType(const PackExpansionType *T) {
  Writer.AddTypeRef(T->getPattern(), Record);
  if (Optional<unsigned> NumExpansions = T->getNumExpansions())
    Record.push_back(*NumExpansions + 1);
  else
    Record.push_back(0);
  Code = TYPE_PACK_EXPANSION;
}

void ASTTypeWriter::VisitParenType(const ParenType *T) {
  Writer.AddTypeRef(T->getInnerType(), Record);
  Code = TYPE_PAREN;
}

void ASTTypeWriter::VisitElaboratedType(const ElaboratedType *T) {
  Record.push_back(T->getKeyword());
  Writer.AddNestedNameSpecifier(T->getQualifier(), Record);
  Writer.AddTypeRef(T->getNamedType(), Record);
  Code = TYPE_ELABORATED;
}

void ASTTypeWriter::VisitInjectedClassNameType(const InjectedClassNameType *T) {
  Writer.AddDeclRef(T->getDecl()->getCanonicalDecl(), Record);
  Writer.AddTypeRef(T->getInjectedSpecializationType(), Record);
  Code = TYPE_INJECTED_CLASS_NAME;
}

void ASTTypeWriter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  Writer.AddDeclRef(T->getDecl()->getCanonicalDecl(), Record);
  Code = TYPE_OBJC_INTERFACE;
}

void ASTTypeWriter::VisitObjCObjectType(const ObjCObjectType *T) {
  Writer.AddTypeRef(T->getBaseType(), Record);
  Record.push_back(T->getNumProtocols());
  for (const auto *I : T->quals())
    Writer.AddDeclRef(I, Record);
  Code = TYPE_OBJC_OBJECT;
}

void
ASTTypeWriter::VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = TYPE_OBJC_OBJECT_POINTER;
}

void
ASTTypeWriter::VisitAtomicType(const AtomicType *T) {
  Writer.AddTypeRef(T->getValueType(), Record);
  Code = TYPE_ATOMIC;
}

namespace {

class TypeLocWriter : public TypeLocVisitor<TypeLocWriter> {
  ASTWriter &Writer;
  ASTWriter::RecordDataImpl &Record;

public:
  TypeLocWriter(ASTWriter &Writer, ASTWriter::RecordDataImpl &Record)
    : Writer(Writer), Record(Record) { }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    void Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc);
#include "clang/AST/TypeLocNodes.def"

  void VisitArrayTypeLoc(ArrayTypeLoc TyLoc);
  void VisitFunctionTypeLoc(FunctionTypeLoc TyLoc);
};

}

void TypeLocWriter::VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
  // nothing to do
}
void TypeLocWriter::VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
  Writer.AddSourceLocation(TL.getBuiltinLoc(), Record);
  if (TL.needsExtraLocalData()) {
    Record.push_back(TL.getWrittenTypeSpec());
    Record.push_back(TL.getWrittenSignSpec());
    Record.push_back(TL.getWrittenWidthSpec());
    Record.push_back(TL.hasModeAttr());
  }
}
void TypeLocWriter::VisitComplexTypeLoc(ComplexTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitPointerTypeLoc(PointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getStarLoc(), Record);
}
void TypeLocWriter::VisitDecayedTypeLoc(DecayedTypeLoc TL) {
  // nothing to do
}
void TypeLocWriter::VisitAdjustedTypeLoc(AdjustedTypeLoc TL) {
  // nothing to do
}
void TypeLocWriter::VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getCaretLoc(), Record);
}
void TypeLocWriter::VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
  Writer.AddSourceLocation(TL.getAmpLoc(), Record);
}
void TypeLocWriter::VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
  Writer.AddSourceLocation(TL.getAmpAmpLoc(), Record);
}
void TypeLocWriter::VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getStarLoc(), Record);
  Writer.AddTypeSourceInfo(TL.getClassTInfo(), Record);
}
void TypeLocWriter::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  Writer.AddSourceLocation(TL.getLBracketLoc(), Record);
  Writer.AddSourceLocation(TL.getRBracketLoc(), Record);
  Record.push_back(TL.getSizeExpr() ? 1 : 0);
  if (TL.getSizeExpr())
    Writer.AddStmt(TL.getSizeExpr());
}
void TypeLocWriter::VisitConstantArrayTypeLoc(ConstantArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitIncompleteArrayTypeLoc(IncompleteArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitVariableArrayTypeLoc(VariableArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitDependentSizedArrayTypeLoc(
                                            DependentSizedArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitDependentSizedExtVectorTypeLoc(
                                        DependentSizedExtVectorTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitVectorTypeLoc(VectorTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitExtVectorTypeLoc(ExtVectorTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitFunctionTypeLoc(FunctionTypeLoc TL) {
  Writer.AddSourceLocation(TL.getLocalRangeBegin(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
  Writer.AddSourceLocation(TL.getLocalRangeEnd(), Record);
  for (unsigned i = 0, e = TL.getNumParams(); i != e; ++i)
    Writer.AddDeclRef(TL.getParam(i), Record);
}
void TypeLocWriter::VisitFunctionProtoTypeLoc(FunctionProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocWriter::VisitFunctionNoProtoTypeLoc(FunctionNoProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocWriter::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTypedefTypeLoc(TypedefTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
  Writer.AddSourceLocation(TL.getTypeofLoc(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
}
void TypeLocWriter::VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
  Writer.AddSourceLocation(TL.getTypeofLoc(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
  Writer.AddTypeSourceInfo(TL.getUnderlyingTInfo(), Record);
}
void TypeLocWriter::VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitUnaryTransformTypeLoc(UnaryTransformTypeLoc TL) {
  Writer.AddSourceLocation(TL.getKWLoc(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
  Writer.AddTypeSourceInfo(TL.getUnderlyingTInfo(), Record);
}
void TypeLocWriter::VisitAutoTypeLoc(AutoTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitRecordTypeLoc(RecordTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitEnumTypeLoc(EnumTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitAttributedTypeLoc(AttributedTypeLoc TL) {
  Writer.AddSourceLocation(TL.getAttrNameLoc(), Record);
  if (TL.hasAttrOperand()) {
    SourceRange range = TL.getAttrOperandParensRange();
    Writer.AddSourceLocation(range.getBegin(), Record);
    Writer.AddSourceLocation(range.getEnd(), Record);
  }
  if (TL.hasAttrExprOperand()) {
    Expr *operand = TL.getAttrExprOperand();
    Record.push_back(operand ? 1 : 0);
    if (operand) Writer.AddStmt(operand);
  } else if (TL.hasAttrEnumOperand()) {
    Writer.AddSourceLocation(TL.getAttrEnumOperandLoc(), Record);
  }
}
void TypeLocWriter::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitSubstTemplateTypeParmTypeLoc(
                                            SubstTemplateTypeParmTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitSubstTemplateTypeParmPackTypeLoc(
                                          SubstTemplateTypeParmPackTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTemplateSpecializationTypeLoc(
                                           TemplateSpecializationTypeLoc TL) {
  Writer.AddSourceLocation(TL.getTemplateKeywordLoc(), Record);
  Writer.AddSourceLocation(TL.getTemplateNameLoc(), Record);
  Writer.AddSourceLocation(TL.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TL.getRAngleLoc(), Record);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    Writer.AddTemplateArgumentLocInfo(TL.getArgLoc(i).getArgument().getKind(),
                                      TL.getArgLoc(i).getLocInfo(), Record);
}
void TypeLocWriter::VisitParenTypeLoc(ParenTypeLoc TL) {
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
}
void TypeLocWriter::VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
  Writer.AddSourceLocation(TL.getElaboratedKeywordLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(TL.getQualifierLoc(), Record);
}
void TypeLocWriter::VisitInjectedClassNameTypeLoc(InjectedClassNameTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
  Writer.AddSourceLocation(TL.getElaboratedKeywordLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(TL.getQualifierLoc(), Record);
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitDependentTemplateSpecializationTypeLoc(
       DependentTemplateSpecializationTypeLoc TL) {
  Writer.AddSourceLocation(TL.getElaboratedKeywordLoc(), Record);
  Writer.AddNestedNameSpecifierLoc(TL.getQualifierLoc(), Record);
  Writer.AddSourceLocation(TL.getTemplateKeywordLoc(), Record);
  Writer.AddSourceLocation(TL.getTemplateNameLoc(), Record);
  Writer.AddSourceLocation(TL.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TL.getRAngleLoc(), Record);
  for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I)
    Writer.AddTemplateArgumentLocInfo(TL.getArgLoc(I).getArgument().getKind(),
                                      TL.getArgLoc(I).getLocInfo(), Record);
}
void TypeLocWriter::VisitPackExpansionTypeLoc(PackExpansionTypeLoc TL) {
  Writer.AddSourceLocation(TL.getEllipsisLoc(), Record);
}
void TypeLocWriter::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
  Record.push_back(TL.hasBaseTypeAsWritten());
  Writer.AddSourceLocation(TL.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TL.getRAngleLoc(), Record);
  for (unsigned i = 0, e = TL.getNumProtocols(); i != e; ++i)
    Writer.AddSourceLocation(TL.getProtocolLoc(i), Record);
}
void TypeLocWriter::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getStarLoc(), Record);
}
void TypeLocWriter::VisitAtomicTypeLoc(AtomicTypeLoc TL) {
  Writer.AddSourceLocation(TL.getKWLoc(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
}

//===----------------------------------------------------------------------===//
// ASTWriter Implementation
//===----------------------------------------------------------------------===//

static void EmitBlockID(unsigned ID, const char *Name,
                        llvm::BitstreamWriter &Stream,
                        ASTWriter::RecordDataImpl &Record) {
  Record.clear();
  Record.push_back(ID);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETBID, Record);

  // Emit the block name if present.
  if (!Name || Name[0] == 0)
    return;
  Record.clear();
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_BLOCKNAME, Record);
}

static void EmitRecordID(unsigned ID, const char *Name,
                         llvm::BitstreamWriter &Stream,
                         ASTWriter::RecordDataImpl &Record) {
  Record.clear();
  Record.push_back(ID);
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETRECORDNAME, Record);
}

static void AddStmtsExprs(llvm::BitstreamWriter &Stream,
                          ASTWriter::RecordDataImpl &Record) {
#define RECORD(X) EmitRecordID(X, #X, Stream, Record)
  RECORD(STMT_STOP);
  RECORD(STMT_NULL_PTR);
  RECORD(STMT_NULL);
  RECORD(STMT_COMPOUND);
  RECORD(STMT_CASE);
  RECORD(STMT_DEFAULT);
  RECORD(STMT_LABEL);
  RECORD(STMT_ATTRIBUTED);
  RECORD(STMT_IF);
  RECORD(STMT_SWITCH);
  RECORD(STMT_WHILE);
  RECORD(STMT_DO);
  RECORD(STMT_FOR);
  RECORD(STMT_GOTO);
  RECORD(STMT_INDIRECT_GOTO);
  RECORD(STMT_CONTINUE);
  RECORD(STMT_BREAK);
  RECORD(STMT_RETURN);
  RECORD(STMT_DECL);
  RECORD(STMT_GCCASM);
  RECORD(STMT_MSASM);
  RECORD(EXPR_PREDEFINED);
  RECORD(EXPR_DECL_REF);
  RECORD(EXPR_INTEGER_LITERAL);
  RECORD(EXPR_FLOATING_LITERAL);
  RECORD(EXPR_IMAGINARY_LITERAL);
  RECORD(EXPR_STRING_LITERAL);
  RECORD(EXPR_CHARACTER_LITERAL);
  RECORD(EXPR_PAREN);
  RECORD(EXPR_UNARY_OPERATOR);
  RECORD(EXPR_SIZEOF_ALIGN_OF);
  RECORD(EXPR_ARRAY_SUBSCRIPT);
  RECORD(EXPR_CALL);
  RECORD(EXPR_MEMBER);
  RECORD(EXPR_BINARY_OPERATOR);
  RECORD(EXPR_COMPOUND_ASSIGN_OPERATOR);
  RECORD(EXPR_CONDITIONAL_OPERATOR);
  RECORD(EXPR_IMPLICIT_CAST);
  RECORD(EXPR_CSTYLE_CAST);
  RECORD(EXPR_COMPOUND_LITERAL);
  RECORD(EXPR_EXT_VECTOR_ELEMENT);
  RECORD(EXPR_INIT_LIST);
  RECORD(EXPR_DESIGNATED_INIT);
  RECORD(EXPR_IMPLICIT_VALUE_INIT);
  RECORD(EXPR_VA_ARG);
  RECORD(EXPR_ADDR_LABEL);
  RECORD(EXPR_STMT);
  RECORD(EXPR_CHOOSE);
  RECORD(EXPR_GNU_NULL);
  RECORD(EXPR_SHUFFLE_VECTOR);
  RECORD(EXPR_BLOCK);
  RECORD(EXPR_GENERIC_SELECTION);
  RECORD(EXPR_OBJC_STRING_LITERAL);
  RECORD(EXPR_OBJC_BOXED_EXPRESSION);
  RECORD(EXPR_OBJC_ARRAY_LITERAL);
  RECORD(EXPR_OBJC_DICTIONARY_LITERAL);
  RECORD(EXPR_OBJC_ENCODE);
  RECORD(EXPR_OBJC_SELECTOR_EXPR);
  RECORD(EXPR_OBJC_PROTOCOL_EXPR);
  RECORD(EXPR_OBJC_IVAR_REF_EXPR);
  RECORD(EXPR_OBJC_PROPERTY_REF_EXPR);
  RECORD(EXPR_OBJC_KVC_REF_EXPR);
  RECORD(EXPR_OBJC_MESSAGE_EXPR);
  RECORD(STMT_OBJC_FOR_COLLECTION);
  RECORD(STMT_OBJC_CATCH);
  RECORD(STMT_OBJC_FINALLY);
  RECORD(STMT_OBJC_AT_TRY);
  RECORD(STMT_OBJC_AT_SYNCHRONIZED);
  RECORD(STMT_OBJC_AT_THROW);
  RECORD(EXPR_OBJC_BOOL_LITERAL);
  RECORD(EXPR_CXX_OPERATOR_CALL);
  RECORD(EXPR_CXX_CONSTRUCT);
  RECORD(EXPR_CXX_STATIC_CAST);
  RECORD(EXPR_CXX_DYNAMIC_CAST);
  RECORD(EXPR_CXX_REINTERPRET_CAST);
  RECORD(EXPR_CXX_CONST_CAST);
  RECORD(EXPR_CXX_FUNCTIONAL_CAST);
  RECORD(EXPR_USER_DEFINED_LITERAL);
  RECORD(EXPR_CXX_STD_INITIALIZER_LIST);
  RECORD(EXPR_CXX_BOOL_LITERAL);
  RECORD(EXPR_CXX_NULL_PTR_LITERAL);
  RECORD(EXPR_CXX_TYPEID_EXPR);
  RECORD(EXPR_CXX_TYPEID_TYPE);
  RECORD(EXPR_CXX_UUIDOF_EXPR);
  RECORD(EXPR_CXX_UUIDOF_TYPE);
  RECORD(EXPR_CXX_THIS);
  RECORD(EXPR_CXX_THROW);
  RECORD(EXPR_CXX_DEFAULT_ARG);
  RECORD(EXPR_CXX_BIND_TEMPORARY);
  RECORD(EXPR_CXX_SCALAR_VALUE_INIT);
  RECORD(EXPR_CXX_NEW);
  RECORD(EXPR_CXX_DELETE);
  RECORD(EXPR_CXX_PSEUDO_DESTRUCTOR);
  RECORD(EXPR_EXPR_WITH_CLEANUPS);
  RECORD(EXPR_CXX_DEPENDENT_SCOPE_MEMBER);
  RECORD(EXPR_CXX_DEPENDENT_SCOPE_DECL_REF);
  RECORD(EXPR_CXX_UNRESOLVED_CONSTRUCT);
  RECORD(EXPR_CXX_UNRESOLVED_MEMBER);
  RECORD(EXPR_CXX_UNRESOLVED_LOOKUP);
  RECORD(EXPR_CXX_NOEXCEPT);
  RECORD(EXPR_OPAQUE_VALUE);
  RECORD(EXPR_PACK_EXPANSION);
  RECORD(EXPR_SIZEOF_PACK);
  RECORD(EXPR_SUBST_NON_TYPE_TEMPLATE_PARM_PACK);
  RECORD(EXPR_CUDA_KERNEL_CALL);
#undef RECORD
}

void ASTWriter::WriteBlockInfoBlock() {
  RecordData Record;
  Stream.EnterSubblock(llvm::bitc::BLOCKINFO_BLOCK_ID, 3);

#define BLOCK(X) EmitBlockID(X ## _ID, #X, Stream, Record)
#define RECORD(X) EmitRecordID(X, #X, Stream, Record)

  // Control Block.
  BLOCK(CONTROL_BLOCK);
  RECORD(METADATA);
  RECORD(MODULE_NAME);
  RECORD(MODULE_MAP_FILE);
  RECORD(IMPORTS);
  RECORD(LANGUAGE_OPTIONS);
  RECORD(TARGET_OPTIONS);
  RECORD(ORIGINAL_FILE);
  RECORD(ORIGINAL_PCH_DIR);
  RECORD(ORIGINAL_FILE_ID);
  RECORD(INPUT_FILE_OFFSETS);
  RECORD(DIAGNOSTIC_OPTIONS);
  RECORD(FILE_SYSTEM_OPTIONS);
  RECORD(HEADER_SEARCH_OPTIONS);
  RECORD(PREPROCESSOR_OPTIONS);

  BLOCK(INPUT_FILES_BLOCK);
  RECORD(INPUT_FILE);

  // AST Top-Level Block.
  BLOCK(AST_BLOCK);
  RECORD(TYPE_OFFSET);
  RECORD(DECL_OFFSET);
  RECORD(IDENTIFIER_OFFSET);
  RECORD(IDENTIFIER_TABLE);
  RECORD(EAGERLY_DESERIALIZED_DECLS);
  RECORD(SPECIAL_TYPES);
  RECORD(STATISTICS);
  RECORD(TENTATIVE_DEFINITIONS);
  RECORD(UNUSED_FILESCOPED_DECLS);
  RECORD(LOCALLY_SCOPED_EXTERN_C_DECLS);
  RECORD(SELECTOR_OFFSETS);
  RECORD(METHOD_POOL);
  RECORD(PP_COUNTER_VALUE);
  RECORD(SOURCE_LOCATION_OFFSETS);
  RECORD(SOURCE_LOCATION_PRELOADS);
  RECORD(EXT_VECTOR_DECLS);
  RECORD(PPD_ENTITIES_OFFSETS);
  RECORD(REFERENCED_SELECTOR_POOL);
  RECORD(TU_UPDATE_LEXICAL);
  RECORD(LOCAL_REDECLARATIONS_MAP);
  RECORD(SEMA_DECL_REFS);
  RECORD(WEAK_UNDECLARED_IDENTIFIERS);
  RECORD(PENDING_IMPLICIT_INSTANTIATIONS);
  RECORD(DECL_REPLACEMENTS);
  RECORD(UPDATE_VISIBLE);
  RECORD(DECL_UPDATE_OFFSETS);
  RECORD(DECL_UPDATES);
  RECORD(CXX_BASE_SPECIFIER_OFFSETS);
  RECORD(DIAG_PRAGMA_MAPPINGS);
  RECORD(CUDA_SPECIAL_DECL_REFS);
  RECORD(HEADER_SEARCH_TABLE);
  RECORD(FP_PRAGMA_OPTIONS);
  RECORD(OPENCL_EXTENSIONS);
  RECORD(DELEGATING_CTORS);
  RECORD(KNOWN_NAMESPACES);
  RECORD(UNDEFINED_BUT_USED);
  RECORD(MODULE_OFFSET_MAP);
  RECORD(SOURCE_MANAGER_LINE_TABLE);
  RECORD(OBJC_CATEGORIES_MAP);
  RECORD(FILE_SORTED_DECLS);
  RECORD(IMPORTED_MODULES);
  RECORD(MERGED_DECLARATIONS);
  RECORD(LOCAL_REDECLARATIONS);
  RECORD(OBJC_CATEGORIES);
  RECORD(MACRO_OFFSET);
  RECORD(MACRO_TABLE);
  RECORD(LATE_PARSED_TEMPLATE);
  RECORD(OPTIMIZE_PRAGMA_OPTIONS);

  // SourceManager Block.
  BLOCK(SOURCE_MANAGER_BLOCK);
  RECORD(SM_SLOC_FILE_ENTRY);
  RECORD(SM_SLOC_BUFFER_ENTRY);
  RECORD(SM_SLOC_BUFFER_BLOB);
  RECORD(SM_SLOC_EXPANSION_ENTRY);

  // Preprocessor Block.
  BLOCK(PREPROCESSOR_BLOCK);
  RECORD(PP_MACRO_OBJECT_LIKE);
  RECORD(PP_MACRO_FUNCTION_LIKE);
  RECORD(PP_TOKEN);
  
  // Decls and Types block.
  BLOCK(DECLTYPES_BLOCK);
  RECORD(TYPE_EXT_QUAL);
  RECORD(TYPE_COMPLEX);
  RECORD(TYPE_POINTER);
  RECORD(TYPE_BLOCK_POINTER);
  RECORD(TYPE_LVALUE_REFERENCE);
  RECORD(TYPE_RVALUE_REFERENCE);
  RECORD(TYPE_MEMBER_POINTER);
  RECORD(TYPE_CONSTANT_ARRAY);
  RECORD(TYPE_INCOMPLETE_ARRAY);
  RECORD(TYPE_VARIABLE_ARRAY);
  RECORD(TYPE_VECTOR);
  RECORD(TYPE_EXT_VECTOR);
  RECORD(TYPE_FUNCTION_PROTO);
  RECORD(TYPE_FUNCTION_NO_PROTO);
  RECORD(TYPE_TYPEDEF);
  RECORD(TYPE_TYPEOF_EXPR);
  RECORD(TYPE_TYPEOF);
  RECORD(TYPE_RECORD);
  RECORD(TYPE_ENUM);
  RECORD(TYPE_OBJC_INTERFACE);
  RECORD(TYPE_OBJC_OBJECT);
  RECORD(TYPE_OBJC_OBJECT_POINTER);
  RECORD(TYPE_DECLTYPE);
  RECORD(TYPE_ELABORATED);
  RECORD(TYPE_SUBST_TEMPLATE_TYPE_PARM);
  RECORD(TYPE_UNRESOLVED_USING);
  RECORD(TYPE_INJECTED_CLASS_NAME);
  RECORD(TYPE_OBJC_OBJECT);
  RECORD(TYPE_TEMPLATE_TYPE_PARM);
  RECORD(TYPE_TEMPLATE_SPECIALIZATION);
  RECORD(TYPE_DEPENDENT_NAME);
  RECORD(TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION);
  RECORD(TYPE_DEPENDENT_SIZED_ARRAY);
  RECORD(TYPE_PAREN);
  RECORD(TYPE_PACK_EXPANSION);
  RECORD(TYPE_ATTRIBUTED);
  RECORD(TYPE_SUBST_TEMPLATE_TYPE_PARM_PACK);
  RECORD(TYPE_ATOMIC);
  RECORD(DECL_TYPEDEF);
  RECORD(DECL_ENUM);
  RECORD(DECL_RECORD);
  RECORD(DECL_ENUM_CONSTANT);
  RECORD(DECL_FUNCTION);
  RECORD(DECL_OBJC_METHOD);
  RECORD(DECL_OBJC_INTERFACE);
  RECORD(DECL_OBJC_PROTOCOL);
  RECORD(DECL_OBJC_IVAR);
  RECORD(DECL_OBJC_AT_DEFS_FIELD);
  RECORD(DECL_OBJC_CATEGORY);
  RECORD(DECL_OBJC_CATEGORY_IMPL);
  RECORD(DECL_OBJC_IMPLEMENTATION);
  RECORD(DECL_OBJC_COMPATIBLE_ALIAS);
  RECORD(DECL_OBJC_PROPERTY);
  RECORD(DECL_OBJC_PROPERTY_IMPL);
  RECORD(DECL_FIELD);
  RECORD(DECL_MS_PROPERTY);
  RECORD(DECL_VAR);
  RECORD(DECL_IMPLICIT_PARAM);
  RECORD(DECL_PARM_VAR);
  RECORD(DECL_FILE_SCOPE_ASM);
  RECORD(DECL_BLOCK);
  RECORD(DECL_CONTEXT_LEXICAL);
  RECORD(DECL_CONTEXT_VISIBLE);
  RECORD(DECL_NAMESPACE);
  RECORD(DECL_NAMESPACE_ALIAS);
  RECORD(DECL_USING);
  RECORD(DECL_USING_SHADOW);
  RECORD(DECL_USING_DIRECTIVE);
  RECORD(DECL_UNRESOLVED_USING_VALUE);
  RECORD(DECL_UNRESOLVED_USING_TYPENAME);
  RECORD(DECL_LINKAGE_SPEC);
  RECORD(DECL_CXX_RECORD);
  RECORD(DECL_CXX_METHOD);
  RECORD(DECL_CXX_CONSTRUCTOR);
  RECORD(DECL_CXX_DESTRUCTOR);
  RECORD(DECL_CXX_CONVERSION);
  RECORD(DECL_ACCESS_SPEC);
  RECORD(DECL_FRIEND);
  RECORD(DECL_FRIEND_TEMPLATE);
  RECORD(DECL_CLASS_TEMPLATE);
  RECORD(DECL_CLASS_TEMPLATE_SPECIALIZATION);
  RECORD(DECL_CLASS_TEMPLATE_PARTIAL_SPECIALIZATION);
  RECORD(DECL_VAR_TEMPLATE);
  RECORD(DECL_VAR_TEMPLATE_SPECIALIZATION);
  RECORD(DECL_VAR_TEMPLATE_PARTIAL_SPECIALIZATION);
  RECORD(DECL_FUNCTION_TEMPLATE);
  RECORD(DECL_TEMPLATE_TYPE_PARM);
  RECORD(DECL_NON_TYPE_TEMPLATE_PARM);
  RECORD(DECL_TEMPLATE_TEMPLATE_PARM);
  RECORD(DECL_STATIC_ASSERT);
  RECORD(DECL_CXX_BASE_SPECIFIERS);
  RECORD(DECL_INDIRECTFIELD);
  RECORD(DECL_EXPANDED_NON_TYPE_TEMPLATE_PARM_PACK);
  
  // Statements and Exprs can occur in the Decls and Types block.
  AddStmtsExprs(Stream, Record);

  BLOCK(PREPROCESSOR_DETAIL_BLOCK);
  RECORD(PPD_MACRO_EXPANSION);
  RECORD(PPD_MACRO_DEFINITION);
  RECORD(PPD_INCLUSION_DIRECTIVE);
  
#undef RECORD
#undef BLOCK
  Stream.ExitBlock();
}

/// \brief Adjusts the given filename to only write out the portion of the
/// filename that is not part of the system root directory.
///
/// \param Filename the file name to adjust.
///
/// \param isysroot When non-NULL, the PCH file is a relocatable PCH file and
/// the returned filename will be adjusted by this system root.
///
/// \returns either the original filename (if it needs no adjustment) or the
/// adjusted filename (which points into the @p Filename parameter).
static const char *
adjustFilenameForRelocatablePCH(const char *Filename, StringRef isysroot) {
  assert(Filename && "No file name to adjust?");

  if (isysroot.empty())
    return Filename;

  // Verify that the filename and the system root have the same prefix.
  unsigned Pos = 0;
  for (; Filename[Pos] && Pos < isysroot.size(); ++Pos)
    if (Filename[Pos] != isysroot[Pos])
      return Filename; // Prefixes don't match.

  // We hit the end of the filename before we hit the end of the system root.
  if (!Filename[Pos])
    return Filename;

  // If the file name has a '/' at the current position, skip over the '/'.
  // We distinguish sysroot-based includes from absolute includes by the
  // absence of '/' at the beginning of sysroot-based includes.
  if (Filename[Pos] == '/')
    ++Pos;

  return Filename + Pos;
}

/// \brief Write the control block.
void ASTWriter::WriteControlBlock(Preprocessor &PP, ASTContext &Context,
                                  StringRef isysroot,
                                  const std::string &OutputFile) {
  using namespace llvm;
  Stream.EnterSubblock(CONTROL_BLOCK_ID, 5);
  RecordData Record;
  
  // Metadata
  BitCodeAbbrev *MetadataAbbrev = new BitCodeAbbrev();
  MetadataAbbrev->Add(BitCodeAbbrevOp(METADATA));
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Major
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Minor
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Clang maj.
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Clang min.
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Relocatable
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Errors
  MetadataAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // SVN branch/tag
  unsigned MetadataAbbrevCode = Stream.EmitAbbrev(MetadataAbbrev);
  Record.push_back(METADATA);
  Record.push_back(VERSION_MAJOR);
  Record.push_back(VERSION_MINOR);
  Record.push_back(CLANG_VERSION_MAJOR);
  Record.push_back(CLANG_VERSION_MINOR);
  Record.push_back(!isysroot.empty());
  Record.push_back(ASTHasCompilerErrors);
  Stream.EmitRecordWithBlob(MetadataAbbrevCode, Record,
                            getClangFullRepositoryVersion());

  // Module name
  if (WritingModule) {
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(MODULE_NAME));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
    unsigned AbbrevCode = Stream.EmitAbbrev(Abbrev);
    RecordData Record;
    Record.push_back(MODULE_NAME);
    Stream.EmitRecordWithBlob(AbbrevCode, Record, WritingModule->Name);
  }

  // Module map file
  if (WritingModule) {
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(MODULE_MAP_FILE));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Filename
    unsigned AbbrevCode = Stream.EmitAbbrev(Abbrev);

    assert(WritingModule->ModuleMap && "missing module map");
    SmallString<128> ModuleMap(WritingModule->ModuleMap->getName());
    llvm::sys::fs::make_absolute(ModuleMap);
    RecordData Record;
    Record.push_back(MODULE_MAP_FILE);
    Stream.EmitRecordWithBlob(AbbrevCode, Record, ModuleMap.str());
  }

  // Imports
  if (Chain) {
    serialization::ModuleManager &Mgr = Chain->getModuleManager();
    Record.clear();

    for (ModuleManager::ModuleIterator M = Mgr.begin(), MEnd = Mgr.end();
         M != MEnd; ++M) {
      // Skip modules that weren't directly imported.
      if (!(*M)->isDirectlyImported())
        continue;

      Record.push_back((unsigned)(*M)->Kind); // FIXME: Stable encoding
      AddSourceLocation((*M)->ImportLoc, Record);
      Record.push_back((*M)->File->getSize());
      Record.push_back((*M)->File->getModificationTime());
      const std::string &FileName = (*M)->FileName;
      Record.push_back(FileName.size());
      Record.append(FileName.begin(), FileName.end());
    }
    Stream.EmitRecord(IMPORTS, Record);
  }

  // Language options.
  Record.clear();
  const LangOptions &LangOpts = Context.getLangOpts();
#define LANGOPT(Name, Bits, Default, Description) \
  Record.push_back(LangOpts.Name);
#define ENUM_LANGOPT(Name, Type, Bits, Default, Description) \
  Record.push_back(static_cast<unsigned>(LangOpts.get##Name()));
#include "clang/Basic/LangOptions.def"  
#define SANITIZER(NAME, ID) Record.push_back(LangOpts.Sanitize.ID);
#include "clang/Basic/Sanitizers.def"

  Record.push_back((unsigned) LangOpts.ObjCRuntime.getKind());
  AddVersionTuple(LangOpts.ObjCRuntime.getVersion(), Record);
  
  Record.push_back(LangOpts.CurrentModule.size());
  Record.append(LangOpts.CurrentModule.begin(), LangOpts.CurrentModule.end());

  // Comment options.
  Record.push_back(LangOpts.CommentOpts.BlockCommandNames.size());
  for (CommentOptions::BlockCommandNamesTy::const_iterator
           I = LangOpts.CommentOpts.BlockCommandNames.begin(),
           IEnd = LangOpts.CommentOpts.BlockCommandNames.end();
       I != IEnd; ++I) {
    AddString(*I, Record);
  }
  Record.push_back(LangOpts.CommentOpts.ParseAllComments);

  Stream.EmitRecord(LANGUAGE_OPTIONS, Record);

  // Target options.
  Record.clear();
  const TargetInfo &Target = Context.getTargetInfo();
  const TargetOptions &TargetOpts = Target.getTargetOpts();
  AddString(TargetOpts.Triple, Record);
  AddString(TargetOpts.CPU, Record);
  AddString(TargetOpts.ABI, Record);
  Record.push_back(TargetOpts.FeaturesAsWritten.size());
  for (unsigned I = 0, N = TargetOpts.FeaturesAsWritten.size(); I != N; ++I) {
    AddString(TargetOpts.FeaturesAsWritten[I], Record);
  }
  Record.push_back(TargetOpts.Features.size());
  for (unsigned I = 0, N = TargetOpts.Features.size(); I != N; ++I) {
    AddString(TargetOpts.Features[I], Record);
  }
  Stream.EmitRecord(TARGET_OPTIONS, Record);

  // Diagnostic options.
  Record.clear();
  const DiagnosticOptions &DiagOpts
    = Context.getDiagnostics().getDiagnosticOptions();
#define DIAGOPT(Name, Bits, Default) Record.push_back(DiagOpts.Name);
#define ENUM_DIAGOPT(Name, Type, Bits, Default) \
  Record.push_back(static_cast<unsigned>(DiagOpts.get##Name()));
#include "clang/Basic/DiagnosticOptions.def"
  Record.push_back(DiagOpts.Warnings.size());
  for (unsigned I = 0, N = DiagOpts.Warnings.size(); I != N; ++I)
    AddString(DiagOpts.Warnings[I], Record);
  Record.push_back(DiagOpts.Remarks.size());
  for (unsigned I = 0, N = DiagOpts.Remarks.size(); I != N; ++I)
    AddString(DiagOpts.Remarks[I], Record);
  // Note: we don't serialize the log or serialization file names, because they
  // are generally transient files and will almost always be overridden.
  Stream.EmitRecord(DIAGNOSTIC_OPTIONS, Record);

  // File system options.
  Record.clear();
  const FileSystemOptions &FSOpts
    = Context.getSourceManager().getFileManager().getFileSystemOptions();
  AddString(FSOpts.WorkingDir, Record);
  Stream.EmitRecord(FILE_SYSTEM_OPTIONS, Record);

  // Header search options.
  Record.clear();
  const HeaderSearchOptions &HSOpts
    = PP.getHeaderSearchInfo().getHeaderSearchOpts();
  AddString(HSOpts.Sysroot, Record);

  // Include entries.
  Record.push_back(HSOpts.UserEntries.size());
  for (unsigned I = 0, N = HSOpts.UserEntries.size(); I != N; ++I) {
    const HeaderSearchOptions::Entry &Entry = HSOpts.UserEntries[I];
    AddString(Entry.Path, Record);
    Record.push_back(static_cast<unsigned>(Entry.Group));
    Record.push_back(Entry.IsFramework);
    Record.push_back(Entry.IgnoreSysRoot);
  }

  // System header prefixes.
  Record.push_back(HSOpts.SystemHeaderPrefixes.size());
  for (unsigned I = 0, N = HSOpts.SystemHeaderPrefixes.size(); I != N; ++I) {
    AddString(HSOpts.SystemHeaderPrefixes[I].Prefix, Record);
    Record.push_back(HSOpts.SystemHeaderPrefixes[I].IsSystemHeader);
  }

  AddString(HSOpts.ResourceDir, Record);
  AddString(HSOpts.ModuleCachePath, Record);
  AddString(HSOpts.ModuleUserBuildPath, Record);
  Record.push_back(HSOpts.DisableModuleHash);
  Record.push_back(HSOpts.UseBuiltinIncludes);
  Record.push_back(HSOpts.UseStandardSystemIncludes);
  Record.push_back(HSOpts.UseStandardCXXIncludes);
  Record.push_back(HSOpts.UseLibcxx);
  Stream.EmitRecord(HEADER_SEARCH_OPTIONS, Record);

  // Preprocessor options.
  Record.clear();
  const PreprocessorOptions &PPOpts = PP.getPreprocessorOpts();

  // Macro definitions.
  Record.push_back(PPOpts.Macros.size());
  for (unsigned I = 0, N = PPOpts.Macros.size(); I != N; ++I) {
    AddString(PPOpts.Macros[I].first, Record);
    Record.push_back(PPOpts.Macros[I].second);
  }

  // Includes
  Record.push_back(PPOpts.Includes.size());
  for (unsigned I = 0, N = PPOpts.Includes.size(); I != N; ++I)
    AddString(PPOpts.Includes[I], Record);

  // Macro includes
  Record.push_back(PPOpts.MacroIncludes.size());
  for (unsigned I = 0, N = PPOpts.MacroIncludes.size(); I != N; ++I)
    AddString(PPOpts.MacroIncludes[I], Record);

  Record.push_back(PPOpts.UsePredefines);
  // Detailed record is important since it is used for the module cache hash.
  Record.push_back(PPOpts.DetailedRecord);
  AddString(PPOpts.ImplicitPCHInclude, Record);
  AddString(PPOpts.ImplicitPTHInclude, Record);
  Record.push_back(static_cast<unsigned>(PPOpts.ObjCXXARCStandardLibrary));
  Stream.EmitRecord(PREPROCESSOR_OPTIONS, Record);

  // Original file name and file ID
  SourceManager &SM = Context.getSourceManager();
  if (const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID())) {
    BitCodeAbbrev *FileAbbrev = new BitCodeAbbrev();
    FileAbbrev->Add(BitCodeAbbrevOp(ORIGINAL_FILE));
    FileAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // File ID
    FileAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
    unsigned FileAbbrevCode = Stream.EmitAbbrev(FileAbbrev);

    SmallString<128> MainFilePath(MainFile->getName());

    llvm::sys::fs::make_absolute(MainFilePath);

    const char *MainFileNameStr = MainFilePath.c_str();
    MainFileNameStr = adjustFilenameForRelocatablePCH(MainFileNameStr,
                                                      isysroot);
    Record.clear();
    Record.push_back(ORIGINAL_FILE);
    Record.push_back(SM.getMainFileID().getOpaqueValue());
    Stream.EmitRecordWithBlob(FileAbbrevCode, Record, MainFileNameStr);
  }

  Record.clear();
  Record.push_back(SM.getMainFileID().getOpaqueValue());
  Stream.EmitRecord(ORIGINAL_FILE_ID, Record);

  // Original PCH directory
  if (!OutputFile.empty() && OutputFile != "-") {
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(ORIGINAL_PCH_DIR));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
    unsigned AbbrevCode = Stream.EmitAbbrev(Abbrev);

    SmallString<128> OutputPath(OutputFile);

    llvm::sys::fs::make_absolute(OutputPath);
    StringRef origDir = llvm::sys::path::parent_path(OutputPath);

    RecordData Record;
    Record.push_back(ORIGINAL_PCH_DIR);
    Stream.EmitRecordWithBlob(AbbrevCode, Record, origDir);
  }

  WriteInputFiles(Context.SourceMgr,
                  PP.getHeaderSearchInfo().getHeaderSearchOpts(),
                  isysroot,
                  PP.getLangOpts().Modules);
  Stream.ExitBlock();
}

namespace  {
  /// \brief An input file.
  struct InputFileEntry {
    const FileEntry *File;
    bool IsSystemFile;
    bool BufferOverridden;
  };
}

void ASTWriter::WriteInputFiles(SourceManager &SourceMgr,
                                HeaderSearchOptions &HSOpts,
                                StringRef isysroot,
                                bool Modules) {
  using namespace llvm;
  Stream.EnterSubblock(INPUT_FILES_BLOCK_ID, 4);
  RecordData Record;
  
  // Create input-file abbreviation.
  BitCodeAbbrev *IFAbbrev = new BitCodeAbbrev();
  IFAbbrev->Add(BitCodeAbbrevOp(INPUT_FILE));
  IFAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // ID
  IFAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 12)); // Size
  IFAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 32)); // Modification time
  IFAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Overridden
  IFAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
  unsigned IFAbbrevCode = Stream.EmitAbbrev(IFAbbrev);

  // Get all ContentCache objects for files, sorted by whether the file is a
  // system one or not. System files go at the back, users files at the front.
  std::deque<InputFileEntry> SortedFiles;
  for (unsigned I = 1, N = SourceMgr.local_sloc_entry_size(); I != N; ++I) {
    // Get this source location entry.
    const SrcMgr::SLocEntry *SLoc = &SourceMgr.getLocalSLocEntry(I);
    assert(&SourceMgr.getSLocEntry(FileID::get(I)) == SLoc);

    // We only care about file entries that were not overridden.
    if (!SLoc->isFile())
      continue;
    const SrcMgr::ContentCache *Cache = SLoc->getFile().getContentCache();
    if (!Cache->OrigEntry)
      continue;

    InputFileEntry Entry;
    Entry.File = Cache->OrigEntry;
    Entry.IsSystemFile = Cache->IsSystemFile;
    Entry.BufferOverridden = Cache->BufferOverridden;
    if (Cache->IsSystemFile)
      SortedFiles.push_back(Entry);
    else
      SortedFiles.push_front(Entry);
  }

  unsigned UserFilesNum = 0;
  // Write out all of the input files.
  std::vector<uint32_t> InputFileOffsets;
  for (std::deque<InputFileEntry>::iterator
         I = SortedFiles.begin(), E = SortedFiles.end(); I != E; ++I) {
    const InputFileEntry &Entry = *I;

    uint32_t &InputFileID = InputFileIDs[Entry.File];
    if (InputFileID != 0)
      continue; // already recorded this file.

    // Record this entry's offset.
    InputFileOffsets.push_back(Stream.GetCurrentBitNo());

    InputFileID = InputFileOffsets.size();

    if (!Entry.IsSystemFile)
      ++UserFilesNum;

    Record.clear();
    Record.push_back(INPUT_FILE);
    Record.push_back(InputFileOffsets.size());

    // Emit size/modification time for this file.
    Record.push_back(Entry.File->getSize());
    Record.push_back(Entry.File->getModificationTime());

    // Whether this file was overridden.
    Record.push_back(Entry.BufferOverridden);

    // Turn the file name into an absolute path, if it isn't already.
    const char *Filename = Entry.File->getName();
    SmallString<128> FilePath(Filename);
    
    // Ask the file manager to fixup the relative path for us. This will 
    // honor the working directory.
    SourceMgr.getFileManager().FixupRelativePath(FilePath);
    
    // FIXME: This call to make_absolute shouldn't be necessary, the
    // call to FixupRelativePath should always return an absolute path.
    llvm::sys::fs::make_absolute(FilePath);
    Filename = FilePath.c_str();
    
    Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);

    Stream.EmitRecordWithBlob(IFAbbrevCode, Record, Filename);
  }  

  Stream.ExitBlock();

  // Create input file offsets abbreviation.
  BitCodeAbbrev *OffsetsAbbrev = new BitCodeAbbrev();
  OffsetsAbbrev->Add(BitCodeAbbrevOp(INPUT_FILE_OFFSETS));
  OffsetsAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // # input files
  OffsetsAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // # non-system
                                                                //   input files
  OffsetsAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));   // Array
  unsigned OffsetsAbbrevCode = Stream.EmitAbbrev(OffsetsAbbrev);

  // Write input file offsets.
  Record.clear();
  Record.push_back(INPUT_FILE_OFFSETS);
  Record.push_back(InputFileOffsets.size());
  Record.push_back(UserFilesNum);
  Stream.EmitRecordWithBlob(OffsetsAbbrevCode, Record, data(InputFileOffsets));
}

//===----------------------------------------------------------------------===//
// Source Manager Serialization
//===----------------------------------------------------------------------===//

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// file.
static unsigned CreateSLocFileAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SM_SLOC_FILE_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Include location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Characteristic
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Line directives
  // FileEntry fields.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Input File ID
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // NumCreatedFIDs
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 24)); // FirstDeclIndex
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // NumDecls
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// buffer.
static unsigned CreateSLocBufferAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SM_SLOC_BUFFER_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Include location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Characteristic
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Line directives
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Buffer name blob
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// buffer's blob.
static unsigned CreateSLocBufferBlobAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SM_SLOC_BUFFER_BLOB));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Blob
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a macro
/// expansion.
static unsigned CreateSLocExpansionAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SM_SLOC_EXPANSION_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Spelling location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Start location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // End location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Token length
  return Stream.EmitAbbrev(Abbrev);
}

namespace {
  // Trait used for the on-disk hash table of header search information.
  class HeaderFileInfoTrait {
    ASTWriter &Writer;
    const HeaderSearch &HS;
    
    // Keep track of the framework names we've used during serialization.
    SmallVector<char, 128> FrameworkStringData;
    llvm::StringMap<unsigned> FrameworkNameOffset;
    
  public:
    HeaderFileInfoTrait(ASTWriter &Writer, const HeaderSearch &HS)
      : Writer(Writer), HS(HS) { }
    
    struct key_type {
      const FileEntry *FE;
      const char *Filename;
    };
    typedef const key_type &key_type_ref;
    
    typedef HeaderFileInfo data_type;
    typedef const data_type &data_type_ref;
    typedef unsigned hash_value_type;
    typedef unsigned offset_type;
    
    static hash_value_type ComputeHash(key_type_ref key) {
      // The hash is based only on size/time of the file, so that the reader can
      // match even when symlinking or excess path elements ("foo/../", "../")
      // change the form of the name. However, complete path is still the key.
      return llvm::hash_combine(key.FE->getSize(),
                                key.FE->getModificationTime());
    }
    
    std::pair<unsigned,unsigned>
    EmitKeyDataLength(raw_ostream& Out, key_type_ref key, data_type_ref Data) {
      using namespace llvm::support;
      endian::Writer<little> Writer(Out);
      unsigned KeyLen = strlen(key.Filename) + 1 + 8 + 8;
      Writer.write<uint16_t>(KeyLen);
      unsigned DataLen = 1 + 2 + 4 + 4;
      if (Data.isModuleHeader)
        DataLen += 4;
      Writer.write<uint8_t>(DataLen);
      return std::make_pair(KeyLen, DataLen);
    }
    
    void EmitKey(raw_ostream& Out, key_type_ref key, unsigned KeyLen) {
      using namespace llvm::support;
      endian::Writer<little> LE(Out);
      LE.write<uint64_t>(key.FE->getSize());
      KeyLen -= 8;
      LE.write<uint64_t>(key.FE->getModificationTime());
      KeyLen -= 8;
      Out.write(key.Filename, KeyLen);
    }
    
    void EmitData(raw_ostream &Out, key_type_ref key,
                  data_type_ref Data, unsigned DataLen) {
      using namespace llvm::support;
      endian::Writer<little> LE(Out);
      uint64_t Start = Out.tell(); (void)Start;
      
      unsigned char Flags = (Data.HeaderRole << 6)
                          | (Data.isImport << 5)
                          | (Data.isPragmaOnce << 4)
                          | (Data.DirInfo << 2)
                          | (Data.Resolved << 1)
                          | Data.IndexHeaderMapHeader;
      LE.write<uint8_t>(Flags);
      LE.write<uint16_t>(Data.NumIncludes);
      
      if (!Data.ControllingMacro)
        LE.write<uint32_t>(Data.ControllingMacroID);
      else
        LE.write<uint32_t>(Writer.getIdentifierRef(Data.ControllingMacro));
      
      unsigned Offset = 0;
      if (!Data.Framework.empty()) {
        // If this header refers into a framework, save the framework name.
        llvm::StringMap<unsigned>::iterator Pos
          = FrameworkNameOffset.find(Data.Framework);
        if (Pos == FrameworkNameOffset.end()) {
          Offset = FrameworkStringData.size() + 1;
          FrameworkStringData.append(Data.Framework.begin(), 
                                     Data.Framework.end());
          FrameworkStringData.push_back(0);
          
          FrameworkNameOffset[Data.Framework] = Offset;
        } else
          Offset = Pos->second;
      }
      LE.write<uint32_t>(Offset);

      if (Data.isModuleHeader) {
        Module *Mod = HS.findModuleForHeader(key.FE).getModule();
        LE.write<uint32_t>(Writer.getExistingSubmoduleID(Mod));
      }

      assert(Out.tell() - Start == DataLen && "Wrong data length");
    }
    
    const char *strings_begin() const { return FrameworkStringData.begin(); }
    const char *strings_end() const { return FrameworkStringData.end(); }
  };
} // end anonymous namespace

/// \brief Write the header search block for the list of files that 
///
/// \param HS The header search structure to save.
void ASTWriter::WriteHeaderSearch(const HeaderSearch &HS, StringRef isysroot) {
  SmallVector<const FileEntry *, 16> FilesByUID;
  HS.getFileMgr().GetUniqueIDMapping(FilesByUID);
  
  if (FilesByUID.size() > HS.header_file_size())
    FilesByUID.resize(HS.header_file_size());
  
  HeaderFileInfoTrait GeneratorTrait(*this, HS);
  llvm::OnDiskChainedHashTableGenerator<HeaderFileInfoTrait> Generator;
  SmallVector<const char *, 4> SavedStrings;
  unsigned NumHeaderSearchEntries = 0;
  for (unsigned UID = 0, LastUID = FilesByUID.size(); UID != LastUID; ++UID) {
    const FileEntry *File = FilesByUID[UID];
    if (!File)
      continue;

    // Use HeaderSearch's getFileInfo to make sure we get the HeaderFileInfo
    // from the external source if it was not provided already.
    HeaderFileInfo HFI;
    if (!HS.tryGetFileInfo(File, HFI) ||
        (HFI.External && Chain) ||
        (HFI.isModuleHeader && !HFI.isCompilingModuleHeader))
      continue;

    // Turn the file name into an absolute path, if it isn't already.
    const char *Filename = File->getName();
    Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
      
    // If we performed any translation on the file name at all, we need to
    // save this string, since the generator will refer to it later.
    if (Filename != File->getName()) {
      Filename = strdup(Filename);
      SavedStrings.push_back(Filename);
    }
    
    HeaderFileInfoTrait::key_type key = { File, Filename };
    Generator.insert(key, HFI, GeneratorTrait);
    ++NumHeaderSearchEntries;
  }
  
  // Create the on-disk hash table in a buffer.
  SmallString<4096> TableData;
  uint32_t BucketOffset;
  {
    using namespace llvm::support;
    llvm::raw_svector_ostream Out(TableData);
    // Make sure that no bucket is at offset 0
    endian::Writer<little>(Out).write<uint32_t>(0);
    BucketOffset = Generator.Emit(Out, GeneratorTrait);
  }

  // Create a blob abbreviation
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(HEADER_SEARCH_TABLE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned TableAbbrev = Stream.EmitAbbrev(Abbrev);
  
  // Write the header search table
  RecordData Record;
  Record.push_back(HEADER_SEARCH_TABLE);
  Record.push_back(BucketOffset);
  Record.push_back(NumHeaderSearchEntries);
  Record.push_back(TableData.size());
  TableData.append(GeneratorTrait.strings_begin(),GeneratorTrait.strings_end());
  Stream.EmitRecordWithBlob(TableAbbrev, Record, TableData.str());
  
  // Free all of the strings we had to duplicate.
  for (unsigned I = 0, N = SavedStrings.size(); I != N; ++I)
    free(const_cast<char *>(SavedStrings[I]));
}

/// \brief Writes the block containing the serialized form of the
/// source manager.
///
/// TODO: We should probably use an on-disk hash table (stored in a
/// blob), indexed based on the file name, so that we only create
/// entries for files that we actually need. In the common case (no
/// errors), we probably won't have to create file entries for any of
/// the files in the AST.
void ASTWriter::WriteSourceManagerBlock(SourceManager &SourceMgr,
                                        const Preprocessor &PP,
                                        StringRef isysroot) {
  RecordData Record;

  // Enter the source manager block.
  Stream.EnterSubblock(SOURCE_MANAGER_BLOCK_ID, 3);

  // Abbreviations for the various kinds of source-location entries.
  unsigned SLocFileAbbrv = CreateSLocFileAbbrev(Stream);
  unsigned SLocBufferAbbrv = CreateSLocBufferAbbrev(Stream);
  unsigned SLocBufferBlobAbbrv = CreateSLocBufferBlobAbbrev(Stream);
  unsigned SLocExpansionAbbrv = CreateSLocExpansionAbbrev(Stream);

  // Write out the source location entry table. We skip the first
  // entry, which is always the same dummy entry.
  std::vector<uint32_t> SLocEntryOffsets;
  RecordData PreloadSLocs;
  SLocEntryOffsets.reserve(SourceMgr.local_sloc_entry_size() - 1);
  for (unsigned I = 1, N = SourceMgr.local_sloc_entry_size();
       I != N; ++I) {
    // Get this source location entry.
    const SrcMgr::SLocEntry *SLoc = &SourceMgr.getLocalSLocEntry(I);
    FileID FID = FileID::get(I);
    assert(&SourceMgr.getSLocEntry(FID) == SLoc);

    // Record the offset of this source-location entry.
    SLocEntryOffsets.push_back(Stream.GetCurrentBitNo());

    // Figure out which record code to use.
    unsigned Code;
    if (SLoc->isFile()) {
      const SrcMgr::ContentCache *Cache = SLoc->getFile().getContentCache();
      if (Cache->OrigEntry) {
        Code = SM_SLOC_FILE_ENTRY;
      } else
        Code = SM_SLOC_BUFFER_ENTRY;
    } else
      Code = SM_SLOC_EXPANSION_ENTRY;
    Record.clear();
    Record.push_back(Code);

    // Starting offset of this entry within this module, so skip the dummy.
    Record.push_back(SLoc->getOffset() - 2);
    if (SLoc->isFile()) {
      const SrcMgr::FileInfo &File = SLoc->getFile();
      Record.push_back(File.getIncludeLoc().getRawEncoding());
      Record.push_back(File.getFileCharacteristic()); // FIXME: stable encoding
      Record.push_back(File.hasLineDirectives());

      const SrcMgr::ContentCache *Content = File.getContentCache();
      if (Content->OrigEntry) {
        assert(Content->OrigEntry == Content->ContentsEntry &&
               "Writing to AST an overridden file is not supported");

        // The source location entry is a file. Emit input file ID.
        assert(InputFileIDs[Content->OrigEntry] != 0 && "Missed file entry");
        Record.push_back(InputFileIDs[Content->OrigEntry]);

        Record.push_back(File.NumCreatedFIDs);
        
        FileDeclIDsTy::iterator FDI = FileDeclIDs.find(FID);
        if (FDI != FileDeclIDs.end()) {
          Record.push_back(FDI->second->FirstDeclIndex);
          Record.push_back(FDI->second->DeclIDs.size());
        } else {
          Record.push_back(0);
          Record.push_back(0);
        }
        
        Stream.EmitRecordWithAbbrev(SLocFileAbbrv, Record);
        
        if (Content->BufferOverridden) {
          Record.clear();
          Record.push_back(SM_SLOC_BUFFER_BLOB);
          const llvm::MemoryBuffer *Buffer
            = Content->getBuffer(PP.getDiagnostics(), PP.getSourceManager());
          Stream.EmitRecordWithBlob(SLocBufferBlobAbbrv, Record,
                                    StringRef(Buffer->getBufferStart(),
                                              Buffer->getBufferSize() + 1));          
        }
      } else {
        // The source location entry is a buffer. The blob associated
        // with this entry contains the contents of the buffer.

        // We add one to the size so that we capture the trailing NULL
        // that is required by llvm::MemoryBuffer::getMemBuffer (on
        // the reader side).
        const llvm::MemoryBuffer *Buffer
          = Content->getBuffer(PP.getDiagnostics(), PP.getSourceManager());
        const char *Name = Buffer->getBufferIdentifier();
        Stream.EmitRecordWithBlob(SLocBufferAbbrv, Record,
                                  StringRef(Name, strlen(Name) + 1));
        Record.clear();
        Record.push_back(SM_SLOC_BUFFER_BLOB);
        Stream.EmitRecordWithBlob(SLocBufferBlobAbbrv, Record,
                                  StringRef(Buffer->getBufferStart(),
                                                  Buffer->getBufferSize() + 1));

        if (strcmp(Name, "<built-in>") == 0) {
          PreloadSLocs.push_back(SLocEntryOffsets.size());
        }
      }
    } else {
      // The source location entry is a macro expansion.
      const SrcMgr::ExpansionInfo &Expansion = SLoc->getExpansion();
      Record.push_back(Expansion.getSpellingLoc().getRawEncoding());
      Record.push_back(Expansion.getExpansionLocStart().getRawEncoding());
      Record.push_back(Expansion.isMacroArgExpansion() ? 0
                             : Expansion.getExpansionLocEnd().getRawEncoding());

      // Compute the token length for this macro expansion.
      unsigned NextOffset = SourceMgr.getNextLocalOffset();
      if (I + 1 != N)
        NextOffset = SourceMgr.getLocalSLocEntry(I + 1).getOffset();
      Record.push_back(NextOffset - SLoc->getOffset() - 1);
      Stream.EmitRecordWithAbbrev(SLocExpansionAbbrv, Record);
    }
  }

  Stream.ExitBlock();

  if (SLocEntryOffsets.empty())
    return;

  // Write the source-location offsets table into the AST block. This
  // table is used for lazily loading source-location information.
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SOURCE_LOCATION_OFFSETS));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // # of slocs
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // total size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // offsets
  unsigned SLocOffsetsAbbrev = Stream.EmitAbbrev(Abbrev);

  Record.clear();
  Record.push_back(SOURCE_LOCATION_OFFSETS);
  Record.push_back(SLocEntryOffsets.size());
  Record.push_back(SourceMgr.getNextLocalOffset() - 1); // skip dummy
  Stream.EmitRecordWithBlob(SLocOffsetsAbbrev, Record, data(SLocEntryOffsets));

  // Write the source location entry preloads array, telling the AST
  // reader which source locations entries it should load eagerly.
  Stream.EmitRecord(SOURCE_LOCATION_PRELOADS, PreloadSLocs);

  // Write the line table. It depends on remapping working, so it must come
  // after the source location offsets.
  if (SourceMgr.hasLineTable()) {
    LineTableInfo &LineTable = SourceMgr.getLineTable();

    Record.clear();
    // Emit the file names
    Record.push_back(LineTable.getNumFilenames());
    for (unsigned I = 0, N = LineTable.getNumFilenames(); I != N; ++I) {
      // Emit the file name
      const char *Filename = LineTable.getFilename(I);
      Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
      unsigned FilenameLen = Filename? strlen(Filename) : 0;
      Record.push_back(FilenameLen);
      if (FilenameLen)
        Record.insert(Record.end(), Filename, Filename + FilenameLen);
    }

    // Emit the line entries
    for (LineTableInfo::iterator L = LineTable.begin(), LEnd = LineTable.end();
         L != LEnd; ++L) {
      // Only emit entries for local files.
      if (L->first.ID < 0)
        continue;

      // Emit the file ID
      Record.push_back(L->first.ID);

      // Emit the line entries
      Record.push_back(L->second.size());
      for (std::vector<LineEntry>::iterator LE = L->second.begin(),
                                         LEEnd = L->second.end();
           LE != LEEnd; ++LE) {
        Record.push_back(LE->FileOffset);
        Record.push_back(LE->LineNo);
        Record.push_back(LE->FilenameID);
        Record.push_back((unsigned)LE->FileKind);
        Record.push_back(LE->IncludeOffset);
      }
    }
    Stream.EmitRecord(SOURCE_MANAGER_LINE_TABLE, Record);
  }
}

//===----------------------------------------------------------------------===//
// Preprocessor Serialization
//===----------------------------------------------------------------------===//

namespace {
class ASTMacroTableTrait {
public:
  typedef IdentID key_type;
  typedef key_type key_type_ref;

  struct Data {
    uint32_t MacroDirectivesOffset;
  };

  typedef Data data_type;
  typedef const data_type &data_type_ref;
  typedef unsigned hash_value_type;
  typedef unsigned offset_type;

  static hash_value_type ComputeHash(IdentID IdID) {
    return llvm::hash_value(IdID);
  }

  std::pair<unsigned,unsigned>
  static EmitKeyDataLength(raw_ostream& Out,
                           key_type_ref Key, data_type_ref Data) {
    unsigned KeyLen = 4; // IdentID.
    unsigned DataLen = 4; // MacroDirectivesOffset.
    return std::make_pair(KeyLen, DataLen);
  }

  static void EmitKey(raw_ostream& Out, key_type_ref Key, unsigned KeyLen) {
    using namespace llvm::support;
    endian::Writer<little>(Out).write<uint32_t>(Key);
  }

  static void EmitData(raw_ostream& Out, key_type_ref Key, data_type_ref Data,
                       unsigned) {
    using namespace llvm::support;
    endian::Writer<little>(Out).write<uint32_t>(Data.MacroDirectivesOffset);
  }
};
} // end anonymous namespace

static int compareMacroDirectives(
    const std::pair<const IdentifierInfo *, MacroDirective *> *X,
    const std::pair<const IdentifierInfo *, MacroDirective *> *Y) {
  return X->first->getName().compare(Y->first->getName());
}

static bool shouldIgnoreMacro(MacroDirective *MD, bool IsModule,
                              const Preprocessor &PP) {
  if (MacroInfo *MI = MD->getMacroInfo())
    if (MI->isBuiltinMacro())
      return true;

  if (IsModule) {
    // Re-export any imported directives.
    // FIXME: Also ensure we re-export imported #undef directives.
    if (auto *DMD = dyn_cast<DefMacroDirective>(MD))
      if (DMD->isImported())
        return false;

    SourceLocation Loc = MD->getLocation();
    if (Loc.isInvalid())
      return true;
    if (PP.getSourceManager().getFileID(Loc) == PP.getPredefinesFileID())
      return true;
  }

  return false;
}

/// \brief Writes the block containing the serialized form of the
/// preprocessor.
///
void ASTWriter::WritePreprocessor(const Preprocessor &PP, bool IsModule) {
  PreprocessingRecord *PPRec = PP.getPreprocessingRecord();
  if (PPRec)
    WritePreprocessorDetail(*PPRec);

  RecordData Record;

  // If the preprocessor __COUNTER__ value has been bumped, remember it.
  if (PP.getCounterValue() != 0) {
    Record.push_back(PP.getCounterValue());
    Stream.EmitRecord(PP_COUNTER_VALUE, Record);
    Record.clear();
  }

  // Enter the preprocessor block.
  Stream.EnterSubblock(PREPROCESSOR_BLOCK_ID, 3);

  // If the AST file contains __DATE__ or __TIME__ emit a warning about this.
  // FIXME: use diagnostics subsystem for localization etc.
  if (PP.SawDateOrTime())
    fprintf(stderr, "warning: precompiled header used __DATE__ or __TIME__.\n");


  // Loop over all the macro directives that are live at the end of the file,
  // emitting each to the PP section.

  // Construct the list of macro directives that need to be serialized.
  SmallVector<std::pair<const IdentifierInfo *, MacroDirective *>, 2>
    MacroDirectives;
  for (Preprocessor::macro_iterator
         I = PP.macro_begin(/*IncludeExternalMacros=*/false),
         E = PP.macro_end(/*IncludeExternalMacros=*/false);
       I != E; ++I) {
    MacroDirectives.push_back(std::make_pair(I->first, I->second));
  }

  // Sort the set of macro definitions that need to be serialized by the
  // name of the macro, to provide a stable ordering.
  llvm::array_pod_sort(MacroDirectives.begin(), MacroDirectives.end(),
                       &compareMacroDirectives);

  llvm::OnDiskChainedHashTableGenerator<ASTMacroTableTrait> Generator;

  // Emit the macro directives as a list and associate the offset with the
  // identifier they belong to.
  for (unsigned I = 0, N = MacroDirectives.size(); I != N; ++I) {
    const IdentifierInfo *Name = MacroDirectives[I].first;
    uint64_t MacroDirectiveOffset = Stream.GetCurrentBitNo();
    MacroDirective *MD = MacroDirectives[I].second;

    // If the macro or identifier need no updates, don't write the macro history
    // for this one.
    // FIXME: Chain the macro history instead of re-writing it.
    if (MD->isFromPCH() &&
        Name->isFromAST() && !Name->hasChangedSinceDeserialization())
      continue;

    // Emit the macro directives in reverse source order.
    for (; MD; MD = MD->getPrevious()) {
      if (shouldIgnoreMacro(MD, IsModule, PP))
        continue;

      AddSourceLocation(MD->getLocation(), Record);
      Record.push_back(MD->getKind());
      if (DefMacroDirective *DefMD = dyn_cast<DefMacroDirective>(MD)) {
        MacroID InfoID = getMacroRef(DefMD->getInfo(), Name);
        Record.push_back(InfoID);
        Record.push_back(DefMD->isImported());
        Record.push_back(DefMD->isAmbiguous());

      } else if (VisibilityMacroDirective *
                   VisMD = dyn_cast<VisibilityMacroDirective>(MD)) {
        Record.push_back(VisMD->isPublic());
      }
    }
    if (Record.empty())
      continue;

    Stream.EmitRecord(PP_MACRO_DIRECTIVE_HISTORY, Record);
    Record.clear();

    IdentMacroDirectivesOffsetMap[Name] = MacroDirectiveOffset;

    IdentID NameID = getIdentifierRef(Name);
    ASTMacroTableTrait::Data data;
    data.MacroDirectivesOffset = MacroDirectiveOffset;
    Generator.insert(NameID, data);
  }

  /// \brief Offsets of each of the macros into the bitstream, indexed by
  /// the local macro ID
  ///
  /// For each identifier that is associated with a macro, this map
  /// provides the offset into the bitstream where that macro is
  /// defined.
  std::vector<uint32_t> MacroOffsets;

  for (unsigned I = 0, N = MacroInfosToEmit.size(); I != N; ++I) {
    const IdentifierInfo *Name = MacroInfosToEmit[I].Name;
    MacroInfo *MI = MacroInfosToEmit[I].MI;
    MacroID ID = MacroInfosToEmit[I].ID;

    if (ID < FirstMacroID) {
      assert(0 && "Loaded MacroInfo entered MacroInfosToEmit ?");
      continue;
    }

    // Record the local offset of this macro.
    unsigned Index = ID - FirstMacroID;
    if (Index == MacroOffsets.size())
      MacroOffsets.push_back(Stream.GetCurrentBitNo());
    else {
      if (Index > MacroOffsets.size())
        MacroOffsets.resize(Index + 1);

      MacroOffsets[Index] = Stream.GetCurrentBitNo();
    }

    AddIdentifierRef(Name, Record);
    Record.push_back(inferSubmoduleIDFromLocation(MI->getDefinitionLoc()));
    AddSourceLocation(MI->getDefinitionLoc(), Record);
    AddSourceLocation(MI->getDefinitionEndLoc(), Record);
    Record.push_back(MI->isUsed());
    Record.push_back(MI->isUsedForHeaderGuard());
    unsigned Code;
    if (MI->isObjectLike()) {
      Code = PP_MACRO_OBJECT_LIKE;
    } else {
      Code = PP_MACRO_FUNCTION_LIKE;

      Record.push_back(MI->isC99Varargs());
      Record.push_back(MI->isGNUVarargs());
      Record.push_back(MI->hasCommaPasting());
      Record.push_back(MI->getNumArgs());
      for (MacroInfo::arg_iterator I = MI->arg_begin(), E = MI->arg_end();
           I != E; ++I)
        AddIdentifierRef(*I, Record);
    }

    // If we have a detailed preprocessing record, record the macro definition
    // ID that corresponds to this macro.
    if (PPRec)
      Record.push_back(MacroDefinitions[PPRec->findMacroDefinition(MI)]);

    Stream.EmitRecord(Code, Record);
    Record.clear();

    // Emit the tokens array.
    for (unsigned TokNo = 0, e = MI->getNumTokens(); TokNo != e; ++TokNo) {
      // Note that we know that the preprocessor does not have any annotation
      // tokens in it because they are created by the parser, and thus can't
      // be in a macro definition.
      const Token &Tok = MI->getReplacementToken(TokNo);
      AddToken(Tok, Record);
      Stream.EmitRecord(PP_TOKEN, Record);
      Record.clear();
    }
    ++NumMacros;
  }

  Stream.ExitBlock();

  // Create the on-disk hash table in a buffer.
  SmallString<4096> MacroTable;
  uint32_t BucketOffset;
  {
    using namespace llvm::support;
    llvm::raw_svector_ostream Out(MacroTable);
    // Make sure that no bucket is at offset 0
    endian::Writer<little>(Out).write<uint32_t>(0);
    BucketOffset = Generator.Emit(Out);
  }

  // Write the macro table
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(MACRO_TABLE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned MacroTableAbbrev = Stream.EmitAbbrev(Abbrev);

  Record.push_back(MACRO_TABLE);
  Record.push_back(BucketOffset);
  Stream.EmitRecordWithBlob(MacroTableAbbrev, Record, MacroTable.str());
  Record.clear();

  // Write the offsets table for macro IDs.
  using namespace llvm;
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(MACRO_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of macros
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // first ID
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));

  unsigned MacroOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(MACRO_OFFSET);
  Record.push_back(MacroOffsets.size());
  Record.push_back(FirstMacroID - NUM_PREDEF_MACRO_IDS);
  Stream.EmitRecordWithBlob(MacroOffsetAbbrev, Record,
                            data(MacroOffsets));
}

void ASTWriter::WritePreprocessorDetail(PreprocessingRecord &PPRec) {
  if (PPRec.local_begin() == PPRec.local_end())
    return;

  SmallVector<PPEntityOffset, 64> PreprocessedEntityOffsets;

  // Enter the preprocessor block.
  Stream.EnterSubblock(PREPROCESSOR_DETAIL_BLOCK_ID, 3);

  // If the preprocessor has a preprocessing record, emit it.
  unsigned NumPreprocessingRecords = 0;
  using namespace llvm;
  
  // Set up the abbreviation for 
  unsigned InclusionAbbrev = 0;
  {
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(PPD_INCLUSION_DIRECTIVE));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // filename length
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // in quotes
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // kind
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // imported module
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    InclusionAbbrev = Stream.EmitAbbrev(Abbrev);
  }
  
  unsigned FirstPreprocessorEntityID 
    = (Chain ? PPRec.getNumLoadedPreprocessedEntities() : 0) 
    + NUM_PREDEF_PP_ENTITY_IDS;
  unsigned NextPreprocessorEntityID = FirstPreprocessorEntityID;
  RecordData Record;
  for (PreprocessingRecord::iterator E = PPRec.local_begin(),
                                  EEnd = PPRec.local_end();
       E != EEnd; 
       (void)++E, ++NumPreprocessingRecords, ++NextPreprocessorEntityID) {
    Record.clear();

    PreprocessedEntityOffsets.push_back(PPEntityOffset((*E)->getSourceRange(),
                                                     Stream.GetCurrentBitNo()));

    if (MacroDefinition *MD = dyn_cast<MacroDefinition>(*E)) {
      // Record this macro definition's ID.
      MacroDefinitions[MD] = NextPreprocessorEntityID;
      
      AddIdentifierRef(MD->getName(), Record);
      Stream.EmitRecord(PPD_MACRO_DEFINITION, Record);
      continue;
    }

    if (MacroExpansion *ME = dyn_cast<MacroExpansion>(*E)) {
      Record.push_back(ME->isBuiltinMacro());
      if (ME->isBuiltinMacro())
        AddIdentifierRef(ME->getName(), Record);
      else
        Record.push_back(MacroDefinitions[ME->getDefinition()]);
      Stream.EmitRecord(PPD_MACRO_EXPANSION, Record);
      continue;
    }

    if (InclusionDirective *ID = dyn_cast<InclusionDirective>(*E)) {
      Record.push_back(PPD_INCLUSION_DIRECTIVE);
      Record.push_back(ID->getFileName().size());
      Record.push_back(ID->wasInQuotes());
      Record.push_back(static_cast<unsigned>(ID->getKind()));
      Record.push_back(ID->importedModule());
      SmallString<64> Buffer;
      Buffer += ID->getFileName();
      // Check that the FileEntry is not null because it was not resolved and
      // we create a PCH even with compiler errors.
      if (ID->getFile())
        Buffer += ID->getFile()->getName();
      Stream.EmitRecordWithBlob(InclusionAbbrev, Record, Buffer);
      continue;
    }
    
    llvm_unreachable("Unhandled PreprocessedEntity in ASTWriter");
  }
  Stream.ExitBlock();

  // Write the offsets table for the preprocessing record.
  if (NumPreprocessingRecords > 0) {
    assert(PreprocessedEntityOffsets.size() == NumPreprocessingRecords);

    // Write the offsets table for identifier IDs.
    using namespace llvm;
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(PPD_ENTITIES_OFFSETS));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // first pp entity
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned PPEOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

    Record.clear();
    Record.push_back(PPD_ENTITIES_OFFSETS);
    Record.push_back(FirstPreprocessorEntityID - NUM_PREDEF_PP_ENTITY_IDS);
    Stream.EmitRecordWithBlob(PPEOffsetAbbrev, Record,
                              data(PreprocessedEntityOffsets));
  }
}

unsigned ASTWriter::getSubmoduleID(Module *Mod) {
  llvm::DenseMap<Module *, unsigned>::iterator Known = SubmoduleIDs.find(Mod);
  if (Known != SubmoduleIDs.end())
    return Known->second;
  
  return SubmoduleIDs[Mod] = NextSubmoduleID++;
}

unsigned ASTWriter::getExistingSubmoduleID(Module *Mod) const {
  if (!Mod)
    return 0;

  llvm::DenseMap<Module *, unsigned>::const_iterator
    Known = SubmoduleIDs.find(Mod);
  if (Known != SubmoduleIDs.end())
    return Known->second;

  return 0;
}

/// \brief Compute the number of modules within the given tree (including the
/// given module).
static unsigned getNumberOfModules(Module *Mod) {
  unsigned ChildModules = 0;
  for (Module::submodule_iterator Sub = Mod->submodule_begin(),
                               SubEnd = Mod->submodule_end();
       Sub != SubEnd; ++Sub)
    ChildModules += getNumberOfModules(*Sub);
  
  return ChildModules + 1;
}

void ASTWriter::WriteSubmodules(Module *WritingModule) {
  // Determine the dependencies of our module and each of it's submodules.
  // FIXME: This feels like it belongs somewhere else, but there are no
  // other consumers of this information.
  SourceManager &SrcMgr = PP->getSourceManager();
  ModuleMap &ModMap = PP->getHeaderSearchInfo().getModuleMap();
  for (const auto *I : Context->local_imports()) {
    if (Module *ImportedFrom
          = ModMap.inferModuleFromLocation(FullSourceLoc(I->getLocation(), 
                                                         SrcMgr))) {
      ImportedFrom->Imports.push_back(I->getImportedModule());
    }
  }
  
  // Enter the submodule description block.
  Stream.EnterSubblock(SUBMODULE_BLOCK_ID, NUM_ALLOWED_ABBREVS_SIZE);
  
  // Write the abbreviations needed for the submodules block.
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_DEFINITION));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // ID
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Parent
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsFramework
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsExplicit
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsSystem
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsExternC
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // InferSubmodules...
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // InferExplicit...
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // InferExportWild...
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // ConfigMacrosExh...
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned DefinitionAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_UMBRELLA_HEADER));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned UmbrellaAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_HEADER));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned HeaderAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_TOPHEADER));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned TopHeaderAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_UMBRELLA_DIR));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned UmbrellaDirAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_REQUIRES));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // State
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));     // Feature
  unsigned RequiresAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_EXCLUDED_HEADER));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned ExcludedHeaderAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_PRIVATE_HEADER));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Name
  unsigned PrivateHeaderAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_LINK_LIBRARY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // IsFramework
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));     // Name
  unsigned LinkLibraryAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_CONFIG_MACRO));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));    // Macro name
  unsigned ConfigMacroAbbrev = Stream.EmitAbbrev(Abbrev);

  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(SUBMODULE_CONFLICT));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));  // Other module
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));    // Message
  unsigned ConflictAbbrev = Stream.EmitAbbrev(Abbrev);

  // Write the submodule metadata block.
  RecordData Record;
  Record.push_back(getNumberOfModules(WritingModule));
  Record.push_back(FirstSubmoduleID - NUM_PREDEF_SUBMODULE_IDS);
  Stream.EmitRecord(SUBMODULE_METADATA, Record);
  
  // Write all of the submodules.
  std::queue<Module *> Q;
  Q.push(WritingModule);
  while (!Q.empty()) {
    Module *Mod = Q.front();
    Q.pop();
    unsigned ID = getSubmoduleID(Mod);
    
    // Emit the definition of the block.
    Record.clear();
    Record.push_back(SUBMODULE_DEFINITION);
    Record.push_back(ID);
    if (Mod->Parent) {
      assert(SubmoduleIDs[Mod->Parent] && "Submodule parent not written?");
      Record.push_back(SubmoduleIDs[Mod->Parent]);
    } else {
      Record.push_back(0);
    }
    Record.push_back(Mod->IsFramework);
    Record.push_back(Mod->IsExplicit);
    Record.push_back(Mod->IsSystem);
    Record.push_back(Mod->IsExternC);
    Record.push_back(Mod->InferSubmodules);
    Record.push_back(Mod->InferExplicitSubmodules);
    Record.push_back(Mod->InferExportWildcard);
    Record.push_back(Mod->ConfigMacrosExhaustive);
    Stream.EmitRecordWithBlob(DefinitionAbbrev, Record, Mod->Name);
    
    // Emit the requirements.
    for (unsigned I = 0, N = Mod->Requirements.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_REQUIRES);
      Record.push_back(Mod->Requirements[I].second);
      Stream.EmitRecordWithBlob(RequiresAbbrev, Record,
                                Mod->Requirements[I].first);
    }

    // Emit the umbrella header, if there is one.
    if (const FileEntry *UmbrellaHeader = Mod->getUmbrellaHeader()) {
      Record.clear();
      Record.push_back(SUBMODULE_UMBRELLA_HEADER);
      Stream.EmitRecordWithBlob(UmbrellaAbbrev, Record, 
                                UmbrellaHeader->getName());
    } else if (const DirectoryEntry *UmbrellaDir = Mod->getUmbrellaDir()) {
      Record.clear();
      Record.push_back(SUBMODULE_UMBRELLA_DIR);
      Stream.EmitRecordWithBlob(UmbrellaDirAbbrev, Record, 
                                UmbrellaDir->getName());      
    }
    
    // Emit the headers.
    for (unsigned I = 0, N = Mod->NormalHeaders.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_HEADER);
      Stream.EmitRecordWithBlob(HeaderAbbrev, Record, 
                                Mod->NormalHeaders[I]->getName());
    }
    // Emit the excluded headers.
    for (unsigned I = 0, N = Mod->ExcludedHeaders.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_EXCLUDED_HEADER);
      Stream.EmitRecordWithBlob(ExcludedHeaderAbbrev, Record, 
                                Mod->ExcludedHeaders[I]->getName());
    }
    // Emit the private headers.
    for (unsigned I = 0, N = Mod->PrivateHeaders.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_PRIVATE_HEADER);
      Stream.EmitRecordWithBlob(PrivateHeaderAbbrev, Record, 
                                Mod->PrivateHeaders[I]->getName());
    }
    ArrayRef<const FileEntry *>
      TopHeaders = Mod->getTopHeaders(PP->getFileManager());
    for (unsigned I = 0, N = TopHeaders.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_TOPHEADER);
      Stream.EmitRecordWithBlob(TopHeaderAbbrev, Record,
                                TopHeaders[I]->getName());
    }

    // Emit the imports. 
    if (!Mod->Imports.empty()) {
      Record.clear();
      for (unsigned I = 0, N = Mod->Imports.size(); I != N; ++I) {
        unsigned ImportedID = getSubmoduleID(Mod->Imports[I]);
        assert(ImportedID && "Unknown submodule!");                                           
        Record.push_back(ImportedID);
      }
      Stream.EmitRecord(SUBMODULE_IMPORTS, Record);
    }

    // Emit the exports. 
    if (!Mod->Exports.empty()) {
      Record.clear();
      for (unsigned I = 0, N = Mod->Exports.size(); I != N; ++I) {
        if (Module *Exported = Mod->Exports[I].getPointer()) {
          unsigned ExportedID = SubmoduleIDs[Exported];
          assert(ExportedID > 0 && "Unknown submodule ID?");
          Record.push_back(ExportedID);
        } else {
          Record.push_back(0);
        }
        
        Record.push_back(Mod->Exports[I].getInt());
      }
      Stream.EmitRecord(SUBMODULE_EXPORTS, Record);
    }

    //FIXME: How do we emit the 'use'd modules?  They may not be submodules.
    // Might be unnecessary as use declarations are only used to build the
    // module itself.

    // Emit the link libraries.
    for (unsigned I = 0, N = Mod->LinkLibraries.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_LINK_LIBRARY);
      Record.push_back(Mod->LinkLibraries[I].IsFramework);
      Stream.EmitRecordWithBlob(LinkLibraryAbbrev, Record,
                                Mod->LinkLibraries[I].Library);
    }

    // Emit the conflicts.
    for (unsigned I = 0, N = Mod->Conflicts.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_CONFLICT);
      unsigned OtherID = getSubmoduleID(Mod->Conflicts[I].Other);
      assert(OtherID && "Unknown submodule!");
      Record.push_back(OtherID);
      Stream.EmitRecordWithBlob(ConflictAbbrev, Record,
                                Mod->Conflicts[I].Message);
    }

    // Emit the configuration macros.
    for (unsigned I = 0, N =  Mod->ConfigMacros.size(); I != N; ++I) {
      Record.clear();
      Record.push_back(SUBMODULE_CONFIG_MACRO);
      Stream.EmitRecordWithBlob(ConfigMacroAbbrev, Record,
                                Mod->ConfigMacros[I]);
    }

    // Queue up the submodules of this module.
    for (Module::submodule_iterator Sub = Mod->submodule_begin(),
                                 SubEnd = Mod->submodule_end();
         Sub != SubEnd; ++Sub)
      Q.push(*Sub);
  }
  
  Stream.ExitBlock();
  
  assert((NextSubmoduleID - FirstSubmoduleID
            == getNumberOfModules(WritingModule)) && "Wrong # of submodules");
}

serialization::SubmoduleID 
ASTWriter::inferSubmoduleIDFromLocation(SourceLocation Loc) {
  if (Loc.isInvalid() || !WritingModule)
    return 0; // No submodule
    
  // Find the module that owns this location.
  ModuleMap &ModMap = PP->getHeaderSearchInfo().getModuleMap();
  Module *OwningMod 
    = ModMap.inferModuleFromLocation(FullSourceLoc(Loc,PP->getSourceManager()));
  if (!OwningMod)
    return 0;
  
  // Check whether this submodule is part of our own module.
  if (WritingModule != OwningMod && !OwningMod->isSubModuleOf(WritingModule))
    return 0;
  
  return getSubmoduleID(OwningMod);
}

void ASTWriter::WritePragmaDiagnosticMappings(const DiagnosticsEngine &Diag,
                                              bool isModule) {
  // Make sure set diagnostic pragmas don't affect the translation unit that
  // imports the module.
  // FIXME: Make diagnostic pragma sections work properly with modules.
  if (isModule)
    return;

  llvm::SmallDenseMap<const DiagnosticsEngine::DiagState *, unsigned, 64>
      DiagStateIDMap;
  unsigned CurrID = 0;
  DiagStateIDMap[&Diag.DiagStates.front()] = ++CurrID; // the command-line one.
  RecordData Record;
  for (DiagnosticsEngine::DiagStatePointsTy::const_iterator
         I = Diag.DiagStatePoints.begin(), E = Diag.DiagStatePoints.end();
         I != E; ++I) {
    const DiagnosticsEngine::DiagStatePoint &point = *I;
    if (point.Loc.isInvalid())
      continue;

    Record.push_back(point.Loc.getRawEncoding());
    unsigned &DiagStateID = DiagStateIDMap[point.State];
    Record.push_back(DiagStateID);
    
    if (DiagStateID == 0) {
      DiagStateID = ++CurrID;
      for (DiagnosticsEngine::DiagState::const_iterator
             I = point.State->begin(), E = point.State->end(); I != E; ++I) {
        if (I->second.isPragma()) {
          Record.push_back(I->first);
          Record.push_back((unsigned)I->second.getSeverity());
        }
      }
      Record.push_back(-1); // mark the end of the diag/map pairs for this
                            // location.
    }
  }

  if (!Record.empty())
    Stream.EmitRecord(DIAG_PRAGMA_MAPPINGS, Record);
}

void ASTWriter::WriteCXXBaseSpecifiersOffsets() {
  if (CXXBaseSpecifiersOffsets.empty())
    return;

  RecordData Record;

  // Create a blob abbreviation for the C++ base specifiers offsets.
  using namespace llvm;
    
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(CXX_BASE_SPECIFIER_OFFSETS));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned BaseSpecifierOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  
  // Write the base specifier offsets table.
  Record.clear();
  Record.push_back(CXX_BASE_SPECIFIER_OFFSETS);
  Record.push_back(CXXBaseSpecifiersOffsets.size());
  Stream.EmitRecordWithBlob(BaseSpecifierOffsetAbbrev, Record,
                            data(CXXBaseSpecifiersOffsets));
}

//===----------------------------------------------------------------------===//
// Type Serialization
//===----------------------------------------------------------------------===//

/// \brief Write the representation of a type to the AST stream.
void ASTWriter::WriteType(QualType T) {
  TypeIdx &Idx = TypeIdxs[T];
  if (Idx.getIndex() == 0) // we haven't seen this type before.
    Idx = TypeIdx(NextTypeID++);

  assert(Idx.getIndex() >= FirstTypeID && "Re-writing a type from a prior AST");

  // Record the offset for this type.
  unsigned Index = Idx.getIndex() - FirstTypeID;
  if (TypeOffsets.size() == Index)
    TypeOffsets.push_back(Stream.GetCurrentBitNo());
  else if (TypeOffsets.size() < Index) {
    TypeOffsets.resize(Index + 1);
    TypeOffsets[Index] = Stream.GetCurrentBitNo();
  }

  RecordData Record;

  // Emit the type's representation.
  ASTTypeWriter W(*this, Record);

  if (T.hasLocalNonFastQualifiers()) {
    Qualifiers Qs = T.getLocalQualifiers();
    AddTypeRef(T.getLocalUnqualifiedType(), Record);
    Record.push_back(Qs.getAsOpaqueValue());
    W.Code = TYPE_EXT_QUAL;
  } else {
    switch (T->getTypeClass()) {
      // For all of the concrete, non-dependent types, call the
      // appropriate visitor function.
#define TYPE(Class, Base) \
    case Type::Class: W.Visit##Class##Type(cast<Class##Type>(T)); break;
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
    }
  }

  // Emit the serialized record.
  Stream.EmitRecord(W.Code, Record);

  // Flush any expressions that were written as part of this type.
  FlushStmts();
}

//===----------------------------------------------------------------------===//
// Declaration Serialization
//===----------------------------------------------------------------------===//

/// \brief Write the block containing all of the declaration IDs
/// lexically declared within the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_LEXICAL block within the
/// bistream, or 0 if no block was written.
uint64_t ASTWriter::WriteDeclContextLexicalBlock(ASTContext &Context,
                                                 DeclContext *DC) {
  if (DC->decls_empty())
    return 0;

  uint64_t Offset = Stream.GetCurrentBitNo();
  RecordData Record;
  Record.push_back(DECL_CONTEXT_LEXICAL);
  SmallVector<KindDeclIDPair, 64> Decls;
  for (const auto *D : DC->decls())
    Decls.push_back(std::make_pair(D->getKind(), GetDeclRef(D)));

  ++NumLexicalDeclContexts;
  Stream.EmitRecordWithBlob(DeclContextLexicalAbbrev, Record, data(Decls));
  return Offset;
}

void ASTWriter::WriteTypeDeclOffsets() {
  using namespace llvm;
  RecordData Record;

  // Write the type offsets array
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(TYPE_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of types
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // base type index
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // types block
  unsigned TypeOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(TYPE_OFFSET);
  Record.push_back(TypeOffsets.size());
  Record.push_back(FirstTypeID - NUM_PREDEF_TYPE_IDS);
  Stream.EmitRecordWithBlob(TypeOffsetAbbrev, Record, data(TypeOffsets));

  // Write the declaration offsets array
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(DECL_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of declarations
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // base decl ID
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // declarations block
  unsigned DeclOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(DECL_OFFSET);
  Record.push_back(DeclOffsets.size());
  Record.push_back(FirstDeclID - NUM_PREDEF_DECL_IDS);
  Stream.EmitRecordWithBlob(DeclOffsetAbbrev, Record, data(DeclOffsets));
}

void ASTWriter::WriteFileDeclIDsMap() {
  using namespace llvm;
  RecordData Record;

  // Join the vectors of DeclIDs from all files.
  SmallVector<DeclID, 256> FileSortedIDs;
  for (FileDeclIDsTy::iterator
         FI = FileDeclIDs.begin(), FE = FileDeclIDs.end(); FI != FE; ++FI) {
    DeclIDInFileInfo &Info = *FI->second;
    Info.FirstDeclIndex = FileSortedIDs.size();
    for (LocDeclIDsTy::iterator
           DI = Info.DeclIDs.begin(), DE = Info.DeclIDs.end(); DI != DE; ++DI)
      FileSortedIDs.push_back(DI->second);
  }

  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(FILE_SORTED_DECLS));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned AbbrevCode = Stream.EmitAbbrev(Abbrev);
  Record.push_back(FILE_SORTED_DECLS);
  Record.push_back(FileSortedIDs.size());
  Stream.EmitRecordWithBlob(AbbrevCode, Record, data(FileSortedIDs));
}

void ASTWriter::WriteComments() {
  Stream.EnterSubblock(COMMENTS_BLOCK_ID, 3);
  ArrayRef<RawComment *> RawComments = Context->Comments.getComments();
  RecordData Record;
  for (ArrayRef<RawComment *>::iterator I = RawComments.begin(),
                                        E = RawComments.end();
       I != E; ++I) {
    Record.clear();
    AddSourceRange((*I)->getSourceRange(), Record);
    Record.push_back((*I)->getKind());
    Record.push_back((*I)->isTrailingComment());
    Record.push_back((*I)->isAlmostTrailingComment());
    Stream.EmitRecord(COMMENTS_RAW_COMMENT, Record);
  }
  Stream.ExitBlock();
}

//===----------------------------------------------------------------------===//
// Global Method Pool and Selector Serialization
//===----------------------------------------------------------------------===//

namespace {
// Trait used for the on-disk hash table used in the method pool.
class ASTMethodPoolTrait {
  ASTWriter &Writer;

public:
  typedef Selector key_type;
  typedef key_type key_type_ref;

  struct data_type {
    SelectorID ID;
    ObjCMethodList Instance, Factory;
  };
  typedef const data_type& data_type_ref;

  typedef unsigned hash_value_type;
  typedef unsigned offset_type;

  explicit ASTMethodPoolTrait(ASTWriter &Writer) : Writer(Writer) { }

  static hash_value_type ComputeHash(Selector Sel) {
    return serialization::ComputeHash(Sel);
  }

  std::pair<unsigned,unsigned>
    EmitKeyDataLength(raw_ostream& Out, Selector Sel,
                      data_type_ref Methods) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    unsigned KeyLen = 2 + (Sel.getNumArgs()? Sel.getNumArgs() * 4 : 4);
    LE.write<uint16_t>(KeyLen);
    unsigned DataLen = 4 + 2 + 2; // 2 bytes for each of the method counts
    for (const ObjCMethodList *Method = &Methods.Instance; Method;
         Method = Method->getNext())
      if (Method->Method)
        DataLen += 4;
    for (const ObjCMethodList *Method = &Methods.Factory; Method;
         Method = Method->getNext())
      if (Method->Method)
        DataLen += 4;
    LE.write<uint16_t>(DataLen);
    return std::make_pair(KeyLen, DataLen);
  }

  void EmitKey(raw_ostream& Out, Selector Sel, unsigned) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    uint64_t Start = Out.tell();
    assert((Start >> 32) == 0 && "Selector key offset too large");
    Writer.SetSelectorOffset(Sel, Start);
    unsigned N = Sel.getNumArgs();
    LE.write<uint16_t>(N);
    if (N == 0)
      N = 1;
    for (unsigned I = 0; I != N; ++I)
      LE.write<uint32_t>(
          Writer.getIdentifierRef(Sel.getIdentifierInfoForSlot(I)));
  }

  void EmitData(raw_ostream& Out, key_type_ref,
                data_type_ref Methods, unsigned DataLen) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    uint64_t Start = Out.tell(); (void)Start;
    LE.write<uint32_t>(Methods.ID);
    unsigned NumInstanceMethods = 0;
    for (const ObjCMethodList *Method = &Methods.Instance; Method;
         Method = Method->getNext())
      if (Method->Method)
        ++NumInstanceMethods;

    unsigned NumFactoryMethods = 0;
    for (const ObjCMethodList *Method = &Methods.Factory; Method;
         Method = Method->getNext())
      if (Method->Method)
        ++NumFactoryMethods;

    unsigned InstanceBits = Methods.Instance.getBits();
    assert(InstanceBits < 4);
    unsigned NumInstanceMethodsAndBits =
        (NumInstanceMethods << 2) | InstanceBits;
    unsigned FactoryBits = Methods.Factory.getBits();
    assert(FactoryBits < 4);
    unsigned NumFactoryMethodsAndBits = (NumFactoryMethods << 2) | FactoryBits;
    LE.write<uint16_t>(NumInstanceMethodsAndBits);
    LE.write<uint16_t>(NumFactoryMethodsAndBits);
    for (const ObjCMethodList *Method = &Methods.Instance; Method;
         Method = Method->getNext())
      if (Method->Method)
        LE.write<uint32_t>(Writer.getDeclID(Method->Method));
    for (const ObjCMethodList *Method = &Methods.Factory; Method;
         Method = Method->getNext())
      if (Method->Method)
        LE.write<uint32_t>(Writer.getDeclID(Method->Method));

    assert(Out.tell() - Start == DataLen && "Data length is wrong");
  }
};
} // end anonymous namespace

/// \brief Write ObjC data: selectors and the method pool.
///
/// The method pool contains both instance and factory methods, stored
/// in an on-disk hash table indexed by the selector. The hash table also
/// contains an empty entry for every other selector known to Sema.
void ASTWriter::WriteSelectors(Sema &SemaRef) {
  using namespace llvm;

  // Do we have to do anything at all?
  if (SemaRef.MethodPool.empty() && SelectorIDs.empty())
    return;
  unsigned NumTableEntries = 0;
  // Create and write out the blob that contains selectors and the method pool.
  {
    llvm::OnDiskChainedHashTableGenerator<ASTMethodPoolTrait> Generator;
    ASTMethodPoolTrait Trait(*this);

    // Create the on-disk hash table representation. We walk through every
    // selector we've seen and look it up in the method pool.
    SelectorOffsets.resize(NextSelectorID - FirstSelectorID);
    for (llvm::DenseMap<Selector, SelectorID>::iterator
             I = SelectorIDs.begin(), E = SelectorIDs.end();
         I != E; ++I) {
      Selector S = I->first;
      Sema::GlobalMethodPool::iterator F = SemaRef.MethodPool.find(S);
      ASTMethodPoolTrait::data_type Data = {
        I->second,
        ObjCMethodList(),
        ObjCMethodList()
      };
      if (F != SemaRef.MethodPool.end()) {
        Data.Instance = F->second.first;
        Data.Factory = F->second.second;
      }
      // Only write this selector if it's not in an existing AST or something
      // changed.
      if (Chain && I->second < FirstSelectorID) {
        // Selector already exists. Did it change?
        bool changed = false;
        for (ObjCMethodList *M = &Data.Instance; !changed && M && M->Method;
             M = M->getNext()) {
          if (!M->Method->isFromASTFile())
            changed = true;
        }
        for (ObjCMethodList *M = &Data.Factory; !changed && M && M->Method;
             M = M->getNext()) {
          if (!M->Method->isFromASTFile())
            changed = true;
        }
        if (!changed)
          continue;
      } else if (Data.Instance.Method || Data.Factory.Method) {
        // A new method pool entry.
        ++NumTableEntries;
      }
      Generator.insert(S, Data, Trait);
    }

    // Create the on-disk hash table in a buffer.
    SmallString<4096> MethodPool;
    uint32_t BucketOffset;
    {
      using namespace llvm::support;
      ASTMethodPoolTrait Trait(*this);
      llvm::raw_svector_ostream Out(MethodPool);
      // Make sure that no bucket is at offset 0
      endian::Writer<little>(Out).write<uint32_t>(0);
      BucketOffset = Generator.Emit(Out, Trait);
    }

    // Create a blob abbreviation
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(METHOD_POOL));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned MethodPoolAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the method pool
    RecordData Record;
    Record.push_back(METHOD_POOL);
    Record.push_back(BucketOffset);
    Record.push_back(NumTableEntries);
    Stream.EmitRecordWithBlob(MethodPoolAbbrev, Record, MethodPool.str());

    // Create a blob abbreviation for the selector table offsets.
    Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(SELECTOR_OFFSETS));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // size
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // first ID
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned SelectorOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the selector offsets table.
    Record.clear();
    Record.push_back(SELECTOR_OFFSETS);
    Record.push_back(SelectorOffsets.size());
    Record.push_back(FirstSelectorID - NUM_PREDEF_SELECTOR_IDS);
    Stream.EmitRecordWithBlob(SelectorOffsetAbbrev, Record,
                              data(SelectorOffsets));
  }
}

/// \brief Write the selectors referenced in @selector expression into AST file.
void ASTWriter::WriteReferencedSelectorsPool(Sema &SemaRef) {
  using namespace llvm;
  if (SemaRef.ReferencedSelectors.empty())
    return;

  RecordData Record;

  // Note: this writes out all references even for a dependent AST. But it is
  // very tricky to fix, and given that @selector shouldn't really appear in
  // headers, probably not worth it. It's not a correctness issue.
  for (DenseMap<Selector, SourceLocation>::iterator S =
       SemaRef.ReferencedSelectors.begin(),
       E = SemaRef.ReferencedSelectors.end(); S != E; ++S) {
    Selector Sel = (*S).first;
    SourceLocation Loc = (*S).second;
    AddSelectorRef(Sel, Record);
    AddSourceLocation(Loc, Record);
  }
  Stream.EmitRecord(REFERENCED_SELECTOR_POOL, Record);
}

//===----------------------------------------------------------------------===//
// Identifier Table Serialization
//===----------------------------------------------------------------------===//

namespace {
class ASTIdentifierTableTrait {
  ASTWriter &Writer;
  Preprocessor &PP;
  IdentifierResolver &IdResolver;
  bool IsModule;
  
  /// \brief Determines whether this is an "interesting" identifier
  /// that needs a full IdentifierInfo structure written into the hash
  /// table.
  bool isInterestingIdentifier(IdentifierInfo *II, MacroDirective *&Macro) {
    if (II->isPoisoned() ||
        II->isExtensionToken() ||
        II->getObjCOrBuiltinID() ||
        II->hasRevertedTokenIDToIdentifier() ||
        II->getFETokenInfo<void>())
      return true;

    return hadMacroDefinition(II, Macro);
  }

  bool hadMacroDefinition(IdentifierInfo *II, MacroDirective *&Macro) {
    if (!II->hadMacroDefinition())
      return false;

    if (Macro || (Macro = PP.getMacroDirectiveHistory(II))) {
      if (!IsModule)
        return !shouldIgnoreMacro(Macro, IsModule, PP);
      SubmoduleID ModID;
      if (getFirstPublicSubmoduleMacro(Macro, ModID))
        return true;
    }

    return false;
  }

  typedef llvm::SmallVectorImpl<SubmoduleID> OverriddenList;

  MacroDirective *
  getFirstPublicSubmoduleMacro(MacroDirective *MD, SubmoduleID &ModID) {
    ModID = 0;
    llvm::SmallVector<SubmoduleID, 1> Overridden;
    if (MacroDirective *NextMD = getPublicSubmoduleMacro(MD, ModID, Overridden))
      if (!shouldIgnoreMacro(NextMD, IsModule, PP))
        return NextMD;
    return nullptr;
  }

  MacroDirective *
  getNextPublicSubmoduleMacro(MacroDirective *MD, SubmoduleID &ModID,
                              OverriddenList &Overridden) {
    if (MacroDirective *NextMD =
            getPublicSubmoduleMacro(MD->getPrevious(), ModID, Overridden))
      if (!shouldIgnoreMacro(NextMD, IsModule, PP))
        return NextMD;
    return nullptr;
  }

  /// \brief Traverses the macro directives history and returns the latest
  /// public macro definition or undefinition that is not in ModID.
  /// A macro that is defined in submodule A and undefined in submodule B
  /// will still be considered as defined/exported from submodule A.
  /// ModID is updated to the module containing the returned directive.
  ///
  /// FIXME: This process breaks down if a module defines a macro, imports
  ///        another submodule that changes the macro, then changes the
  ///        macro again itself.
  MacroDirective *getPublicSubmoduleMacro(MacroDirective *MD,
                                          SubmoduleID &ModID,
                                          OverriddenList &Overridden) {
    Overridden.clear();
    if (!MD)
      return nullptr;

    SubmoduleID OrigModID = ModID;
    Optional<bool> IsPublic;
    for (; MD; MD = MD->getPrevious()) {
      SubmoduleID ThisModID = getSubmoduleID(MD);
      if (ThisModID == 0) {
        IsPublic = Optional<bool>();

        // If we have no directive location, this macro was installed when
        // finalizing the ASTReader.
        if (DefMacroDirective *DefMD = dyn_cast<DefMacroDirective>(MD))
          if (DefMD->getInfo()->getOwningModuleID())
            return MD;
        // Skip imports that only produce #undefs for now.
        // FIXME: We should still re-export them!

        continue;
      }
      if (ThisModID != ModID) {
        ModID = ThisModID;
        IsPublic = Optional<bool>();
      }

      // If this is a definition from a submodule import, that submodule's
      // definition is overridden by the definition or undefinition that we
      // started with.
      // FIXME: This should only apply to macros defined in OrigModID.
      // We can't do that currently, because a #include of a different submodule
      // of the same module just leaks through macros instead of providing new
      // DefMacroDirectives for them.
      if (DefMacroDirective *DefMD = dyn_cast<DefMacroDirective>(MD)) {
        // Figure out which submodule the macro was originally defined within.
        SubmoduleID SourceID = DefMD->getInfo()->getOwningModuleID();
        if (!SourceID) {
          SourceLocation DefLoc = DefMD->getInfo()->getDefinitionLoc();
          if (DefLoc == MD->getLocation())
            SourceID = ThisModID;
          else
            SourceID = Writer.inferSubmoduleIDFromLocation(DefLoc);
        }
        if (OrigModID && SourceID != OrigModID)
          Overridden.push_back(SourceID);
      }

      // We are looking for a definition in a different submodule than the one
      // that we started with. If a submodule has re-definitions of the same
      // macro, only the last definition will be used as the "exported" one.
      if (ModID == OrigModID)
        continue;

      // The latest visibility directive for a name in a submodule affects all
      // the directives that come before it.
      if (VisibilityMacroDirective *VisMD =
              dyn_cast<VisibilityMacroDirective>(MD)) {
        if (!IsPublic.hasValue())
          IsPublic = VisMD->isPublic();
      } else if (!IsPublic.hasValue() || IsPublic.getValue()) {
        // FIXME: If we find an imported macro, we should include its list of
        // overrides in our export.
        return MD;
      }
    }

    return nullptr;
  }

  SubmoduleID getSubmoduleID(MacroDirective *MD) {
    return Writer.inferSubmoduleIDFromLocation(MD->getLocation());
  }

public:
  typedef IdentifierInfo* key_type;
  typedef key_type  key_type_ref;

  typedef IdentID data_type;
  typedef data_type data_type_ref;

  typedef unsigned hash_value_type;
  typedef unsigned offset_type;

  ASTIdentifierTableTrait(ASTWriter &Writer, Preprocessor &PP, 
                          IdentifierResolver &IdResolver, bool IsModule)
    : Writer(Writer), PP(PP), IdResolver(IdResolver), IsModule(IsModule) { }

  static hash_value_type ComputeHash(const IdentifierInfo* II) {
    return llvm::HashString(II->getName());
  }

  std::pair<unsigned,unsigned>
  EmitKeyDataLength(raw_ostream& Out, IdentifierInfo* II, IdentID ID) {
    unsigned KeyLen = II->getLength() + 1;
    unsigned DataLen = 4; // 4 bytes for the persistent ID << 1
    MacroDirective *Macro = nullptr;
    if (isInterestingIdentifier(II, Macro)) {
      DataLen += 2; // 2 bytes for builtin ID
      DataLen += 2; // 2 bytes for flags
      if (hadMacroDefinition(II, Macro)) {
        DataLen += 4; // MacroDirectives offset.
        if (IsModule) {
          SubmoduleID ModID;
          llvm::SmallVector<SubmoduleID, 4> Overridden;
          for (MacroDirective *
                 MD = getFirstPublicSubmoduleMacro(Macro, ModID);
                 MD; MD = getNextPublicSubmoduleMacro(MD, ModID, Overridden)) {
            // Previous macro's overrides.
            if (!Overridden.empty())
              DataLen += 4 * (1 + Overridden.size());
            DataLen += 4; // MacroInfo ID or ModuleID.
          }
          // Previous macro's overrides.
          if (!Overridden.empty())
            DataLen += 4 * (1 + Overridden.size());
          DataLen += 4;
        }
      }

      for (IdentifierResolver::iterator D = IdResolver.begin(II),
                                     DEnd = IdResolver.end();
           D != DEnd; ++D)
        DataLen += sizeof(DeclID);
    }
    using namespace llvm::support;
    endian::Writer<little> LE(Out);

    LE.write<uint16_t>(DataLen);
    // We emit the key length after the data length so that every
    // string is preceded by a 16-bit length. This matches the PTH
    // format for storing identifiers.
    LE.write<uint16_t>(KeyLen);
    return std::make_pair(KeyLen, DataLen);
  }

  void EmitKey(raw_ostream& Out, const IdentifierInfo* II,
               unsigned KeyLen) {
    // Record the location of the key data.  This is used when generating
    // the mapping from persistent IDs to strings.
    Writer.SetIdentifierOffset(II, Out.tell());
    Out.write(II->getNameStart(), KeyLen);
  }

  static void emitMacroOverrides(raw_ostream &Out,
                                 ArrayRef<SubmoduleID> Overridden) {
    if (!Overridden.empty()) {
      using namespace llvm::support;
      endian::Writer<little> LE(Out);
      LE.write<uint32_t>(Overridden.size() | 0x80000000U);
      for (unsigned I = 0, N = Overridden.size(); I != N; ++I)
        LE.write<uint32_t>(Overridden[I]);
    }
  }

  void EmitData(raw_ostream& Out, IdentifierInfo* II,
                IdentID ID, unsigned) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    MacroDirective *Macro = nullptr;
    if (!isInterestingIdentifier(II, Macro)) {
      LE.write<uint32_t>(ID << 1);
      return;
    }

    LE.write<uint32_t>((ID << 1) | 0x01);
    uint32_t Bits = (uint32_t)II->getObjCOrBuiltinID();
    assert((Bits & 0xffff) == Bits && "ObjCOrBuiltinID too big for ASTReader.");
    LE.write<uint16_t>(Bits);
    Bits = 0;
    bool HadMacroDefinition = hadMacroDefinition(II, Macro);
    Bits = (Bits << 1) | unsigned(HadMacroDefinition);
    Bits = (Bits << 1) | unsigned(IsModule);
    Bits = (Bits << 1) | unsigned(II->isExtensionToken());
    Bits = (Bits << 1) | unsigned(II->isPoisoned());
    Bits = (Bits << 1) | unsigned(II->hasRevertedTokenIDToIdentifier());
    Bits = (Bits << 1) | unsigned(II->isCPlusPlusOperatorKeyword());
    LE.write<uint16_t>(Bits);

    if (HadMacroDefinition) {
      LE.write<uint32_t>(Writer.getMacroDirectivesOffset(II));
      if (IsModule) {
        // Write the IDs of macros coming from different submodules.
        SubmoduleID ModID;
        llvm::SmallVector<SubmoduleID, 4> Overridden;
        for (MacroDirective *
               MD = getFirstPublicSubmoduleMacro(Macro, ModID);
               MD; MD = getNextPublicSubmoduleMacro(MD, ModID, Overridden)) {
          MacroID InfoID = 0;
          emitMacroOverrides(Out, Overridden);
          if (DefMacroDirective *DefMD = dyn_cast<DefMacroDirective>(MD)) {
            InfoID = Writer.getMacroID(DefMD->getInfo());
            assert(InfoID);
            LE.write<uint32_t>(InfoID << 1);
          } else {
            assert(isa<UndefMacroDirective>(MD));
            LE.write<uint32_t>((ModID << 1) | 1);
          }
        }
        emitMacroOverrides(Out, Overridden);
        LE.write<uint32_t>(0);
      }
    }

    // Emit the declaration IDs in reverse order, because the
    // IdentifierResolver provides the declarations as they would be
    // visible (e.g., the function "stat" would come before the struct
    // "stat"), but the ASTReader adds declarations to the end of the list 
    // (so we need to see the struct "status" before the function "status").
    // Only emit declarations that aren't from a chained PCH, though.
    SmallVector<Decl *, 16> Decls(IdResolver.begin(II),
                                  IdResolver.end());
    for (SmallVectorImpl<Decl *>::reverse_iterator D = Decls.rbegin(),
                                                DEnd = Decls.rend();
         D != DEnd; ++D)
      LE.write<uint32_t>(Writer.getDeclID(getMostRecentLocalDecl(*D)));
  }

  /// \brief Returns the most recent local decl or the given decl if there are
  /// no local ones. The given decl is assumed to be the most recent one.
  Decl *getMostRecentLocalDecl(Decl *Orig) {
    // The only way a "from AST file" decl would be more recent from a local one
    // is if it came from a module.
    if (!PP.getLangOpts().Modules)
      return Orig;

    // Look for a local in the decl chain.
    for (Decl *D = Orig; D; D = D->getPreviousDecl()) {
      if (!D->isFromASTFile())
        return D;
      // If we come up a decl from a (chained-)PCH stop since we won't find a
      // local one.
      if (D->getOwningModuleID() == 0)
        break;
    }

    return Orig;
  }
};
} // end anonymous namespace

/// \brief Write the identifier table into the AST file.
///
/// The identifier table consists of a blob containing string data
/// (the actual identifiers themselves) and a separate "offsets" index
/// that maps identifier IDs to locations within the blob.
void ASTWriter::WriteIdentifierTable(Preprocessor &PP, 
                                     IdentifierResolver &IdResolver,
                                     bool IsModule) {
  using namespace llvm;

  // Create and write out the blob that contains the identifier
  // strings.
  {
    llvm::OnDiskChainedHashTableGenerator<ASTIdentifierTableTrait> Generator;
    ASTIdentifierTableTrait Trait(*this, PP, IdResolver, IsModule);

    // Look for any identifiers that were named while processing the
    // headers, but are otherwise not needed. We add these to the hash
    // table to enable checking of the predefines buffer in the case
    // where the user adds new macro definitions when building the AST
    // file.
    for (IdentifierTable::iterator ID = PP.getIdentifierTable().begin(),
                                IDEnd = PP.getIdentifierTable().end();
         ID != IDEnd; ++ID)
      getIdentifierRef(ID->second);

    // Create the on-disk hash table representation. We only store offsets
    // for identifiers that appear here for the first time.
    IdentifierOffsets.resize(NextIdentID - FirstIdentID);
    for (llvm::DenseMap<const IdentifierInfo *, IdentID>::iterator
           ID = IdentifierIDs.begin(), IDEnd = IdentifierIDs.end();
         ID != IDEnd; ++ID) {
      assert(ID->first && "NULL identifier in identifier table");
      if (!Chain || !ID->first->isFromAST() || 
          ID->first->hasChangedSinceDeserialization())
        Generator.insert(const_cast<IdentifierInfo *>(ID->first), ID->second,
                         Trait);
    }

    // Create the on-disk hash table in a buffer.
    SmallString<4096> IdentifierTable;
    uint32_t BucketOffset;
    {
      using namespace llvm::support;
      ASTIdentifierTableTrait Trait(*this, PP, IdResolver, IsModule);
      llvm::raw_svector_ostream Out(IdentifierTable);
      // Make sure that no bucket is at offset 0
      endian::Writer<little>(Out).write<uint32_t>(0);
      BucketOffset = Generator.Emit(Out, Trait);
    }

    // Create a blob abbreviation
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(IDENTIFIER_TABLE));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned IDTableAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the identifier table
    RecordData Record;
    Record.push_back(IDENTIFIER_TABLE);
    Record.push_back(BucketOffset);
    Stream.EmitRecordWithBlob(IDTableAbbrev, Record, IdentifierTable.str());
  }

  // Write the offsets table for identifier IDs.
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(IDENTIFIER_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of identifiers
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // first ID
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned IdentifierOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

#ifndef NDEBUG
  for (unsigned I = 0, N = IdentifierOffsets.size(); I != N; ++I)
    assert(IdentifierOffsets[I] && "Missing identifier offset?");
#endif
  
  RecordData Record;
  Record.push_back(IDENTIFIER_OFFSET);
  Record.push_back(IdentifierOffsets.size());
  Record.push_back(FirstIdentID - NUM_PREDEF_IDENT_IDS);
  Stream.EmitRecordWithBlob(IdentifierOffsetAbbrev, Record,
                            data(IdentifierOffsets));
}

//===----------------------------------------------------------------------===//
// DeclContext's Name Lookup Table Serialization
//===----------------------------------------------------------------------===//

namespace {
// Trait used for the on-disk hash table used in the method pool.
class ASTDeclContextNameLookupTrait {
  ASTWriter &Writer;

public:
  typedef DeclarationName key_type;
  typedef key_type key_type_ref;

  typedef DeclContext::lookup_result data_type;
  typedef const data_type& data_type_ref;

  typedef unsigned hash_value_type;
  typedef unsigned offset_type;

  explicit ASTDeclContextNameLookupTrait(ASTWriter &Writer) : Writer(Writer) { }

  hash_value_type ComputeHash(DeclarationName Name) {
    llvm::FoldingSetNodeID ID;
    ID.AddInteger(Name.getNameKind());

    switch (Name.getNameKind()) {
    case DeclarationName::Identifier:
      ID.AddString(Name.getAsIdentifierInfo()->getName());
      break;
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      ID.AddInteger(serialization::ComputeHash(Name.getObjCSelector()));
      break;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
      break;
    case DeclarationName::CXXOperatorName:
      ID.AddInteger(Name.getCXXOverloadedOperator());
      break;
    case DeclarationName::CXXLiteralOperatorName:
      ID.AddString(Name.getCXXLiteralIdentifier()->getName());
    case DeclarationName::CXXUsingDirective:
      break;
    }

    return ID.ComputeHash();
  }

  std::pair<unsigned,unsigned>
    EmitKeyDataLength(raw_ostream& Out, DeclarationName Name,
                      data_type_ref Lookup) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    unsigned KeyLen = 1;
    switch (Name.getNameKind()) {
    case DeclarationName::Identifier:
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
    case DeclarationName::CXXLiteralOperatorName:
      KeyLen += 4;
      break;
    case DeclarationName::CXXOperatorName:
      KeyLen += 1;
      break;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
    case DeclarationName::CXXUsingDirective:
      break;
    }
    LE.write<uint16_t>(KeyLen);

    // 2 bytes for num of decls and 4 for each DeclID.
    unsigned DataLen = 2 + 4 * Lookup.size();
    LE.write<uint16_t>(DataLen);

    return std::make_pair(KeyLen, DataLen);
  }

  void EmitKey(raw_ostream& Out, DeclarationName Name, unsigned) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    LE.write<uint8_t>(Name.getNameKind());
    switch (Name.getNameKind()) {
    case DeclarationName::Identifier:
      LE.write<uint32_t>(Writer.getIdentifierRef(Name.getAsIdentifierInfo()));
      return;
    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      LE.write<uint32_t>(Writer.getSelectorRef(Name.getObjCSelector()));
      return;
    case DeclarationName::CXXOperatorName:
      assert(Name.getCXXOverloadedOperator() < NUM_OVERLOADED_OPERATORS &&
             "Invalid operator?");
      LE.write<uint8_t>(Name.getCXXOverloadedOperator());
      return;
    case DeclarationName::CXXLiteralOperatorName:
      LE.write<uint32_t>(Writer.getIdentifierRef(Name.getCXXLiteralIdentifier()));
      return;
    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
    case DeclarationName::CXXUsingDirective:
      return;
    }

    llvm_unreachable("Invalid name kind?");
  }

  void EmitData(raw_ostream& Out, key_type_ref,
                data_type Lookup, unsigned DataLen) {
    using namespace llvm::support;
    endian::Writer<little> LE(Out);
    uint64_t Start = Out.tell(); (void)Start;
    LE.write<uint16_t>(Lookup.size());
    for (DeclContext::lookup_iterator I = Lookup.begin(), E = Lookup.end();
         I != E; ++I)
      LE.write<uint32_t>(Writer.GetDeclRef(*I));

    assert(Out.tell() - Start == DataLen && "Data length is wrong");
  }
};
} // end anonymous namespace

template<typename Visitor>
static void visitLocalLookupResults(const DeclContext *ConstDC,
                                    bool NeedToReconcileExternalVisibleStorage,
                                    Visitor AddLookupResult) {
  // FIXME: We need to build the lookups table, which is logically const.
  DeclContext *DC = const_cast<DeclContext*>(ConstDC);
  assert(DC == DC->getPrimaryContext() && "only primary DC has lookup table");

  SmallVector<DeclarationName, 16> ExternalNames;
  for (auto &Lookup : *DC->buildLookup()) {
    if (Lookup.second.hasExternalDecls() ||
        NeedToReconcileExternalVisibleStorage) {
      // We don't know for sure what declarations are found by this name,
      // because the external source might have a different set from the set
      // that are in the lookup map, and we can't update it now without
      // risking invalidating our lookup iterator. So add it to a queue to
      // deal with later.
      ExternalNames.push_back(Lookup.first);
      continue;
    }

    AddLookupResult(Lookup.first, Lookup.second.getLookupResult());
  }

  // Add the names we needed to defer. Note, this shouldn't add any new decls
  // to the list we need to serialize: any new declarations we find here should
  // be imported from an external source.
  // FIXME: What if the external source isn't an ASTReader?
  for (const auto &Name : ExternalNames)
    AddLookupResult(Name, DC->lookup(Name));
}

void ASTWriter::AddUpdatedDeclContext(const DeclContext *DC) {
  if (UpdatedDeclContexts.insert(DC) && WritingAST) {
    // Ensure we emit all the visible declarations.
    visitLocalLookupResults(DC, DC->NeedToReconcileExternalVisibleStorage,
                            [&](DeclarationName Name,
                                DeclContext::lookup_const_result Result) {
      for (auto *Decl : Result)
        GetDeclRef(Decl);
    });
  }
}

uint32_t
ASTWriter::GenerateNameLookupTable(const DeclContext *DC,
                                   llvm::SmallVectorImpl<char> &LookupTable) {
  assert(!DC->LookupPtr.getInt() && "must call buildLookups first");

  llvm::OnDiskChainedHashTableGenerator<ASTDeclContextNameLookupTrait>
      Generator;
  ASTDeclContextNameLookupTrait Trait(*this);

  // Create the on-disk hash table representation.
  DeclarationName ConstructorName;
  DeclarationName ConversionName;
  SmallVector<NamedDecl *, 8> ConstructorDecls;
  SmallVector<NamedDecl *, 4> ConversionDecls;

  visitLocalLookupResults(DC, DC->NeedToReconcileExternalVisibleStorage,
                          [&](DeclarationName Name,
                              DeclContext::lookup_result Result) {
    if (Result.empty())
      return;

    // Different DeclarationName values of certain kinds are mapped to
    // identical serialized keys, because we don't want to use type
    // identifiers in the keys (since type ids are local to the module).
    switch (Name.getNameKind()) {
    case DeclarationName::CXXConstructorName:
      // There may be different CXXConstructorName DeclarationName values
      // in a DeclContext because a UsingDecl that inherits constructors
      // has the DeclarationName of the inherited constructors.
      if (!ConstructorName)
        ConstructorName = Name;
      ConstructorDecls.append(Result.begin(), Result.end());
      return;

    case DeclarationName::CXXConversionFunctionName:
      if (!ConversionName)
        ConversionName = Name;
      ConversionDecls.append(Result.begin(), Result.end());
      return;

    default:
      break;
    }

    Generator.insert(Name, Result, Trait);
  });

  // Add the constructors.
  if (!ConstructorDecls.empty()) {
    Generator.insert(ConstructorName,
                     DeclContext::lookup_result(ConstructorDecls.begin(),
                                                ConstructorDecls.end()),
                     Trait);
  }

  // Add the conversion functions.
  if (!ConversionDecls.empty()) {
    Generator.insert(ConversionName,
                     DeclContext::lookup_result(ConversionDecls.begin(),
                                                ConversionDecls.end()),
                     Trait);
  }

  // Create the on-disk hash table in a buffer.
  llvm::raw_svector_ostream Out(LookupTable);
  // Make sure that no bucket is at offset 0
  using namespace llvm::support;
  endian::Writer<little>(Out).write<uint32_t>(0);
  return Generator.Emit(Out, Trait);
}

/// \brief Write the block containing all of the declaration IDs
/// visible from the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_VISIBLE block within the
/// bitstream, or 0 if no block was written.
uint64_t ASTWriter::WriteDeclContextVisibleBlock(ASTContext &Context,
                                                 DeclContext *DC) {
  if (DC->getPrimaryContext() != DC)
    return 0;

  // Since there is no name lookup into functions or methods, don't bother to
  // build a visible-declarations table for these entities.
  if (DC->isFunctionOrMethod())
    return 0;

  // If not in C++, we perform name lookup for the translation unit via the
  // IdentifierInfo chains, don't bother to build a visible-declarations table.
  if (DC->isTranslationUnit() && !Context.getLangOpts().CPlusPlus)
    return 0;

  // Serialize the contents of the mapping used for lookup. Note that,
  // although we have two very different code paths, the serialized
  // representation is the same for both cases: a declaration name,
  // followed by a size, followed by references to the visible
  // declarations that have that name.
  uint64_t Offset = Stream.GetCurrentBitNo();
  StoredDeclsMap *Map = DC->buildLookup();
  if (!Map || Map->empty())
    return 0;

  // Create the on-disk hash table in a buffer.
  SmallString<4096> LookupTable;
  uint32_t BucketOffset = GenerateNameLookupTable(DC, LookupTable);

  // Write the lookup table
  RecordData Record;
  Record.push_back(DECL_CONTEXT_VISIBLE);
  Record.push_back(BucketOffset);
  Stream.EmitRecordWithBlob(DeclContextVisibleLookupAbbrev, Record,
                            LookupTable.str());

  Stream.EmitRecord(DECL_CONTEXT_VISIBLE, Record);
  ++NumVisibleDeclContexts;
  return Offset;
}

/// \brief Write an UPDATE_VISIBLE block for the given context.
///
/// UPDATE_VISIBLE blocks contain the declarations that are added to an existing
/// DeclContext in a dependent AST file. As such, they only exist for the TU
/// (in C++), for namespaces, and for classes with forward-declared unscoped
/// enumeration members (in C++11).
void ASTWriter::WriteDeclContextVisibleUpdate(const DeclContext *DC) {
  StoredDeclsMap *Map = DC->getLookupPtr();
  if (!Map || Map->empty())
    return;

  // Create the on-disk hash table in a buffer.
  SmallString<4096> LookupTable;
  uint32_t BucketOffset = GenerateNameLookupTable(DC, LookupTable);

  // Write the lookup table
  RecordData Record;
  Record.push_back(UPDATE_VISIBLE);
  Record.push_back(getDeclID(cast<Decl>(DC)));
  Record.push_back(BucketOffset);
  Stream.EmitRecordWithBlob(UpdateVisibleAbbrev, Record, LookupTable.str());
}

/// \brief Write an FP_PRAGMA_OPTIONS block for the given FPOptions.
void ASTWriter::WriteFPPragmaOptions(const FPOptions &Opts) {
  RecordData Record;
  Record.push_back(Opts.fp_contract);
  Stream.EmitRecord(FP_PRAGMA_OPTIONS, Record);
}

/// \brief Write an OPENCL_EXTENSIONS block for the given OpenCLOptions.
void ASTWriter::WriteOpenCLExtensions(Sema &SemaRef) {
  if (!SemaRef.Context.getLangOpts().OpenCL)
    return;

  const OpenCLOptions &Opts = SemaRef.getOpenCLOptions();
  RecordData Record;
#define OPENCLEXT(nm)  Record.push_back(Opts.nm);
#include "clang/Basic/OpenCLExtensions.def"
  Stream.EmitRecord(OPENCL_EXTENSIONS, Record);
}

void ASTWriter::WriteRedeclarations() {
  RecordData LocalRedeclChains;
  SmallVector<serialization::LocalRedeclarationsInfo, 2> LocalRedeclsMap;

  for (unsigned I = 0, N = Redeclarations.size(); I != N; ++I) {
    Decl *First = Redeclarations[I];
    assert(First->isFirstDecl() && "Not the first declaration?");
    
    Decl *MostRecent = First->getMostRecentDecl();
    
    // If we only have a single declaration, there is no point in storing
    // a redeclaration chain.
    if (First == MostRecent)
      continue;
    
    unsigned Offset = LocalRedeclChains.size();
    unsigned Size = 0;
    LocalRedeclChains.push_back(0); // Placeholder for the size.
    
    // Collect the set of local redeclarations of this declaration.
    for (Decl *Prev = MostRecent; Prev != First;
         Prev = Prev->getPreviousDecl()) { 
      if (!Prev->isFromASTFile()) {
        AddDeclRef(Prev, LocalRedeclChains);
        ++Size;
      }
    }

    if (!First->isFromASTFile() && Chain) {
      Decl *FirstFromAST = MostRecent;
      for (Decl *Prev = MostRecent; Prev; Prev = Prev->getPreviousDecl()) {
        if (Prev->isFromASTFile())
          FirstFromAST = Prev;
      }

      Chain->MergedDecls[FirstFromAST].push_back(getDeclID(First));
    }

    LocalRedeclChains[Offset] = Size;
    
    // Reverse the set of local redeclarations, so that we store them in
    // order (since we found them in reverse order).
    std::reverse(LocalRedeclChains.end() - Size, LocalRedeclChains.end());
    
    // Add the mapping from the first ID from the AST to the set of local
    // declarations.
    LocalRedeclarationsInfo Info = { getDeclID(First), Offset };
    LocalRedeclsMap.push_back(Info);
    
    assert(N == Redeclarations.size() && 
           "Deserialized a declaration we shouldn't have");
  }
  
  if (LocalRedeclChains.empty())
    return;
  
  // Sort the local redeclarations map by the first declaration ID,
  // since the reader will be performing binary searches on this information.
  llvm::array_pod_sort(LocalRedeclsMap.begin(), LocalRedeclsMap.end());
  
  // Emit the local redeclarations map.
  using namespace llvm;
  llvm::BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(LOCAL_REDECLARATIONS_MAP));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // # of entries
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned AbbrevID = Stream.EmitAbbrev(Abbrev);
  
  RecordData Record;
  Record.push_back(LOCAL_REDECLARATIONS_MAP);
  Record.push_back(LocalRedeclsMap.size());
  Stream.EmitRecordWithBlob(AbbrevID, Record, 
    reinterpret_cast<char*>(LocalRedeclsMap.data()),
    LocalRedeclsMap.size() * sizeof(LocalRedeclarationsInfo));

  // Emit the redeclaration chains.
  Stream.EmitRecord(LOCAL_REDECLARATIONS, LocalRedeclChains);
}

void ASTWriter::WriteObjCCategories() {
  SmallVector<ObjCCategoriesInfo, 2> CategoriesMap;
  RecordData Categories;
  
  for (unsigned I = 0, N = ObjCClassesWithCategories.size(); I != N; ++I) {
    unsigned Size = 0;
    unsigned StartIndex = Categories.size();
    
    ObjCInterfaceDecl *Class = ObjCClassesWithCategories[I];
    
    // Allocate space for the size.
    Categories.push_back(0);
    
    // Add the categories.
    for (ObjCInterfaceDecl::known_categories_iterator
           Cat = Class->known_categories_begin(),
           CatEnd = Class->known_categories_end();
         Cat != CatEnd; ++Cat, ++Size) {
      assert(getDeclID(*Cat) != 0 && "Bogus category");
      AddDeclRef(*Cat, Categories);
    }
    
    // Update the size.
    Categories[StartIndex] = Size;
    
    // Record this interface -> category map.
    ObjCCategoriesInfo CatInfo = { getDeclID(Class), StartIndex };
    CategoriesMap.push_back(CatInfo);
  }

  // Sort the categories map by the definition ID, since the reader will be
  // performing binary searches on this information.
  llvm::array_pod_sort(CategoriesMap.begin(), CategoriesMap.end());

  // Emit the categories map.
  using namespace llvm;
  llvm::BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(OBJC_CATEGORIES_MAP));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // # of entries
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned AbbrevID = Stream.EmitAbbrev(Abbrev);
  
  RecordData Record;
  Record.push_back(OBJC_CATEGORIES_MAP);
  Record.push_back(CategoriesMap.size());
  Stream.EmitRecordWithBlob(AbbrevID, Record, 
                            reinterpret_cast<char*>(CategoriesMap.data()),
                            CategoriesMap.size() * sizeof(ObjCCategoriesInfo));
  
  // Emit the category lists.
  Stream.EmitRecord(OBJC_CATEGORIES, Categories);
}

void ASTWriter::WriteMergedDecls() {
  if (!Chain || Chain->MergedDecls.empty())
    return;
  
  RecordData Record;
  for (ASTReader::MergedDeclsMap::iterator I = Chain->MergedDecls.begin(),
                                        IEnd = Chain->MergedDecls.end();
       I != IEnd; ++I) {
    DeclID CanonID = I->first->isFromASTFile()? I->first->getGlobalID()
                                              : GetDeclRef(I->first);
    assert(CanonID && "Merged declaration not known?");
    
    Record.push_back(CanonID);
    Record.push_back(I->second.size());
    Record.append(I->second.begin(), I->second.end());
  }
  Stream.EmitRecord(MERGED_DECLARATIONS, Record);
}

void ASTWriter::WriteLateParsedTemplates(Sema &SemaRef) {
  Sema::LateParsedTemplateMapT &LPTMap = SemaRef.LateParsedTemplateMap;

  if (LPTMap.empty())
    return;

  RecordData Record;
  for (Sema::LateParsedTemplateMapT::iterator It = LPTMap.begin(),
                                              ItEnd = LPTMap.end();
       It != ItEnd; ++It) {
    LateParsedTemplate *LPT = It->second;
    AddDeclRef(It->first, Record);
    AddDeclRef(LPT->D, Record);
    Record.push_back(LPT->Toks.size());

    for (CachedTokens::iterator TokIt = LPT->Toks.begin(),
                                TokEnd = LPT->Toks.end();
         TokIt != TokEnd; ++TokIt) {
      AddToken(*TokIt, Record);
    }
  }
  Stream.EmitRecord(LATE_PARSED_TEMPLATE, Record);
}

/// \brief Write the state of 'pragma clang optimize' at the end of the module.
void ASTWriter::WriteOptimizePragmaOptions(Sema &SemaRef) {
  RecordData Record;
  SourceLocation PragmaLoc = SemaRef.getOptimizeOffPragmaLocation();
  AddSourceLocation(PragmaLoc, Record);
  Stream.EmitRecord(OPTIMIZE_PRAGMA_OPTIONS, Record);
}

//===----------------------------------------------------------------------===//
// General Serialization Routines
//===----------------------------------------------------------------------===//

/// \brief Write a record containing the given attributes.
void ASTWriter::WriteAttributes(ArrayRef<const Attr*> Attrs,
                                RecordDataImpl &Record) {
  Record.push_back(Attrs.size());
  for (ArrayRef<const Attr *>::iterator i = Attrs.begin(),
                                        e = Attrs.end(); i != e; ++i){
    const Attr *A = *i;
    Record.push_back(A->getKind()); // FIXME: stable encoding, target attrs
    AddSourceRange(A->getRange(), Record);

#include "clang/Serialization/AttrPCHWrite.inc"

  }
}

void ASTWriter::AddToken(const Token &Tok, RecordDataImpl &Record) {
  AddSourceLocation(Tok.getLocation(), Record);
  Record.push_back(Tok.getLength());

  // FIXME: When reading literal tokens, reconstruct the literal pointer
  // if it is needed.
  AddIdentifierRef(Tok.getIdentifierInfo(), Record);
  // FIXME: Should translate token kind to a stable encoding.
  Record.push_back(Tok.getKind());
  // FIXME: Should translate token flags to a stable encoding.
  Record.push_back(Tok.getFlags());
}

void ASTWriter::AddString(StringRef Str, RecordDataImpl &Record) {
  Record.push_back(Str.size());
  Record.insert(Record.end(), Str.begin(), Str.end());
}

void ASTWriter::AddVersionTuple(const VersionTuple &Version,
                                RecordDataImpl &Record) {
  Record.push_back(Version.getMajor());
  if (Optional<unsigned> Minor = Version.getMinor())
    Record.push_back(*Minor + 1);
  else
    Record.push_back(0);
  if (Optional<unsigned> Subminor = Version.getSubminor())
    Record.push_back(*Subminor + 1);
  else
    Record.push_back(0);
}

/// \brief Note that the identifier II occurs at the given offset
/// within the identifier table.
void ASTWriter::SetIdentifierOffset(const IdentifierInfo *II, uint32_t Offset) {
  IdentID ID = IdentifierIDs[II];
  // Only store offsets new to this AST file. Other identifier names are looked
  // up earlier in the chain and thus don't need an offset.
  if (ID >= FirstIdentID)
    IdentifierOffsets[ID - FirstIdentID] = Offset;
}

/// \brief Note that the selector Sel occurs at the given offset
/// within the method pool/selector table.
void ASTWriter::SetSelectorOffset(Selector Sel, uint32_t Offset) {
  unsigned ID = SelectorIDs[Sel];
  assert(ID && "Unknown selector");
  // Don't record offsets for selectors that are also available in a different
  // file.
  if (ID < FirstSelectorID)
    return;
  SelectorOffsets[ID - FirstSelectorID] = Offset;
}

ASTWriter::ASTWriter(llvm::BitstreamWriter &Stream)
  : Stream(Stream), Context(nullptr), PP(nullptr), Chain(nullptr),
    WritingModule(nullptr), WritingAST(false), DoneWritingDeclsAndTypes(false),
    ASTHasCompilerErrors(false),
    FirstDeclID(NUM_PREDEF_DECL_IDS), NextDeclID(FirstDeclID),
    FirstTypeID(NUM_PREDEF_TYPE_IDS), NextTypeID(FirstTypeID),
    FirstIdentID(NUM_PREDEF_IDENT_IDS), NextIdentID(FirstIdentID),
    FirstMacroID(NUM_PREDEF_MACRO_IDS), NextMacroID(FirstMacroID),
    FirstSubmoduleID(NUM_PREDEF_SUBMODULE_IDS), 
    NextSubmoduleID(FirstSubmoduleID),
    FirstSelectorID(NUM_PREDEF_SELECTOR_IDS), NextSelectorID(FirstSelectorID),
    CollectedStmts(&StmtsToEmit),
    NumStatements(0), NumMacros(0), NumLexicalDeclContexts(0),
    NumVisibleDeclContexts(0),
    NextCXXBaseSpecifiersID(1),
    DeclParmVarAbbrev(0), DeclContextLexicalAbbrev(0),
    DeclContextVisibleLookupAbbrev(0), UpdateVisibleAbbrev(0),
    DeclRefExprAbbrev(0), CharacterLiteralAbbrev(0),
    DeclRecordAbbrev(0), IntegerLiteralAbbrev(0),
    DeclTypedefAbbrev(0),
    DeclVarAbbrev(0), DeclFieldAbbrev(0),
    DeclEnumAbbrev(0), DeclObjCIvarAbbrev(0)
{
}

ASTWriter::~ASTWriter() {
  llvm::DeleteContainerSeconds(FileDeclIDs);
}

void ASTWriter::WriteAST(Sema &SemaRef,
                         const std::string &OutputFile,
                         Module *WritingModule, StringRef isysroot,
                         bool hasErrors) {
  WritingAST = true;
  
  ASTHasCompilerErrors = hasErrors;
  
  // Emit the file header.
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'P', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'H', 8);

  WriteBlockInfoBlock();

  Context = &SemaRef.Context;
  PP = &SemaRef.PP;
  this->WritingModule = WritingModule;
  WriteASTCore(SemaRef, isysroot, OutputFile, WritingModule);
  Context = nullptr;
  PP = nullptr;
  this->WritingModule = nullptr;

  WritingAST = false;
}

template<typename Vector>
static void AddLazyVectorDecls(ASTWriter &Writer, Vector &Vec,
                               ASTWriter::RecordData &Record) {
  for (typename Vector::iterator I = Vec.begin(nullptr, true), E = Vec.end();
       I != E; ++I) {
    Writer.AddDeclRef(*I, Record);
  }
}

void ASTWriter::WriteASTCore(Sema &SemaRef,
                             StringRef isysroot,
                             const std::string &OutputFile, 
                             Module *WritingModule) {
  using namespace llvm;

  bool isModule = WritingModule != nullptr;

  // Make sure that the AST reader knows to finalize itself.
  if (Chain)
    Chain->finalizeForWriting();
  
  ASTContext &Context = SemaRef.Context;
  Preprocessor &PP = SemaRef.PP;

  // Set up predefined declaration IDs.
  DeclIDs[Context.getTranslationUnitDecl()] = PREDEF_DECL_TRANSLATION_UNIT_ID;
  if (Context.ObjCIdDecl)
    DeclIDs[Context.ObjCIdDecl] = PREDEF_DECL_OBJC_ID_ID;
  if (Context.ObjCSelDecl)
    DeclIDs[Context.ObjCSelDecl] = PREDEF_DECL_OBJC_SEL_ID;
  if (Context.ObjCClassDecl)
    DeclIDs[Context.ObjCClassDecl] = PREDEF_DECL_OBJC_CLASS_ID;
  if (Context.ObjCProtocolClassDecl)
    DeclIDs[Context.ObjCProtocolClassDecl] = PREDEF_DECL_OBJC_PROTOCOL_ID;
  if (Context.Int128Decl)
    DeclIDs[Context.Int128Decl] = PREDEF_DECL_INT_128_ID;
  if (Context.UInt128Decl)
    DeclIDs[Context.UInt128Decl] = PREDEF_DECL_UNSIGNED_INT_128_ID;
  if (Context.ObjCInstanceTypeDecl)
    DeclIDs[Context.ObjCInstanceTypeDecl] = PREDEF_DECL_OBJC_INSTANCETYPE_ID;
  if (Context.BuiltinVaListDecl)
    DeclIDs[Context.getBuiltinVaListDecl()] = PREDEF_DECL_BUILTIN_VA_LIST_ID;

  if (!Chain) {
    // Make sure that we emit IdentifierInfos (and any attached
    // declarations) for builtins. We don't need to do this when we're
    // emitting chained PCH files, because all of the builtins will be
    // in the original PCH file.
    // FIXME: Modules won't like this at all.
    IdentifierTable &Table = PP.getIdentifierTable();
    SmallVector<const char *, 32> BuiltinNames;
    if (!Context.getLangOpts().NoBuiltin) {
      Context.BuiltinInfo.GetBuiltinNames(BuiltinNames);
    }
    for (unsigned I = 0, N = BuiltinNames.size(); I != N; ++I)
      getIdentifierRef(&Table.get(BuiltinNames[I]));
  }

  // If there are any out-of-date identifiers, bring them up to date.
  if (ExternalPreprocessorSource *ExtSource = PP.getExternalSource()) {
    // Find out-of-date identifiers.
    SmallVector<IdentifierInfo *, 4> OutOfDate;
    for (IdentifierTable::iterator ID = PP.getIdentifierTable().begin(),
                                IDEnd = PP.getIdentifierTable().end();
         ID != IDEnd; ++ID) {
      if (ID->second->isOutOfDate())
        OutOfDate.push_back(ID->second);
    }

    // Update the out-of-date identifiers.
    for (unsigned I = 0, N = OutOfDate.size(); I != N; ++I) {
      ExtSource->updateOutOfDateIdentifier(*OutOfDate[I]);
    }
  }

  // If we saw any DeclContext updates before we started writing the AST file,
  // make sure all visible decls in those DeclContexts are written out.
  if (!UpdatedDeclContexts.empty()) {
    auto OldUpdatedDeclContexts = std::move(UpdatedDeclContexts);
    UpdatedDeclContexts.clear();
    for (auto *DC : OldUpdatedDeclContexts)
      AddUpdatedDeclContext(DC);
  }

  // Build a record containing all of the tentative definitions in this file, in
  // TentativeDefinitions order.  Generally, this record will be empty for
  // headers.
  RecordData TentativeDefinitions;
  AddLazyVectorDecls(*this, SemaRef.TentativeDefinitions, TentativeDefinitions);
  
  // Build a record containing all of the file scoped decls in this file.
  RecordData UnusedFileScopedDecls;
  if (!isModule)
    AddLazyVectorDecls(*this, SemaRef.UnusedFileScopedDecls,
                       UnusedFileScopedDecls);

  // Build a record containing all of the delegating constructors we still need
  // to resolve.
  RecordData DelegatingCtorDecls;
  if (!isModule)
    AddLazyVectorDecls(*this, SemaRef.DelegatingCtorDecls, DelegatingCtorDecls);

  // Write the set of weak, undeclared identifiers. We always write the
  // entire table, since later PCH files in a PCH chain are only interested in
  // the results at the end of the chain.
  RecordData WeakUndeclaredIdentifiers;
  if (!SemaRef.WeakUndeclaredIdentifiers.empty()) {
    for (llvm::DenseMap<IdentifierInfo*,WeakInfo>::iterator
         I = SemaRef.WeakUndeclaredIdentifiers.begin(),
         E = SemaRef.WeakUndeclaredIdentifiers.end(); I != E; ++I) {
      AddIdentifierRef(I->first, WeakUndeclaredIdentifiers);
      AddIdentifierRef(I->second.getAlias(), WeakUndeclaredIdentifiers);
      AddSourceLocation(I->second.getLocation(), WeakUndeclaredIdentifiers);
      WeakUndeclaredIdentifiers.push_back(I->second.getUsed());
    }
  }

  // Build a record containing all of the locally-scoped extern "C"
  // declarations in this header file. Generally, this record will be
  // empty.
  RecordData LocallyScopedExternCDecls;
  // FIXME: This is filling in the AST file in densemap order which is
  // nondeterminstic!
  for (llvm::DenseMap<DeclarationName, NamedDecl *>::iterator
         TD = SemaRef.LocallyScopedExternCDecls.begin(),
         TDEnd = SemaRef.LocallyScopedExternCDecls.end();
       TD != TDEnd; ++TD) {
    if (!TD->second->isFromASTFile())
      AddDeclRef(TD->second, LocallyScopedExternCDecls);
  }
  
  // Build a record containing all of the ext_vector declarations.
  RecordData ExtVectorDecls;
  AddLazyVectorDecls(*this, SemaRef.ExtVectorDecls, ExtVectorDecls);

  // Build a record containing all of the VTable uses information.
  RecordData VTableUses;
  if (!SemaRef.VTableUses.empty()) {
    for (unsigned I = 0, N = SemaRef.VTableUses.size(); I != N; ++I) {
      AddDeclRef(SemaRef.VTableUses[I].first, VTableUses);
      AddSourceLocation(SemaRef.VTableUses[I].second, VTableUses);
      VTableUses.push_back(SemaRef.VTablesUsed[SemaRef.VTableUses[I].first]);
    }
  }

  // Build a record containing all of dynamic classes declarations.
  RecordData DynamicClasses;
  AddLazyVectorDecls(*this, SemaRef.DynamicClasses, DynamicClasses);

  // Build a record containing all of pending implicit instantiations.
  RecordData PendingInstantiations;
  for (std::deque<Sema::PendingImplicitInstantiation>::iterator
         I = SemaRef.PendingInstantiations.begin(),
         N = SemaRef.PendingInstantiations.end(); I != N; ++I) {
    AddDeclRef(I->first, PendingInstantiations);
    AddSourceLocation(I->second, PendingInstantiations);
  }
  assert(SemaRef.PendingLocalImplicitInstantiations.empty() &&
         "There are local ones at end of translation unit!");

  // Build a record containing some declaration references.
  RecordData SemaDeclRefs;
  if (SemaRef.StdNamespace || SemaRef.StdBadAlloc) {
    AddDeclRef(SemaRef.getStdNamespace(), SemaDeclRefs);
    AddDeclRef(SemaRef.getStdBadAlloc(), SemaDeclRefs);
  }

  RecordData CUDASpecialDeclRefs;
  if (Context.getcudaConfigureCallDecl()) {
    AddDeclRef(Context.getcudaConfigureCallDecl(), CUDASpecialDeclRefs);
  }

  // Build a record containing all of the known namespaces.
  RecordData KnownNamespaces;
  for (llvm::MapVector<NamespaceDecl*, bool>::iterator
            I = SemaRef.KnownNamespaces.begin(),
         IEnd = SemaRef.KnownNamespaces.end();
       I != IEnd; ++I) {
    if (!I->second)
      AddDeclRef(I->first, KnownNamespaces);
  }

  // Build a record of all used, undefined objects that require definitions.
  RecordData UndefinedButUsed;

  SmallVector<std::pair<NamedDecl *, SourceLocation>, 16> Undefined;
  SemaRef.getUndefinedButUsed(Undefined);
  for (SmallVectorImpl<std::pair<NamedDecl *, SourceLocation> >::iterator
         I = Undefined.begin(), E = Undefined.end(); I != E; ++I) {
    AddDeclRef(I->first, UndefinedButUsed);
    AddSourceLocation(I->second, UndefinedButUsed);
  }

  // Write the control block
  WriteControlBlock(PP, Context, isysroot, OutputFile);

  // Write the remaining AST contents.
  RecordData Record;
  Stream.EnterSubblock(AST_BLOCK_ID, 5);

  // This is so that older clang versions, before the introduction
  // of the control block, can read and reject the newer PCH format.
  Record.clear();
  Record.push_back(VERSION_MAJOR);
  Stream.EmitRecord(METADATA_OLD_FORMAT, Record);

  // Create a lexical update block containing all of the declarations in the
  // translation unit that do not come from other AST files.
  const TranslationUnitDecl *TU = Context.getTranslationUnitDecl();
  SmallVector<KindDeclIDPair, 64> NewGlobalDecls;
  for (const auto *I : TU->noload_decls()) {
    if (!I->isFromASTFile())
      NewGlobalDecls.push_back(std::make_pair(I->getKind(), GetDeclRef(I)));
  }
  
  llvm::BitCodeAbbrev *Abv = new llvm::BitCodeAbbrev();
  Abv->Add(llvm::BitCodeAbbrevOp(TU_UPDATE_LEXICAL));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Blob));
  unsigned TuUpdateLexicalAbbrev = Stream.EmitAbbrev(Abv);
  Record.clear();
  Record.push_back(TU_UPDATE_LEXICAL);
  Stream.EmitRecordWithBlob(TuUpdateLexicalAbbrev, Record,
                            data(NewGlobalDecls));
  
  // And a visible updates block for the translation unit.
  Abv = new llvm::BitCodeAbbrev();
  Abv->Add(llvm::BitCodeAbbrevOp(UPDATE_VISIBLE));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::VBR, 6));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Fixed, 32));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Blob));
  UpdateVisibleAbbrev = Stream.EmitAbbrev(Abv);
  WriteDeclContextVisibleUpdate(TU);
  
  // If the translation unit has an anonymous namespace, and we don't already
  // have an update block for it, write it as an update block.
  // FIXME: Why do we not do this if there's already an update block?
  if (NamespaceDecl *NS = TU->getAnonymousNamespace()) {
    ASTWriter::UpdateRecord &Record = DeclUpdates[TU];
    if (Record.empty())
      Record.push_back(DeclUpdate(UPD_CXX_ADDED_ANONYMOUS_NAMESPACE, NS));
  }

  // Add update records for all mangling numbers and static local numbers.
  // These aren't really update records, but this is a convenient way of
  // tagging this rare extra data onto the declarations.
  for (const auto &Number : Context.MangleNumbers)
    if (!Number.first->isFromASTFile())
      DeclUpdates[Number.first].push_back(DeclUpdate(UPD_MANGLING_NUMBER,
                                                     Number.second));
  for (const auto &Number : Context.StaticLocalNumbers)
    if (!Number.first->isFromASTFile())
      DeclUpdates[Number.first].push_back(DeclUpdate(UPD_STATIC_LOCAL_NUMBER,
                                                     Number.second));

  // Make sure visible decls, added to DeclContexts previously loaded from
  // an AST file, are registered for serialization.
  for (SmallVectorImpl<const Decl *>::iterator
         I = UpdatingVisibleDecls.begin(),
         E = UpdatingVisibleDecls.end(); I != E; ++I) {
    GetDeclRef(*I);
  }

  // Make sure all decls associated with an identifier are registered for
  // serialization.
  for (IdentifierTable::iterator ID = PP.getIdentifierTable().begin(),
                              IDEnd = PP.getIdentifierTable().end();
       ID != IDEnd; ++ID) {
    const IdentifierInfo *II = ID->second;
    if (!Chain || !II->isFromAST() || II->hasChangedSinceDeserialization()) {
      for (IdentifierResolver::iterator D = SemaRef.IdResolver.begin(II),
                                     DEnd = SemaRef.IdResolver.end();
           D != DEnd; ++D) {
        GetDeclRef(*D);
      }
    }
  }

  // Form the record of special types.
  RecordData SpecialTypes;
  AddTypeRef(Context.getRawCFConstantStringType(), SpecialTypes);
  AddTypeRef(Context.getFILEType(), SpecialTypes);
  AddTypeRef(Context.getjmp_bufType(), SpecialTypes);
  AddTypeRef(Context.getsigjmp_bufType(), SpecialTypes);
  AddTypeRef(Context.ObjCIdRedefinitionType, SpecialTypes);
  AddTypeRef(Context.ObjCClassRedefinitionType, SpecialTypes);
  AddTypeRef(Context.ObjCSelRedefinitionType, SpecialTypes);
  AddTypeRef(Context.getucontext_tType(), SpecialTypes);

  if (Chain) {
    // Write the mapping information describing our module dependencies and how
    // each of those modules were mapped into our own offset/ID space, so that
    // the reader can build the appropriate mapping to its own offset/ID space.
    // The map consists solely of a blob with the following format:
    // *(module-name-len:i16 module-name:len*i8
    //   source-location-offset:i32
    //   identifier-id:i32
    //   preprocessed-entity-id:i32
    //   macro-definition-id:i32
    //   submodule-id:i32
    //   selector-id:i32
    //   declaration-id:i32
    //   c++-base-specifiers-id:i32
    //   type-id:i32)
    // 
    llvm::BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(MODULE_OFFSET_MAP));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned ModuleOffsetMapAbbrev = Stream.EmitAbbrev(Abbrev);
    SmallString<2048> Buffer;
    {
      llvm::raw_svector_ostream Out(Buffer);
      for (ModuleManager::ModuleConstIterator M = Chain->ModuleMgr.begin(),
                                           MEnd = Chain->ModuleMgr.end();
           M != MEnd; ++M) {
        using namespace llvm::support;
        endian::Writer<little> LE(Out);
        StringRef FileName = (*M)->FileName;
        LE.write<uint16_t>(FileName.size());
        Out.write(FileName.data(), FileName.size());
        LE.write<uint32_t>((*M)->SLocEntryBaseOffset);
        LE.write<uint32_t>((*M)->BaseIdentifierID);
        LE.write<uint32_t>((*M)->BaseMacroID);
        LE.write<uint32_t>((*M)->BasePreprocessedEntityID);
        LE.write<uint32_t>((*M)->BaseSubmoduleID);
        LE.write<uint32_t>((*M)->BaseSelectorID);
        LE.write<uint32_t>((*M)->BaseDeclID);
        LE.write<uint32_t>((*M)->BaseTypeIndex);
      }
    }
    Record.clear();
    Record.push_back(MODULE_OFFSET_MAP);
    Stream.EmitRecordWithBlob(ModuleOffsetMapAbbrev, Record,
                              Buffer.data(), Buffer.size());
  }

  RecordData DeclUpdatesOffsetsRecord;

  // Keep writing types, declarations, and declaration update records
  // until we've emitted all of them.
  Stream.EnterSubblock(DECLTYPES_BLOCK_ID, NUM_ALLOWED_ABBREVS_SIZE);
  WriteDeclsBlockAbbrevs();
  for (DeclsToRewriteTy::iterator I = DeclsToRewrite.begin(),
                                  E = DeclsToRewrite.end();
       I != E; ++I)
    DeclTypesToEmit.push(const_cast<Decl*>(*I));
  do {
    WriteDeclUpdatesBlocks(DeclUpdatesOffsetsRecord);
    while (!DeclTypesToEmit.empty()) {
      DeclOrType DOT = DeclTypesToEmit.front();
      DeclTypesToEmit.pop();
      if (DOT.isType())
        WriteType(DOT.getType());
      else
        WriteDecl(Context, DOT.getDecl());
    }
  } while (!DeclUpdates.empty());
  Stream.ExitBlock();

  DoneWritingDeclsAndTypes = true;

  // These things can only be done once we've written out decls and types.
  WriteTypeDeclOffsets();
  if (!DeclUpdatesOffsetsRecord.empty())
    Stream.EmitRecord(DECL_UPDATE_OFFSETS, DeclUpdatesOffsetsRecord);
  WriteCXXBaseSpecifiersOffsets();
  WriteFileDeclIDsMap();
  WriteSourceManagerBlock(Context.getSourceManager(), PP, isysroot);

  WriteComments();
  WritePreprocessor(PP, isModule);
  WriteHeaderSearch(PP.getHeaderSearchInfo(), isysroot);
  WriteSelectors(SemaRef);
  WriteReferencedSelectorsPool(SemaRef);
  WriteIdentifierTable(PP, SemaRef.IdResolver, isModule);
  WriteFPPragmaOptions(SemaRef.getFPOptions());
  WriteOpenCLExtensions(SemaRef);
  WritePragmaDiagnosticMappings(Context.getDiagnostics(), isModule);

  // If we're emitting a module, write out the submodule information.  
  if (WritingModule)
    WriteSubmodules(WritingModule);

  Stream.EmitRecord(SPECIAL_TYPES, SpecialTypes);

  // Write the record containing external, unnamed definitions.
  if (!EagerlyDeserializedDecls.empty())
    Stream.EmitRecord(EAGERLY_DESERIALIZED_DECLS, EagerlyDeserializedDecls);

  // Write the record containing tentative definitions.
  if (!TentativeDefinitions.empty())
    Stream.EmitRecord(TENTATIVE_DEFINITIONS, TentativeDefinitions);

  // Write the record containing unused file scoped decls.
  if (!UnusedFileScopedDecls.empty())
    Stream.EmitRecord(UNUSED_FILESCOPED_DECLS, UnusedFileScopedDecls);

  // Write the record containing weak undeclared identifiers.
  if (!WeakUndeclaredIdentifiers.empty())
    Stream.EmitRecord(WEAK_UNDECLARED_IDENTIFIERS,
                      WeakUndeclaredIdentifiers);

  // Write the record containing locally-scoped extern "C" definitions.
  if (!LocallyScopedExternCDecls.empty())
    Stream.EmitRecord(LOCALLY_SCOPED_EXTERN_C_DECLS,
                      LocallyScopedExternCDecls);

  // Write the record containing ext_vector type names.
  if (!ExtVectorDecls.empty())
    Stream.EmitRecord(EXT_VECTOR_DECLS, ExtVectorDecls);

  // Write the record containing VTable uses information.
  if (!VTableUses.empty())
    Stream.EmitRecord(VTABLE_USES, VTableUses);

  // Write the record containing dynamic classes declarations.
  if (!DynamicClasses.empty())
    Stream.EmitRecord(DYNAMIC_CLASSES, DynamicClasses);

  // Write the record containing pending implicit instantiations.
  if (!PendingInstantiations.empty())
    Stream.EmitRecord(PENDING_IMPLICIT_INSTANTIATIONS, PendingInstantiations);

  // Write the record containing declaration references of Sema.
  if (!SemaDeclRefs.empty())
    Stream.EmitRecord(SEMA_DECL_REFS, SemaDeclRefs);

  // Write the record containing CUDA-specific declaration references.
  if (!CUDASpecialDeclRefs.empty())
    Stream.EmitRecord(CUDA_SPECIAL_DECL_REFS, CUDASpecialDeclRefs);
  
  // Write the delegating constructors.
  if (!DelegatingCtorDecls.empty())
    Stream.EmitRecord(DELEGATING_CTORS, DelegatingCtorDecls);

  // Write the known namespaces.
  if (!KnownNamespaces.empty())
    Stream.EmitRecord(KNOWN_NAMESPACES, KnownNamespaces);

  // Write the undefined internal functions and variables, and inline functions.
  if (!UndefinedButUsed.empty())
    Stream.EmitRecord(UNDEFINED_BUT_USED, UndefinedButUsed);
  
  // Write the visible updates to DeclContexts.
  for (auto *DC : UpdatedDeclContexts)
    WriteDeclContextVisibleUpdate(DC);

  if (!WritingModule) {
    // Write the submodules that were imported, if any.
    struct ModuleInfo {
      uint64_t ID;
      Module *M;
      ModuleInfo(uint64_t ID, Module *M) : ID(ID), M(M) {}
    };
    llvm::SmallVector<ModuleInfo, 64> Imports;
    for (const auto *I : Context.local_imports()) {
      assert(SubmoduleIDs.find(I->getImportedModule()) != SubmoduleIDs.end());
      Imports.push_back(ModuleInfo(SubmoduleIDs[I->getImportedModule()],
                         I->getImportedModule()));
    }

    if (!Imports.empty()) {
      auto Cmp = [](const ModuleInfo &A, const ModuleInfo &B) {
        return A.ID < B.ID;
      };

      // Sort and deduplicate module IDs.
      std::sort(Imports.begin(), Imports.end(), Cmp);
      Imports.erase(std::unique(Imports.begin(), Imports.end(), Cmp),
                    Imports.end());

      RecordData ImportedModules;
      for (const auto &Import : Imports) {
        ImportedModules.push_back(Import.ID);
        // FIXME: If the module has macros imported then later has declarations
        // imported, this location won't be the right one as a location for the
        // declaration imports.
        AddSourceLocation(Import.M->MacroVisibilityLoc, ImportedModules);
      }

      Stream.EmitRecord(IMPORTED_MODULES, ImportedModules);
    }
  }

  WriteDeclReplacementsBlock();
  WriteRedeclarations();
  WriteMergedDecls();
  WriteObjCCategories();
  WriteLateParsedTemplates(SemaRef);
  if(!WritingModule)
    WriteOptimizePragmaOptions(SemaRef);

  // Some simple statistics
  Record.clear();
  Record.push_back(NumStatements);
  Record.push_back(NumMacros);
  Record.push_back(NumLexicalDeclContexts);
  Record.push_back(NumVisibleDeclContexts);
  Stream.EmitRecord(STATISTICS, Record);
  Stream.ExitBlock();
}

void ASTWriter::WriteDeclUpdatesBlocks(RecordDataImpl &OffsetsRecord) {
  if (DeclUpdates.empty())
    return;

  DeclUpdateMap LocalUpdates;
  LocalUpdates.swap(DeclUpdates);

  for (auto &DeclUpdate : LocalUpdates) {
    const Decl *D = DeclUpdate.first;
    if (isRewritten(D))
      continue; // The decl will be written completely,no need to store updates.

    bool HasUpdatedBody = false;
    RecordData Record;
    for (auto &Update : DeclUpdate.second) {
      DeclUpdateKind Kind = (DeclUpdateKind)Update.getKind();

      Record.push_back(Kind);
      switch (Kind) {
      case UPD_CXX_ADDED_IMPLICIT_MEMBER:
      case UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION:
      case UPD_CXX_ADDED_ANONYMOUS_NAMESPACE:
        assert(Update.getDecl() && "no decl to add?");
        Record.push_back(GetDeclRef(Update.getDecl()));
        break;

      case UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER:
        AddSourceLocation(Update.getLoc(), Record);
        break;

      case UPD_CXX_INSTANTIATED_FUNCTION_DEFINITION:
        // An updated body is emitted last, so that the reader doesn't need
        // to skip over the lazy body to reach statements for other records.
        Record.pop_back();
        HasUpdatedBody = true;
        break;

      case UPD_CXX_INSTANTIATED_CLASS_DEFINITION: {
        auto *RD = cast<CXXRecordDecl>(D);
        AddUpdatedDeclContext(RD->getPrimaryContext());
        AddCXXDefinitionData(RD, Record);
        Record.push_back(WriteDeclContextLexicalBlock(
            *Context, const_cast<CXXRecordDecl *>(RD)));

        // This state is sometimes updated by template instantiation, when we
        // switch from the specialization referring to the template declaration
        // to it referring to the template definition.
        if (auto *MSInfo = RD->getMemberSpecializationInfo()) {
          Record.push_back(MSInfo->getTemplateSpecializationKind());
          AddSourceLocation(MSInfo->getPointOfInstantiation(), Record);
        } else {
          auto *Spec = cast<ClassTemplateSpecializationDecl>(RD);
          Record.push_back(Spec->getTemplateSpecializationKind());
          AddSourceLocation(Spec->getPointOfInstantiation(), Record);

          // The instantiation might have been resolved to a partial
          // specialization. If so, record which one.
          auto From = Spec->getInstantiatedFrom();
          if (auto PartialSpec =
                From.dyn_cast<ClassTemplatePartialSpecializationDecl*>()) {
            Record.push_back(true);
            AddDeclRef(PartialSpec, Record);
            AddTemplateArgumentList(&Spec->getTemplateInstantiationArgs(),
                                    Record);
          } else {
            Record.push_back(false);
          }
        }
        Record.push_back(RD->getTagKind());
        AddSourceLocation(RD->getLocation(), Record);
        AddSourceLocation(RD->getLocStart(), Record);
        AddSourceLocation(RD->getRBraceLoc(), Record);

        // Instantiation may change attributes; write them all out afresh.
        Record.push_back(D->hasAttrs());
        if (Record.back())
          WriteAttributes(ArrayRef<const Attr*>(D->getAttrs().begin(),
                                                D->getAttrs().size()), Record);

        // FIXME: Ensure we don't get here for explicit instantiations.
        break;
      }

      case UPD_CXX_RESOLVED_EXCEPTION_SPEC:
        addExceptionSpec(
            *this,
            cast<FunctionDecl>(D)->getType()->castAs<FunctionProtoType>(),
            Record);
        break;

      case UPD_CXX_DEDUCED_RETURN_TYPE:
        Record.push_back(GetOrCreateTypeID(Update.getType()));
        break;

      case UPD_DECL_MARKED_USED:
        break;

      case UPD_MANGLING_NUMBER:
      case UPD_STATIC_LOCAL_NUMBER:
        Record.push_back(Update.getNumber());
        break;
      }
    }

    if (HasUpdatedBody) {
      const FunctionDecl *Def = cast<FunctionDecl>(D);
      Record.push_back(UPD_CXX_INSTANTIATED_FUNCTION_DEFINITION);
      Record.push_back(Def->isInlined());
      AddSourceLocation(Def->getInnerLocStart(), Record);
      AddFunctionDefinition(Def, Record);
    }

    OffsetsRecord.push_back(GetDeclRef(D));
    OffsetsRecord.push_back(Stream.GetCurrentBitNo());

    Stream.EmitRecord(DECL_UPDATES, Record);

    // Flush any statements that were written as part of this update record.
    FlushStmts();

    // Flush C++ base specifiers, if there are any.
    FlushCXXBaseSpecifiers();
  }
}

void ASTWriter::WriteDeclReplacementsBlock() {
  if (ReplacedDecls.empty())
    return;

  RecordData Record;
  for (SmallVectorImpl<ReplacedDeclInfo>::iterator
         I = ReplacedDecls.begin(), E = ReplacedDecls.end(); I != E; ++I) {
    Record.push_back(I->ID);
    Record.push_back(I->Offset);
    Record.push_back(I->Loc);
  }
  Stream.EmitRecord(DECL_REPLACEMENTS, Record);
}

void ASTWriter::AddSourceLocation(SourceLocation Loc, RecordDataImpl &Record) {
  Record.push_back(Loc.getRawEncoding());
}

void ASTWriter::AddSourceRange(SourceRange Range, RecordDataImpl &Record) {
  AddSourceLocation(Range.getBegin(), Record);
  AddSourceLocation(Range.getEnd(), Record);
}

void ASTWriter::AddAPInt(const llvm::APInt &Value, RecordDataImpl &Record) {
  Record.push_back(Value.getBitWidth());
  const uint64_t *Words = Value.getRawData();
  Record.append(Words, Words + Value.getNumWords());
}

void ASTWriter::AddAPSInt(const llvm::APSInt &Value, RecordDataImpl &Record) {
  Record.push_back(Value.isUnsigned());
  AddAPInt(Value, Record);
}

void ASTWriter::AddAPFloat(const llvm::APFloat &Value, RecordDataImpl &Record) {
  AddAPInt(Value.bitcastToAPInt(), Record);
}

void ASTWriter::AddIdentifierRef(const IdentifierInfo *II, RecordDataImpl &Record) {
  Record.push_back(getIdentifierRef(II));
}

IdentID ASTWriter::getIdentifierRef(const IdentifierInfo *II) {
  if (!II)
    return 0;

  IdentID &ID = IdentifierIDs[II];
  if (ID == 0)
    ID = NextIdentID++;
  return ID;
}

MacroID ASTWriter::getMacroRef(MacroInfo *MI, const IdentifierInfo *Name) {
  // Don't emit builtin macros like __LINE__ to the AST file unless they
  // have been redefined by the header (in which case they are not
  // isBuiltinMacro).
  if (!MI || MI->isBuiltinMacro())
    return 0;

  MacroID &ID = MacroIDs[MI];
  if (ID == 0) {
    ID = NextMacroID++;
    MacroInfoToEmitData Info = { Name, MI, ID };
    MacroInfosToEmit.push_back(Info);
  }
  return ID;
}

MacroID ASTWriter::getMacroID(MacroInfo *MI) {
  if (!MI || MI->isBuiltinMacro())
    return 0;
  
  assert(MacroIDs.find(MI) != MacroIDs.end() && "Macro not emitted!");
  return MacroIDs[MI];
}

uint64_t ASTWriter::getMacroDirectivesOffset(const IdentifierInfo *Name) {
  assert(IdentMacroDirectivesOffsetMap[Name] && "not set!");
  return IdentMacroDirectivesOffsetMap[Name];
}

void ASTWriter::AddSelectorRef(const Selector SelRef, RecordDataImpl &Record) {
  Record.push_back(getSelectorRef(SelRef));
}

SelectorID ASTWriter::getSelectorRef(Selector Sel) {
  if (Sel.getAsOpaquePtr() == nullptr) {
    return 0;
  }

  SelectorID SID = SelectorIDs[Sel];
  if (SID == 0 && Chain) {
    // This might trigger a ReadSelector callback, which will set the ID for
    // this selector.
    Chain->LoadSelector(Sel);
    SID = SelectorIDs[Sel];
  }
  if (SID == 0) {
    SID = NextSelectorID++;
    SelectorIDs[Sel] = SID;
  }
  return SID;
}

void ASTWriter::AddCXXTemporary(const CXXTemporary *Temp, RecordDataImpl &Record) {
  AddDeclRef(Temp->getDestructor(), Record);
}

void ASTWriter::AddCXXBaseSpecifiersRef(CXXBaseSpecifier const *Bases,
                                      CXXBaseSpecifier const *BasesEnd,
                                        RecordDataImpl &Record) {
  assert(Bases != BasesEnd && "Empty base-specifier sets are not recorded");
  CXXBaseSpecifiersToWrite.push_back(
                                QueuedCXXBaseSpecifiers(NextCXXBaseSpecifiersID,
                                                        Bases, BasesEnd));
  Record.push_back(NextCXXBaseSpecifiersID++);
}

void ASTWriter::AddTemplateArgumentLocInfo(TemplateArgument::ArgKind Kind,
                                           const TemplateArgumentLocInfo &Arg,
                                           RecordDataImpl &Record) {
  switch (Kind) {
  case TemplateArgument::Expression:
    AddStmt(Arg.getAsExpr());
    break;
  case TemplateArgument::Type:
    AddTypeSourceInfo(Arg.getAsTypeSourceInfo(), Record);
    break;
  case TemplateArgument::Template:
    AddNestedNameSpecifierLoc(Arg.getTemplateQualifierLoc(), Record);
    AddSourceLocation(Arg.getTemplateNameLoc(), Record);
    break;
  case TemplateArgument::TemplateExpansion:
    AddNestedNameSpecifierLoc(Arg.getTemplateQualifierLoc(), Record);
    AddSourceLocation(Arg.getTemplateNameLoc(), Record);
    AddSourceLocation(Arg.getTemplateEllipsisLoc(), Record);
    break;
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
  case TemplateArgument::Declaration:
  case TemplateArgument::NullPtr:
  case TemplateArgument::Pack:
    // FIXME: Is this right?
    break;
  }
}

void ASTWriter::AddTemplateArgumentLoc(const TemplateArgumentLoc &Arg,
                                       RecordDataImpl &Record) {
  AddTemplateArgument(Arg.getArgument(), Record);

  if (Arg.getArgument().getKind() == TemplateArgument::Expression) {
    bool InfoHasSameExpr
      = Arg.getArgument().getAsExpr() == Arg.getLocInfo().getAsExpr();
    Record.push_back(InfoHasSameExpr);
    if (InfoHasSameExpr)
      return; // Avoid storing the same expr twice.
  }
  AddTemplateArgumentLocInfo(Arg.getArgument().getKind(), Arg.getLocInfo(),
                             Record);
}

void ASTWriter::AddTypeSourceInfo(TypeSourceInfo *TInfo, 
                                  RecordDataImpl &Record) {
  if (!TInfo) {
    AddTypeRef(QualType(), Record);
    return;
  }

  AddTypeLoc(TInfo->getTypeLoc(), Record);
}

void ASTWriter::AddTypeLoc(TypeLoc TL, RecordDataImpl &Record) {
  AddTypeRef(TL.getType(), Record);

  TypeLocWriter TLW(*this, Record);
  for (; !TL.isNull(); TL = TL.getNextTypeLoc())
    TLW.Visit(TL);
}

void ASTWriter::AddTypeRef(QualType T, RecordDataImpl &Record) {
  Record.push_back(GetOrCreateTypeID(T));
}

TypeID ASTWriter::GetOrCreateTypeID( QualType T) {
  assert(Context);
  return MakeTypeID(*Context, T,
              std::bind1st(std::mem_fun(&ASTWriter::GetOrCreateTypeIdx), this));
}

TypeID ASTWriter::getTypeID(QualType T) const {
  assert(Context);
  return MakeTypeID(*Context, T,
              std::bind1st(std::mem_fun(&ASTWriter::getTypeIdx), this));
}

TypeIdx ASTWriter::GetOrCreateTypeIdx(QualType T) {
  if (T.isNull())
    return TypeIdx();
  assert(!T.getLocalFastQualifiers());

  TypeIdx &Idx = TypeIdxs[T];
  if (Idx.getIndex() == 0) {
    if (DoneWritingDeclsAndTypes) {
      assert(0 && "New type seen after serializing all the types to emit!");
      return TypeIdx();
    }

    // We haven't seen this type before. Assign it a new ID and put it
    // into the queue of types to emit.
    Idx = TypeIdx(NextTypeID++);
    DeclTypesToEmit.push(T);
  }
  return Idx;
}

TypeIdx ASTWriter::getTypeIdx(QualType T) const {
  if (T.isNull())
    return TypeIdx();
  assert(!T.getLocalFastQualifiers());

  TypeIdxMap::const_iterator I = TypeIdxs.find(T);
  assert(I != TypeIdxs.end() && "Type not emitted!");
  return I->second;
}

void ASTWriter::AddDeclRef(const Decl *D, RecordDataImpl &Record) {
  Record.push_back(GetDeclRef(D));
}

DeclID ASTWriter::GetDeclRef(const Decl *D) {
  assert(WritingAST && "Cannot request a declaration ID before AST writing");

  if (!D) {
    return 0;
  }
  
  // If D comes from an AST file, its declaration ID is already known and
  // fixed.
  if (D->isFromASTFile())
    return D->getGlobalID();
  
  assert(!(reinterpret_cast<uintptr_t>(D) & 0x01) && "Invalid decl pointer");
  DeclID &ID = DeclIDs[D];
  if (ID == 0) {
    if (DoneWritingDeclsAndTypes) {
      assert(0 && "New decl seen after serializing all the decls to emit!");
      return 0;
    }

    // We haven't seen this declaration before. Give it a new ID and
    // enqueue it in the list of declarations to emit.
    ID = NextDeclID++;
    DeclTypesToEmit.push(const_cast<Decl *>(D));
  }

  return ID;
}

DeclID ASTWriter::getDeclID(const Decl *D) {
  if (!D)
    return 0;

  // If D comes from an AST file, its declaration ID is already known and
  // fixed.
  if (D->isFromASTFile())
    return D->getGlobalID();

  assert(DeclIDs.find(D) != DeclIDs.end() && "Declaration not emitted!");
  return DeclIDs[D];
}

void ASTWriter::associateDeclWithFile(const Decl *D, DeclID ID) {
  assert(ID);
  assert(D);

  SourceLocation Loc = D->getLocation();
  if (Loc.isInvalid())
    return;

  // We only keep track of the file-level declarations of each file.
  if (!D->getLexicalDeclContext()->isFileContext())
    return;
  // FIXME: ParmVarDecls that are part of a function type of a parameter of
  // a function/objc method, should not have TU as lexical context.
  if (isa<ParmVarDecl>(D))
    return;

  SourceManager &SM = Context->getSourceManager();
  SourceLocation FileLoc = SM.getFileLoc(Loc);
  assert(SM.isLocalSourceLocation(FileLoc));
  FileID FID;
  unsigned Offset;
  std::tie(FID, Offset) = SM.getDecomposedLoc(FileLoc);
  if (FID.isInvalid())
    return;
  assert(SM.getSLocEntry(FID).isFile());

  DeclIDInFileInfo *&Info = FileDeclIDs[FID];
  if (!Info)
    Info = new DeclIDInFileInfo();

  std::pair<unsigned, serialization::DeclID> LocDecl(Offset, ID);
  LocDeclIDsTy &Decls = Info->DeclIDs;

  if (Decls.empty() || Decls.back().first <= Offset) {
    Decls.push_back(LocDecl);
    return;
  }

  LocDeclIDsTy::iterator I =
      std::upper_bound(Decls.begin(), Decls.end(), LocDecl, llvm::less_first());

  Decls.insert(I, LocDecl);
}

void ASTWriter::AddDeclarationName(DeclarationName Name, RecordDataImpl &Record) {
  // FIXME: Emit a stable enum for NameKind.  0 = Identifier etc.
  Record.push_back(Name.getNameKind());
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    AddIdentifierRef(Name.getAsIdentifierInfo(), Record);
    break;

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    AddSelectorRef(Name.getObjCSelector(), Record);
    break;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    AddTypeRef(Name.getCXXNameType(), Record);
    break;

  case DeclarationName::CXXOperatorName:
    Record.push_back(Name.getCXXOverloadedOperator());
    break;

  case DeclarationName::CXXLiteralOperatorName:
    AddIdentifierRef(Name.getCXXLiteralIdentifier(), Record);
    break;

  case DeclarationName::CXXUsingDirective:
    // No extra data to emit
    break;
  }
}

void ASTWriter::AddDeclarationNameLoc(const DeclarationNameLoc &DNLoc,
                                     DeclarationName Name, RecordDataImpl &Record) {
  switch (Name.getNameKind()) {
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    AddTypeSourceInfo(DNLoc.NamedType.TInfo, Record);
    break;

  case DeclarationName::CXXOperatorName:
    AddSourceLocation(
       SourceLocation::getFromRawEncoding(DNLoc.CXXOperatorName.BeginOpNameLoc),
       Record);
    AddSourceLocation(
        SourceLocation::getFromRawEncoding(DNLoc.CXXOperatorName.EndOpNameLoc),
        Record);
    break;

  case DeclarationName::CXXLiteralOperatorName:
    AddSourceLocation(
     SourceLocation::getFromRawEncoding(DNLoc.CXXLiteralOperatorName.OpNameLoc),
     Record);
    break;

  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXUsingDirective:
    break;
  }
}

void ASTWriter::AddDeclarationNameInfo(const DeclarationNameInfo &NameInfo,
                                       RecordDataImpl &Record) {
  AddDeclarationName(NameInfo.getName(), Record);
  AddSourceLocation(NameInfo.getLoc(), Record);
  AddDeclarationNameLoc(NameInfo.getInfo(), NameInfo.getName(), Record);
}

void ASTWriter::AddQualifierInfo(const QualifierInfo &Info,
                                 RecordDataImpl &Record) {
  AddNestedNameSpecifierLoc(Info.QualifierLoc, Record);
  Record.push_back(Info.NumTemplParamLists);
  for (unsigned i=0, e=Info.NumTemplParamLists; i != e; ++i)
    AddTemplateParameterList(Info.TemplParamLists[i], Record);
}

void ASTWriter::AddNestedNameSpecifier(NestedNameSpecifier *NNS,
                                       RecordDataImpl &Record) {
  // Nested name specifiers usually aren't too long. I think that 8 would
  // typically accommodate the vast majority.
  SmallVector<NestedNameSpecifier *, 8> NestedNames;

  // Push each of the NNS's onto a stack for serialization in reverse order.
  while (NNS) {
    NestedNames.push_back(NNS);
    NNS = NNS->getPrefix();
  }

  Record.push_back(NestedNames.size());
  while(!NestedNames.empty()) {
    NNS = NestedNames.pop_back_val();
    NestedNameSpecifier::SpecifierKind Kind = NNS->getKind();
    Record.push_back(Kind);
    switch (Kind) {
    case NestedNameSpecifier::Identifier:
      AddIdentifierRef(NNS->getAsIdentifier(), Record);
      break;

    case NestedNameSpecifier::Namespace:
      AddDeclRef(NNS->getAsNamespace(), Record);
      break;

    case NestedNameSpecifier::NamespaceAlias:
      AddDeclRef(NNS->getAsNamespaceAlias(), Record);
      break;

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
      AddTypeRef(QualType(NNS->getAsType(), 0), Record);
      Record.push_back(Kind == NestedNameSpecifier::TypeSpecWithTemplate);
      break;

    case NestedNameSpecifier::Global:
      // Don't need to write an associated value.
      break;
    }
  }
}

void ASTWriter::AddNestedNameSpecifierLoc(NestedNameSpecifierLoc NNS,
                                          RecordDataImpl &Record) {
  // Nested name specifiers usually aren't too long. I think that 8 would
  // typically accommodate the vast majority.
  SmallVector<NestedNameSpecifierLoc , 8> NestedNames;

  // Push each of the nested-name-specifiers's onto a stack for
  // serialization in reverse order.
  while (NNS) {
    NestedNames.push_back(NNS);
    NNS = NNS.getPrefix();
  }

  Record.push_back(NestedNames.size());
  while(!NestedNames.empty()) {
    NNS = NestedNames.pop_back_val();
    NestedNameSpecifier::SpecifierKind Kind
      = NNS.getNestedNameSpecifier()->getKind();
    Record.push_back(Kind);
    switch (Kind) {
    case NestedNameSpecifier::Identifier:
      AddIdentifierRef(NNS.getNestedNameSpecifier()->getAsIdentifier(), Record);
      AddSourceRange(NNS.getLocalSourceRange(), Record);
      break;

    case NestedNameSpecifier::Namespace:
      AddDeclRef(NNS.getNestedNameSpecifier()->getAsNamespace(), Record);
      AddSourceRange(NNS.getLocalSourceRange(), Record);
      break;

    case NestedNameSpecifier::NamespaceAlias:
      AddDeclRef(NNS.getNestedNameSpecifier()->getAsNamespaceAlias(), Record);
      AddSourceRange(NNS.getLocalSourceRange(), Record);
      break;

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
      Record.push_back(Kind == NestedNameSpecifier::TypeSpecWithTemplate);
      AddTypeLoc(NNS.getTypeLoc(), Record);
      AddSourceLocation(NNS.getLocalSourceRange().getEnd(), Record);
      break;

    case NestedNameSpecifier::Global:
      AddSourceLocation(NNS.getLocalSourceRange().getEnd(), Record);
      break;
    }
  }
}

void ASTWriter::AddTemplateName(TemplateName Name, RecordDataImpl &Record) {
  TemplateName::NameKind Kind = Name.getKind();
  Record.push_back(Kind);
  switch (Kind) {
  case TemplateName::Template:
    AddDeclRef(Name.getAsTemplateDecl(), Record);
    break;

  case TemplateName::OverloadedTemplate: {
    OverloadedTemplateStorage *OvT = Name.getAsOverloadedTemplate();
    Record.push_back(OvT->size());
    for (OverloadedTemplateStorage::iterator I = OvT->begin(), E = OvT->end();
           I != E; ++I)
      AddDeclRef(*I, Record);
    break;
  }

  case TemplateName::QualifiedTemplate: {
    QualifiedTemplateName *QualT = Name.getAsQualifiedTemplateName();
    AddNestedNameSpecifier(QualT->getQualifier(), Record);
    Record.push_back(QualT->hasTemplateKeyword());
    AddDeclRef(QualT->getTemplateDecl(), Record);
    break;
  }

  case TemplateName::DependentTemplate: {
    DependentTemplateName *DepT = Name.getAsDependentTemplateName();
    AddNestedNameSpecifier(DepT->getQualifier(), Record);
    Record.push_back(DepT->isIdentifier());
    if (DepT->isIdentifier())
      AddIdentifierRef(DepT->getIdentifier(), Record);
    else
      Record.push_back(DepT->getOperator());
    break;
  }

  case TemplateName::SubstTemplateTemplateParm: {
    SubstTemplateTemplateParmStorage *subst
      = Name.getAsSubstTemplateTemplateParm();
    AddDeclRef(subst->getParameter(), Record);
    AddTemplateName(subst->getReplacement(), Record);
    break;
  }
      
  case TemplateName::SubstTemplateTemplateParmPack: {
    SubstTemplateTemplateParmPackStorage *SubstPack
      = Name.getAsSubstTemplateTemplateParmPack();
    AddDeclRef(SubstPack->getParameterPack(), Record);
    AddTemplateArgument(SubstPack->getArgumentPack(), Record);
    break;
  }
  }
}

void ASTWriter::AddTemplateArgument(const TemplateArgument &Arg,
                                    RecordDataImpl &Record) {
  Record.push_back(Arg.getKind());
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    break;
  case TemplateArgument::Type:
    AddTypeRef(Arg.getAsType(), Record);
    break;
  case TemplateArgument::Declaration:
    AddDeclRef(Arg.getAsDecl(), Record);
    Record.push_back(Arg.isDeclForReferenceParam());
    break;
  case TemplateArgument::NullPtr:
    AddTypeRef(Arg.getNullPtrType(), Record);
    break;
  case TemplateArgument::Integral:
    AddAPSInt(Arg.getAsIntegral(), Record);
    AddTypeRef(Arg.getIntegralType(), Record);
    break;
  case TemplateArgument::Template:
    AddTemplateName(Arg.getAsTemplateOrTemplatePattern(), Record);
    break;
  case TemplateArgument::TemplateExpansion:
    AddTemplateName(Arg.getAsTemplateOrTemplatePattern(), Record);
    if (Optional<unsigned> NumExpansions = Arg.getNumTemplateExpansions())
      Record.push_back(*NumExpansions + 1);
    else
      Record.push_back(0);
    break;
  case TemplateArgument::Expression:
    AddStmt(Arg.getAsExpr());
    break;
  case TemplateArgument::Pack:
    Record.push_back(Arg.pack_size());
    for (const auto &P : Arg.pack_elements())
      AddTemplateArgument(P, Record);
    break;
  }
}

void
ASTWriter::AddTemplateParameterList(const TemplateParameterList *TemplateParams,
                                    RecordDataImpl &Record) {
  assert(TemplateParams && "No TemplateParams!");
  AddSourceLocation(TemplateParams->getTemplateLoc(), Record);
  AddSourceLocation(TemplateParams->getLAngleLoc(), Record);
  AddSourceLocation(TemplateParams->getRAngleLoc(), Record);
  Record.push_back(TemplateParams->size());
  for (TemplateParameterList::const_iterator
         P = TemplateParams->begin(), PEnd = TemplateParams->end();
         P != PEnd; ++P)
    AddDeclRef(*P, Record);
}

/// \brief Emit a template argument list.
void
ASTWriter::AddTemplateArgumentList(const TemplateArgumentList *TemplateArgs,
                                   RecordDataImpl &Record) {
  assert(TemplateArgs && "No TemplateArgs!");
  Record.push_back(TemplateArgs->size());
  for (int i=0, e = TemplateArgs->size(); i != e; ++i)
    AddTemplateArgument(TemplateArgs->get(i), Record);
}

void
ASTWriter::AddASTTemplateArgumentListInfo
(const ASTTemplateArgumentListInfo *ASTTemplArgList, RecordDataImpl &Record) {
  assert(ASTTemplArgList && "No ASTTemplArgList!");
  AddSourceLocation(ASTTemplArgList->LAngleLoc, Record);
  AddSourceLocation(ASTTemplArgList->RAngleLoc, Record);
  Record.push_back(ASTTemplArgList->NumTemplateArgs);
  const TemplateArgumentLoc *TemplArgs = ASTTemplArgList->getTemplateArgs();
  for (int i=0, e = ASTTemplArgList->NumTemplateArgs; i != e; ++i)
    AddTemplateArgumentLoc(TemplArgs[i], Record);
}

void
ASTWriter::AddUnresolvedSet(const ASTUnresolvedSet &Set, RecordDataImpl &Record) {
  Record.push_back(Set.size());
  for (ASTUnresolvedSet::const_iterator
         I = Set.begin(), E = Set.end(); I != E; ++I) {
    AddDeclRef(I.getDecl(), Record);
    Record.push_back(I.getAccess());
  }
}

void ASTWriter::AddCXXBaseSpecifier(const CXXBaseSpecifier &Base,
                                    RecordDataImpl &Record) {
  Record.push_back(Base.isVirtual());
  Record.push_back(Base.isBaseOfClass());
  Record.push_back(Base.getAccessSpecifierAsWritten());
  Record.push_back(Base.getInheritConstructors());
  AddTypeSourceInfo(Base.getTypeSourceInfo(), Record);
  AddSourceRange(Base.getSourceRange(), Record);
  AddSourceLocation(Base.isPackExpansion()? Base.getEllipsisLoc() 
                                          : SourceLocation(),
                    Record);
}

void ASTWriter::FlushCXXBaseSpecifiers() {
  RecordData Record;
  for (unsigned I = 0, N = CXXBaseSpecifiersToWrite.size(); I != N; ++I) {
    Record.clear();
    
    // Record the offset of this base-specifier set.
    unsigned Index = CXXBaseSpecifiersToWrite[I].ID - 1;
    if (Index == CXXBaseSpecifiersOffsets.size())
      CXXBaseSpecifiersOffsets.push_back(Stream.GetCurrentBitNo());
    else {
      if (Index > CXXBaseSpecifiersOffsets.size())
        CXXBaseSpecifiersOffsets.resize(Index + 1);
      CXXBaseSpecifiersOffsets[Index] = Stream.GetCurrentBitNo();
    }

    const CXXBaseSpecifier *B = CXXBaseSpecifiersToWrite[I].Bases,
                        *BEnd = CXXBaseSpecifiersToWrite[I].BasesEnd;
    Record.push_back(BEnd - B);
    for (; B != BEnd; ++B)
      AddCXXBaseSpecifier(*B, Record);
    Stream.EmitRecord(serialization::DECL_CXX_BASE_SPECIFIERS, Record);
    
    // Flush any expressions that were written as part of the base specifiers.
    FlushStmts();
  }

  CXXBaseSpecifiersToWrite.clear();
}

void ASTWriter::AddCXXCtorInitializers(
                             const CXXCtorInitializer * const *CtorInitializers,
                             unsigned NumCtorInitializers,
                             RecordDataImpl &Record) {
  Record.push_back(NumCtorInitializers);
  for (unsigned i=0; i != NumCtorInitializers; ++i) {
    const CXXCtorInitializer *Init = CtorInitializers[i];

    if (Init->isBaseInitializer()) {
      Record.push_back(CTOR_INITIALIZER_BASE);
      AddTypeSourceInfo(Init->getTypeSourceInfo(), Record);
      Record.push_back(Init->isBaseVirtual());
    } else if (Init->isDelegatingInitializer()) {
      Record.push_back(CTOR_INITIALIZER_DELEGATING);
      AddTypeSourceInfo(Init->getTypeSourceInfo(), Record);
    } else if (Init->isMemberInitializer()){
      Record.push_back(CTOR_INITIALIZER_MEMBER);
      AddDeclRef(Init->getMember(), Record);
    } else {
      Record.push_back(CTOR_INITIALIZER_INDIRECT_MEMBER);
      AddDeclRef(Init->getIndirectMember(), Record);
    }

    AddSourceLocation(Init->getMemberLocation(), Record);
    AddStmt(Init->getInit());
    AddSourceLocation(Init->getLParenLoc(), Record);
    AddSourceLocation(Init->getRParenLoc(), Record);
    Record.push_back(Init->isWritten());
    if (Init->isWritten()) {
      Record.push_back(Init->getSourceOrder());
    } else {
      Record.push_back(Init->getNumArrayIndices());
      for (unsigned i=0, e=Init->getNumArrayIndices(); i != e; ++i)
        AddDeclRef(Init->getArrayIndex(i), Record);
    }
  }
}

void ASTWriter::AddCXXDefinitionData(const CXXRecordDecl *D, RecordDataImpl &Record) {
  auto &Data = D->data();
  Record.push_back(Data.IsLambda);
  Record.push_back(Data.UserDeclaredConstructor);
  Record.push_back(Data.UserDeclaredSpecialMembers);
  Record.push_back(Data.Aggregate);
  Record.push_back(Data.PlainOldData);
  Record.push_back(Data.Empty);
  Record.push_back(Data.Polymorphic);
  Record.push_back(Data.Abstract);
  Record.push_back(Data.IsStandardLayout);
  Record.push_back(Data.HasNoNonEmptyBases);
  Record.push_back(Data.HasPrivateFields);
  Record.push_back(Data.HasProtectedFields);
  Record.push_back(Data.HasPublicFields);
  Record.push_back(Data.HasMutableFields);
  Record.push_back(Data.HasVariantMembers);
  Record.push_back(Data.HasOnlyCMembers);
  Record.push_back(Data.HasInClassInitializer);
  Record.push_back(Data.HasUninitializedReferenceMember);
  Record.push_back(Data.NeedOverloadResolutionForMoveConstructor);
  Record.push_back(Data.NeedOverloadResolutionForMoveAssignment);
  Record.push_back(Data.NeedOverloadResolutionForDestructor);
  Record.push_back(Data.DefaultedMoveConstructorIsDeleted);
  Record.push_back(Data.DefaultedMoveAssignmentIsDeleted);
  Record.push_back(Data.DefaultedDestructorIsDeleted);
  Record.push_back(Data.HasTrivialSpecialMembers);
  Record.push_back(Data.DeclaredNonTrivialSpecialMembers);
  Record.push_back(Data.HasIrrelevantDestructor);
  Record.push_back(Data.HasConstexprNonCopyMoveConstructor);
  Record.push_back(Data.DefaultedDefaultConstructorIsConstexpr);
  Record.push_back(Data.HasConstexprDefaultConstructor);
  Record.push_back(Data.HasNonLiteralTypeFieldsOrBases);
  Record.push_back(Data.ComputedVisibleConversions);
  Record.push_back(Data.UserProvidedDefaultConstructor);
  Record.push_back(Data.DeclaredSpecialMembers);
  Record.push_back(Data.ImplicitCopyConstructorHasConstParam);
  Record.push_back(Data.ImplicitCopyAssignmentHasConstParam);
  Record.push_back(Data.HasDeclaredCopyConstructorWithConstParam);
  Record.push_back(Data.HasDeclaredCopyAssignmentWithConstParam);
  // IsLambda bit is already saved.

  Record.push_back(Data.NumBases);
  if (Data.NumBases > 0)
    AddCXXBaseSpecifiersRef(Data.getBases(), Data.getBases() + Data.NumBases, 
                            Record);
  
  // FIXME: Make VBases lazily computed when needed to avoid storing them.
  Record.push_back(Data.NumVBases);
  if (Data.NumVBases > 0)
    AddCXXBaseSpecifiersRef(Data.getVBases(), Data.getVBases() + Data.NumVBases, 
                            Record);

  AddUnresolvedSet(Data.Conversions.get(*Context), Record);
  AddUnresolvedSet(Data.VisibleConversions.get(*Context), Record);
  // Data.Definition is the owning decl, no need to write it. 
  AddDeclRef(D->getFirstFriend(), Record);
  
  // Add lambda-specific data.
  if (Data.IsLambda) {
    auto &Lambda = D->getLambdaData();
    Record.push_back(Lambda.Dependent);
    Record.push_back(Lambda.IsGenericLambda);
    Record.push_back(Lambda.CaptureDefault);
    Record.push_back(Lambda.NumCaptures);
    Record.push_back(Lambda.NumExplicitCaptures);
    Record.push_back(Lambda.ManglingNumber);
    AddDeclRef(Lambda.ContextDecl, Record);
    AddTypeSourceInfo(Lambda.MethodTyInfo, Record);
    for (unsigned I = 0, N = Lambda.NumCaptures; I != N; ++I) {
      const LambdaCapture &Capture = Lambda.Captures[I];
      AddSourceLocation(Capture.getLocation(), Record);
      Record.push_back(Capture.isImplicit());
      Record.push_back(Capture.getCaptureKind());
      switch (Capture.getCaptureKind()) {
      case LCK_This:
        break;
      case LCK_ByCopy:
      case LCK_ByRef:
        VarDecl *Var =
            Capture.capturesVariable() ? Capture.getCapturedVar() : nullptr;
        AddDeclRef(Var, Record);
        AddSourceLocation(Capture.isPackExpansion() ? Capture.getEllipsisLoc()
                                                    : SourceLocation(),
                          Record);
        break;
      }
    }
  }
}

void ASTWriter::ReaderInitialized(ASTReader *Reader) {
  assert(Reader && "Cannot remove chain");
  assert((!Chain || Chain == Reader) && "Cannot replace chain");
  assert(FirstDeclID == NextDeclID &&
         FirstTypeID == NextTypeID &&
         FirstIdentID == NextIdentID &&
         FirstMacroID == NextMacroID &&
         FirstSubmoduleID == NextSubmoduleID &&
         FirstSelectorID == NextSelectorID &&
         "Setting chain after writing has started.");

  Chain = Reader;

  FirstDeclID = NUM_PREDEF_DECL_IDS + Chain->getTotalNumDecls();
  FirstTypeID = NUM_PREDEF_TYPE_IDS + Chain->getTotalNumTypes();
  FirstIdentID = NUM_PREDEF_IDENT_IDS + Chain->getTotalNumIdentifiers();
  FirstMacroID = NUM_PREDEF_MACRO_IDS + Chain->getTotalNumMacros();
  FirstSubmoduleID = NUM_PREDEF_SUBMODULE_IDS + Chain->getTotalNumSubmodules();
  FirstSelectorID = NUM_PREDEF_SELECTOR_IDS + Chain->getTotalNumSelectors();
  NextDeclID = FirstDeclID;
  NextTypeID = FirstTypeID;
  NextIdentID = FirstIdentID;
  NextMacroID = FirstMacroID;
  NextSelectorID = FirstSelectorID;
  NextSubmoduleID = FirstSubmoduleID;
}

void ASTWriter::IdentifierRead(IdentID ID, IdentifierInfo *II) {
  // Always keep the highest ID. See \p TypeRead() for more information.
  IdentID &StoredID = IdentifierIDs[II];
  if (ID > StoredID)
    StoredID = ID;
}

void ASTWriter::MacroRead(serialization::MacroID ID, MacroInfo *MI) {
  // Always keep the highest ID. See \p TypeRead() for more information.
  MacroID &StoredID = MacroIDs[MI];
  if (ID > StoredID)
    StoredID = ID;
}

void ASTWriter::TypeRead(TypeIdx Idx, QualType T) {
  // Always take the highest-numbered type index. This copes with an interesting
  // case for chained AST writing where we schedule writing the type and then,
  // later, deserialize the type from another AST. In this case, we want to
  // keep the higher-numbered entry so that we can properly write it out to
  // the AST file.
  TypeIdx &StoredIdx = TypeIdxs[T];
  if (Idx.getIndex() >= StoredIdx.getIndex())
    StoredIdx = Idx;
}

void ASTWriter::SelectorRead(SelectorID ID, Selector S) {
  // Always keep the highest ID. See \p TypeRead() for more information.
  SelectorID &StoredID = SelectorIDs[S];
  if (ID > StoredID)
    StoredID = ID;
}

void ASTWriter::MacroDefinitionRead(serialization::PreprocessedEntityID ID,
                                    MacroDefinition *MD) {
  assert(MacroDefinitions.find(MD) == MacroDefinitions.end());
  MacroDefinitions[MD] = ID;
}

void ASTWriter::ModuleRead(serialization::SubmoduleID ID, Module *Mod) {
  assert(SubmoduleIDs.find(Mod) == SubmoduleIDs.end());
  SubmoduleIDs[Mod] = ID;
}

void ASTWriter::CompletedTagDefinition(const TagDecl *D) {
  assert(D->isCompleteDefinition());
  assert(!WritingAST && "Already writing the AST!");
  if (const CXXRecordDecl *RD = dyn_cast<CXXRecordDecl>(D)) {
    // We are interested when a PCH decl is modified.
    if (RD->isFromASTFile()) {
      // A forward reference was mutated into a definition. Rewrite it.
      // FIXME: This happens during template instantiation, should we
      // have created a new definition decl instead ?
      assert(isTemplateInstantiation(RD->getTemplateSpecializationKind()) &&
             "completed a tag from another module but not by instantiation?");
      DeclUpdates[RD].push_back(
          DeclUpdate(UPD_CXX_INSTANTIATED_CLASS_DEFINITION));
    }
  }
}

void ASTWriter::AddedVisibleDecl(const DeclContext *DC, const Decl *D) {
  assert(!WritingAST && "Already writing the AST!");

  // TU and namespaces are handled elsewhere.
  if (isa<TranslationUnitDecl>(DC) || isa<NamespaceDecl>(DC))
    return;

  if (!(!D->isFromASTFile() && cast<Decl>(DC)->isFromASTFile()))
    return; // Not a source decl added to a DeclContext from PCH.

  assert(!getDefinitiveDeclContext(DC) && "DeclContext not definitive!");
  AddUpdatedDeclContext(DC);
  UpdatingVisibleDecls.push_back(D);
}

void ASTWriter::AddedCXXImplicitMember(const CXXRecordDecl *RD, const Decl *D) {
  assert(!WritingAST && "Already writing the AST!");
  assert(D->isImplicit());
  if (!(!D->isFromASTFile() && RD->isFromASTFile()))
    return; // Not a source member added to a class from PCH.
  if (!isa<CXXMethodDecl>(D))
    return; // We are interested in lazily declared implicit methods.

  // A decl coming from PCH was modified.
  assert(RD->isCompleteDefinition());
  DeclUpdates[RD].push_back(DeclUpdate(UPD_CXX_ADDED_IMPLICIT_MEMBER, D));
}

void ASTWriter::AddedCXXTemplateSpecialization(const ClassTemplateDecl *TD,
                                     const ClassTemplateSpecializationDecl *D) {
  // The specializations set is kept in the canonical template.
  assert(!WritingAST && "Already writing the AST!");
  TD = TD->getCanonicalDecl();
  if (!(!D->isFromASTFile() && TD->isFromASTFile()))
    return; // Not a source specialization added to a template from PCH.

  DeclUpdates[TD].push_back(DeclUpdate(UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION,
                                       D));
}

void ASTWriter::AddedCXXTemplateSpecialization(
    const VarTemplateDecl *TD, const VarTemplateSpecializationDecl *D) {
  // The specializations set is kept in the canonical template.
  assert(!WritingAST && "Already writing the AST!");
  TD = TD->getCanonicalDecl();
  if (!(!D->isFromASTFile() && TD->isFromASTFile()))
    return; // Not a source specialization added to a template from PCH.

  DeclUpdates[TD].push_back(DeclUpdate(UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION,
                                       D));
}

void ASTWriter::AddedCXXTemplateSpecialization(const FunctionTemplateDecl *TD,
                                               const FunctionDecl *D) {
  // The specializations set is kept in the canonical template.
  assert(!WritingAST && "Already writing the AST!");
  TD = TD->getCanonicalDecl();
  if (!(!D->isFromASTFile() && TD->isFromASTFile()))
    return; // Not a source specialization added to a template from PCH.

  DeclUpdates[TD].push_back(DeclUpdate(UPD_CXX_ADDED_TEMPLATE_SPECIALIZATION,
                                       D));
}

void ASTWriter::ResolvedExceptionSpec(const FunctionDecl *FD) {
  assert(!WritingAST && "Already writing the AST!");
  FD = FD->getCanonicalDecl();
  if (!FD->isFromASTFile())
    return; // Not a function declared in PCH and defined outside.

  DeclUpdates[FD].push_back(UPD_CXX_RESOLVED_EXCEPTION_SPEC);
}

void ASTWriter::DeducedReturnType(const FunctionDecl *FD, QualType ReturnType) {
  assert(!WritingAST && "Already writing the AST!");
  FD = FD->getCanonicalDecl();
  if (!FD->isFromASTFile())
    return; // Not a function declared in PCH and defined outside.

  DeclUpdates[FD].push_back(DeclUpdate(UPD_CXX_DEDUCED_RETURN_TYPE, ReturnType));
}

void ASTWriter::CompletedImplicitDefinition(const FunctionDecl *D) {
  assert(!WritingAST && "Already writing the AST!");
  if (!D->isFromASTFile())
    return; // Declaration not imported from PCH.

  // Implicit decl from a PCH was defined.
  // FIXME: Should implicit definition be a separate FunctionDecl?
  RewriteDecl(D);
}

void ASTWriter::FunctionDefinitionInstantiated(const FunctionDecl *D) {
  assert(!WritingAST && "Already writing the AST!");
  if (!D->isFromASTFile())
    return;

  // Since the actual instantiation is delayed, this really means that we need
  // to update the instantiation location.
  DeclUpdates[D].push_back(
      DeclUpdate(UPD_CXX_INSTANTIATED_FUNCTION_DEFINITION));
}

void ASTWriter::StaticDataMemberInstantiated(const VarDecl *D) {
  assert(!WritingAST && "Already writing the AST!");
  if (!D->isFromASTFile())
    return;

  // Since the actual instantiation is delayed, this really means that we need
  // to update the instantiation location.
  DeclUpdates[D].push_back(
      DeclUpdate(UPD_CXX_INSTANTIATED_STATIC_DATA_MEMBER,
       D->getMemberSpecializationInfo()->getPointOfInstantiation()));
}

void ASTWriter::AddedObjCCategoryToInterface(const ObjCCategoryDecl *CatD,
                                             const ObjCInterfaceDecl *IFD) {
  assert(!WritingAST && "Already writing the AST!");
  if (!IFD->isFromASTFile())
    return; // Declaration not imported from PCH.
  
  assert(IFD->getDefinition() && "Category on a class without a definition?");
  ObjCClassesWithCategories.insert(
    const_cast<ObjCInterfaceDecl *>(IFD->getDefinition()));
}


void ASTWriter::AddedObjCPropertyInClassExtension(const ObjCPropertyDecl *Prop,
                                          const ObjCPropertyDecl *OrigProp,
                                          const ObjCCategoryDecl *ClassExt) {
  const ObjCInterfaceDecl *D = ClassExt->getClassInterface();
  if (!D)
    return;

  assert(!WritingAST && "Already writing the AST!");
  if (!D->isFromASTFile())
    return; // Declaration not imported from PCH.

  RewriteDecl(D);
}

void ASTWriter::DeclarationMarkedUsed(const Decl *D) {
  assert(!WritingAST && "Already writing the AST!");
  if (!D->isFromASTFile())
    return;

  DeclUpdates[D].push_back(DeclUpdate(UPD_DECL_MARKED_USED));
}
