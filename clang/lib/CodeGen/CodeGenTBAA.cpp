//===-- CodeGenTBAA.cpp - TBAA information for LLVM CodeGen ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that manages TBAA information and defines the TBAA policy
// for the optimizer to use. Relevant standards text includes:
//
//   C99 6.5p7
//   C++ [basic.lval] (p10 in n3126, p15 in some earlier versions)
//
//===----------------------------------------------------------------------===//

#include "CodeGenTBAA.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/CodeGenOptions.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
using namespace clang;
using namespace CodeGen;

CodeGenTBAA::CodeGenTBAA(ASTContext &Ctx, llvm::Module &M,
                         const CodeGenOptions &CGO,
                         const LangOptions &Features, MangleContext &MContext)
  : Context(Ctx), Module(M), CodeGenOpts(CGO),
    Features(Features), MContext(MContext), MDHelper(M.getContext()),
    Root(nullptr), Char(nullptr)
{}

CodeGenTBAA::~CodeGenTBAA() {
}

llvm::MDNode *CodeGenTBAA::getRoot() {
  // Define the root of the tree. This identifies the tree, so that
  // if our LLVM IR is linked with LLVM IR from a different front-end
  // (or a different version of this front-end), their TBAA trees will
  // remain distinct, and the optimizer will treat them conservatively.
  if (!Root) {
    if (Features.CPlusPlus)
      Root = MDHelper.createTBAARoot("Simple C++ TBAA");
    else
      Root = MDHelper.createTBAARoot("Simple C/C++ TBAA");
  }

  return Root;
}

llvm::MDNode *CodeGenTBAA::createScalarTypeNode(StringRef Name,
                                                llvm::MDNode *Parent,
                                                uint64_t Size) {
  if (CodeGenOpts.NewStructPathTBAA) {
    llvm::Metadata *Id = MDHelper.createString(Name);
    return MDHelper.createTBAATypeNode(Parent, Size, Id);
  }
  return MDHelper.createTBAAScalarTypeNode(Name, Parent);
}

llvm::MDNode *CodeGenTBAA::getChar() {
  // Define the root of the tree for user-accessible memory. C and C++
  // give special powers to char and certain similar types. However,
  // these special powers only cover user-accessible memory, and doesn't
  // include things like vtables.
  if (!Char)
    Char = createScalarTypeNode("omnipotent char", getRoot(), /* Size= */ 1);

  return Char;
}

static bool TypeHasMayAlias(QualType QTy) {
  // Tagged types have declarations, and therefore may have attributes.
  if (auto *TD = QTy->getAsTagDecl())
    if (TD->hasAttr<MayAliasAttr>())
      return true;

  // Also look for may_alias as a declaration attribute on a typedef.
  // FIXME: We should follow GCC and model may_alias as a type attribute
  // rather than as a declaration attribute.
  while (auto *TT = QTy->getAs<TypedefType>()) {
    if (TT->getDecl()->hasAttr<MayAliasAttr>())
      return true;
    QTy = TT->desugar();
  }
  return false;
}

/// Check if the given type is a valid base type to be used in access tags.
static bool isValidBaseType(QualType QTy, const CodeGenOptions &CodeGenOpts) {
  if (QTy->isReferenceType())
    return false;
  if (const RecordType *TTy = QTy->getAs<RecordType>()) {
    const RecordDecl *RD = TTy->getDecl()->getDefinition();
    // Incomplete types are not valid base access types.
    if (!RD)
      return false;
    if (RD->hasFlexibleArrayMember())
      return false;
    // RD can be struct, union, class, interface or enum.
    if (RD->isStruct() || RD->isClass() ||
        (RD->isUnion() && CodeGenOpts.UnionTBAA))
      return true;
  }
  return false;
}

// Appends unique tag for compatible pointee types.
void CodeGenTBAA::appendPointeeName(llvm::raw_ostream &OS, const Type *Ty) {
  // Although type compatibilty in C standard requires cv-qualification
  // match and exact type match, here more relaxed rules are applied.
  //
  // For built-in types consider them 'compatible' if their respective
  // TBAA metadata tag is same(e.g. that makes 'int' and 'unsigned'
  // compatible).
  if (isa<BuiltinType>(Ty)) {
    llvm::MDNode *ScalarMD = getTypeInfoHelper(Ty);
    auto &Op = ScalarMD->getOperand(CodeGenOpts.NewStructPathTBAA ? 2 : 0);
    assert(isa<llvm::MDString>(Op) && "Expected MDString operand");
    OS << cast<llvm::MDString>(Op)->getString().str();
  }

  // Non-builtin types are considered compatible if their tag matches.
  OS << Ty->getUnqualifiedDesugaredType()
      ->getCanonicalTypeInternal()
      .getAsString();
}

/// Return an LLVM TBAA metadata node appropriate for an access through
/// an l-value of the given type.  Type-based alias analysis takes advantage
/// of the following rules from the language standards:
///
/// C 6.5p7:
///   An object shall have its stored value accessed only by an lvalue
///   expression that has one of the following types:
///   - a type compatible with the effective type of the object,
///   - a qualified version of a type compatible with the effective
///     type of the object,
///   - a type that is the signed or unsigned type corresponding
///     to the effective type of the object,
///   - a type that is the signed or unsigned type corresponding
///     to a qualified version of the effective type of the object,
///   - an aggregate or union type that includes one of the
///     aforementioned types among its members (including,
///     recursively, a member of a subaggregate or contained union), or
///   - a character type.
///
/// C++ [basic.lval]p11:
///   If a program attempts to access the stored value of an object
///   through a glvalue whose type is not similar to one of the following
///   types the behavior is undefined:
///   - the dynamic type of the object,
///   - a type that is the signed or unsigned type corresponding
///     to the dynamic type of the object, or
///   - a char, unsigned char, or std::byte type.
///
/// The C and C++ rules about effective/dynamic type are broadly similar
/// and permit memory to be reused with a different type.  C does not have
/// an explicit operation to change the effective type of memory; any store
/// can do it.  While C++ arguably does have such an operation (the standard
/// global `operator new(void*, size_t)`), in practice it is important to
/// be just as permissive as C.  We therefore treat all stores as being able to
/// change the effective type of memory, regardless of language mode.  That is,
/// loads have both a precondition and a postcondition on the effective
/// type of the memory, but stores only have a postcondition.  This imposes
/// an inherent limitation that TBAA can only be used to reorder loads
/// before stores.  This is quite restrictive, but we don't have much of a
/// choice.  In practice, hoisting loads is the most important optimization
/// for alias analysis to enable anyway.
///
/// Therefore, given a load (and its precondition) and an earlier store
/// (and its postcondition), the question posed to TBAA is whether there
/// exists a type that is consistent with both accesses.  If there isn't,
/// it's fine to hoist the load because either the memory is non-overlapping
/// or the precondition on the load is wrong (which would be UB).
///
/// LLVM TBAA says that two accesses with TBAA metadata nodes may alias if:
/// - the metadata nodes are the same,
/// - one of the metadata nodes is a base of the other (this can be
///   recursive, but it has to be the original node that's a base,
///   not just that the nodes have a common base), or
/// - one of the metadata nodes is a `tbaa.struct` node (the access
///   necessarily being a `memcpy`) with a subobject node that would
///   be allowed to alias with the other.
///
/// Our job here is to produce metadata nodes that will never say that
/// an alias is not allowed when there exists a type that would be consistent
/// with the types of the accesses from which the nodes were produced.
///
/// The last clause in both language rules permits character types to
/// alias objects of any type.  We handle this by converting all character
/// types (as well as `std::byte` and types with the `mayalias` attribute)
/// to a single metadata node (the `char` node), then making sure that
/// that node is a base of every other metadata node we generate.
/// We can always just conservatively use this node if we aren't otherwise
/// sure how to implement the language rules for a type.
///
/// Read literally, the C rule for aggregates permits an aggregate l-value
/// (e.g. of type `struct { int x; }`) to be used to access an object that
/// is not part of an aggregate object of that type (e.g. a local variable
/// of type `int`).  That case is perhaps sensical, but it would also permit
/// e.g. an l-value of type `struct { int x; float f; }` to be used to
/// access an object of type `float`, which is nonsense.  We interpret this
/// clause as just intending to permit objects to be accessed through an
/// l-value that properly references a containing object.
///
/// C++ does not have an explicit rule for aggregates because in C++
/// a non-member access to an aggregate l-value is always a call to a
/// constructor or assignment operator, which then accesses all the
/// subobjects.  In general, however, our interpretation of member
/// accesses is that they are also an access to the containing object
/// and therefore require such an object to exist at that address;
/// this permits us to just use the C rule for the accesses done by
/// trivial copy/move constructors/operators.
///
/// Both C and C++ permit some qualification differences.  In C, however,
/// qualification can only differ at the outermost level, whereas C++
/// allows qualification to differ in nested positions through the
/// similar-types rule.  This means that e.g. an l-value of type
/// `const float *` is not permitted to access an object of type
/// `float *` in C, but it is in C++.  We use the C++ rule
/// unconditionally; the C rule is needlessly strict and frequently
/// violated in practice by code that we don't want to say is wrong.
/// We implement this by just discarding type qualifiers within pointer-like
/// types when deriving TBAA nodes; basically, we produce the TBAA node
/// for the type that is unqualified at all the recursive positions
/// considered by the C++ similar type rule.  The implementation
/// doesn't actually construct this recursively-qualified type as a
/// `QualType`; it just ignores qualifiers when recursing into types.
///
/// The similar-type rule only really applies to the standard CVR
/// qualifiers, which never affect representations.  Qualifiers such as
/// address spaces that may involve a representation difference would
/// be totally appropriate to distinguish for TBAA purposes.  However,
/// the current implementation just discards all qualifiers.
///
/// We handle the signed/unsigned clause by just making unsigned types
/// use the the metadata node for the signed variant of the type.  In the
/// language rules, this only applies at the outermost level, and e.g. an
/// l-value of type `signed int *` is not permitted to alias an object of
/// type `unsigned int *`.  We choose not to distinguish those types when
/// pointer-type TBAA is enabled, however.
///
/// After discarding qualifiers and signedness differences as above,
/// the language rules come down to whether the types are compatible
/// (in C) or identical (in C++).  Even in C, most types are compatible
/// only with themselves.  The exceptions will be considered in the cases
/// below.
llvm::MDNode *CodeGenTBAA::getTypeInfoHelper(const Type *Ty) {
  uint64_t Size = Context.getTypeSizeInChars(Ty).getQuantity();

  // Handle builtin types.
  if (const BuiltinType *BTy = dyn_cast<BuiltinType>(Ty)) {
    switch (BTy->getKind()) {
    // Character types are special and can alias anything.
    // In C++, this technically only includes "char" and "unsigned char",
    // and not "signed char". In C, it includes all three. For now,
    // the risk of exploiting this detail in C++ seems likely to outweigh
    // the benefit.
    case BuiltinType::Char_U:
    case BuiltinType::Char_S:
    case BuiltinType::UChar:
    case BuiltinType::SChar:
      return getChar();

    // Unsigned types can alias their corresponding signed types.
    case BuiltinType::UShort:
      return getTypeInfo(Context.ShortTy);
    case BuiltinType::UInt:
      return getTypeInfo(Context.IntTy);
    case BuiltinType::ULong:
      return getTypeInfo(Context.LongTy);
    case BuiltinType::ULongLong:
      return getTypeInfo(Context.LongLongTy);
    case BuiltinType::UInt128:
      return getTypeInfo(Context.Int128Ty);

    case BuiltinType::UShortFract:
      return getTypeInfo(Context.ShortFractTy);
    case BuiltinType::UFract:
      return getTypeInfo(Context.FractTy);
    case BuiltinType::ULongFract:
      return getTypeInfo(Context.LongFractTy);

    case BuiltinType::SatUShortFract:
      return getTypeInfo(Context.SatShortFractTy);
    case BuiltinType::SatUFract:
      return getTypeInfo(Context.SatFractTy);
    case BuiltinType::SatULongFract:
      return getTypeInfo(Context.SatLongFractTy);

    case BuiltinType::UShortAccum:
      return getTypeInfo(Context.ShortAccumTy);
    case BuiltinType::UAccum:
      return getTypeInfo(Context.AccumTy);
    case BuiltinType::ULongAccum:
      return getTypeInfo(Context.LongAccumTy);

    case BuiltinType::SatUShortAccum:
      return getTypeInfo(Context.SatShortAccumTy);
    case BuiltinType::SatUAccum:
      return getTypeInfo(Context.SatAccumTy);
    case BuiltinType::SatULongAccum:
      return getTypeInfo(Context.SatLongAccumTy);

    // Treat all other builtin types as distinct types. This includes
    // treating wchar_t, char16_t, and char32_t as distinct from their
    // "underlying types".
    default:
      return createScalarTypeNode(BTy->getName(Features), getChar(), Size);
    }
  }

  // C++1z [basic.lval]p10: "If a program attempts to access the stored value of
  // an object through a glvalue of other than one of the following types the
  // behavior is undefined: [...] a char, unsigned char, or std::byte type."
  if (Ty->isStdByteType())
    return getChar();

  // Handle pointers and references.
  //
  // When PointerTBAA is disabled, all pointers and references use the same
  // "any pointer" TBAA node. Otherwise, we generate a type-specific TBAA
  // node and use the "any pointer" node as its base for compatibility between
  // TUs with different settings.  To implement the C++ similar-type rules
  // (which we also adopt in C), we need to ignore qualifiers on the
  // pointee type, and that has to be done recursively if the pointee type
  // is itself a pointer-like type.
  //
  // Currently we ignore the differences between pointer-like types and just
  // and use this tag for the type: `p<pointer depth> <inner type tag>`.
  // This means we give e.g. `char **` and `char A::**` the same TBAA tag.
  if ((Ty->isPointerType() || Ty->isReferenceType())) {
    llvm::MDNode *AnyPtr = createScalarTypeNode("any pointer", getChar(), Size);
    if (!CodeGenOpts.PointerTBAA)
      return AnyPtr;
    unsigned PtrDepth = 0;
    do {
      PtrDepth++;
      Ty = Ty->getPointeeType().getTypePtr()->getUnqualifiedDesugaredType();
      // Any array-like type is considered a pointer-to qualification.
      if (Ty && Ty->isArrayType()) {
        Ty = Ty->getAsArrayTypeUnsafe()->getElementType().getTypePtr();
      }
    } while (!Ty->getPointeeType().isNull());
    std::string PtrName;
    llvm::raw_string_ostream OS{PtrName};
    OS << "p" << PtrDepth << " ";
    appendPointeeName(OS, Ty);
    return createScalarTypeNode(PtrName, AnyPtr, Size);
  }

  // Accesses to arrays are accesses to objects of their element types.
  if (CodeGenOpts.ArrayTBAA && Ty->isArrayType())
    return getTypeInfo(cast<ArrayType>(Ty)->getElementType());

  // Enum types are distinct types. In C++ they have "underlying types",
  // however they aren't related for TBAA.
  if (const EnumType *ETy = dyn_cast<EnumType>(Ty)) {
    if (!Features.CPlusPlus)
      return getTypeInfo(ETy->getDecl()->getIntegerType());

    // In C++ mode, types have linkage, so we can rely on the ODR and
    // on their mangled names, if they're external.
    // TODO: Is there a way to get a program-wide unique name for a
    // decl with local linkage or no linkage?
    if (!ETy->getDecl()->isExternallyVisible())
      return getChar();

    SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    MContext.mangleCanonicalTypeName(QualType(ETy, 0), Out);
    return createScalarTypeNode(OutName, getChar(), Size);
  }

  if (const auto *EIT = dyn_cast<BitIntType>(Ty)) {
    SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    // Don't specify signed/unsigned since integer types can alias despite sign
    // differences.
    Out << "_BitInt(" << EIT->getNumBits() << ')';
    return createScalarTypeNode(OutName, getChar(), Size);
  }

  // For now, handle any other kind of type conservatively.
  return getChar();
}

llvm::MDNode *CodeGenTBAA::getTypeInfo(QualType QTy) {
  // At -O0 or relaxed aliasing, TBAA is not emitted for regular types.
  if (CodeGenOpts.OptimizationLevel == 0 || CodeGenOpts.RelaxedAliasing)
    return nullptr;

  // If the type has the may_alias attribute (even on a typedef), it is
  // effectively in the general char alias class.
  if (TypeHasMayAlias(QTy))
    return getChar();

  // We need this function to not fall back to returning the "omnipotent char"
  // type node for aggregate and union types. Otherwise, any dereference of an
  // aggregate will result into the may-alias access descriptor, meaning all
  // subsequent accesses to direct and indirect members of that aggregate will
  // be considered may-alias too.
  // TODO: Combine getTypeInfo() and getBaseTypeInfo() into a single function.
  if (isValidBaseType(QTy, CodeGenOpts))
    return getBaseTypeInfo(QTy);

  const Type *Ty = Context.getCanonicalType(QTy).getTypePtr();
  if (llvm::MDNode *N = MetadataCache[Ty])
    return N;

  // Note that the following helper call is allowed to add new nodes to the
  // cache, which invalidates all its previously obtained iterators. So we
  // first generate the node for the type and then add that node to the cache.
  llvm::MDNode *TypeNode = getTypeInfoHelper(Ty);
  return MetadataCache[Ty] = TypeNode;
}

TBAAAccessInfo CodeGenTBAA::getAccessInfo(QualType AccessType) {
  // Pointee values may have incomplete types, but they shall never be
  // dereferenced.
  if (AccessType->isIncompleteType())
    return TBAAAccessInfo::getIncompleteInfo();

  if (TypeHasMayAlias(AccessType))
    return TBAAAccessInfo::getMayAliasInfo();

  uint64_t Size = Context.getTypeSizeInChars(AccessType).getQuantity();
  return TBAAAccessInfo(getTypeInfo(AccessType), Size);
}

TBAAAccessInfo CodeGenTBAA::getVTablePtrAccessInfo(llvm::Type *VTablePtrType) {
  llvm::DataLayout DL(&Module);
  unsigned Size = DL.getPointerTypeSize(VTablePtrType);
  return TBAAAccessInfo(createScalarTypeNode("vtable pointer", getRoot(), Size),
                        Size);
}

bool
CodeGenTBAA::CollectFields(uint64_t BaseOffset,
                           QualType QTy,
                           SmallVectorImpl<llvm::MDBuilder::TBAAStructField> &
                             Fields,
                           bool MayAlias) {
  /* Things not handled yet include: C++ base classes, bitfields, */

  if (const RecordType *TTy = QTy->getAs<RecordType>()) {
    const RecordDecl *RD = TTy->getDecl()->getDefinition();
    if (RD->hasFlexibleArrayMember())
      return false;

    // TODO: Handle C++ base classes.
    if (const CXXRecordDecl *Decl = dyn_cast<CXXRecordDecl>(RD))
      if (Decl->bases_begin() != Decl->bases_end())
        return false;

    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);

    unsigned idx = 0;
    for (RecordDecl::field_iterator i = RD->field_begin(),
         e = RD->field_end(); i != e; ++i, ++idx) {
      if ((*i)->isZeroSize(Context) || (*i)->isUnnamedBitfield())
        continue;
      uint64_t Offset = BaseOffset +
                        Layout.getFieldOffset(idx) / Context.getCharWidth();
      QualType FieldQTy = i->getType();
      if (!CollectFields(Offset, FieldQTy, Fields,
                         MayAlias || TypeHasMayAlias(FieldQTy)))
        return false;
    }
    return true;
  }

  /* Otherwise, treat whatever it is as a field. */
  uint64_t Offset = BaseOffset;
  uint64_t Size = Context.getTypeSizeInChars(QTy).getQuantity();
  llvm::MDNode *TBAAType = MayAlias ? getChar() : getTypeInfo(QTy);
  llvm::MDNode *TBAATag = getAccessTagInfo(TBAAAccessInfo(TBAAType, Size));
  Fields.push_back(llvm::MDBuilder::TBAAStructField(Offset, Size, TBAATag));
  return true;
}

llvm::MDNode *
CodeGenTBAA::getTBAAStructInfo(QualType QTy) {
  const Type *Ty = Context.getCanonicalType(QTy).getTypePtr();

  if (llvm::MDNode *N = StructMetadataCache[Ty])
    return N;

  SmallVector<llvm::MDBuilder::TBAAStructField, 4> Fields;
  if (CollectFields(0, QTy, Fields, TypeHasMayAlias(QTy)))
    return MDHelper.createTBAAStructNode(Fields);

  // For now, handle any other kind of type conservatively.
  return StructMetadataCache[Ty] = nullptr;
}

llvm::MDNode *CodeGenTBAA::getBaseTypeInfoHelper(const Type *Ty) {
  if (auto *TTy = dyn_cast<RecordType>(Ty)) {
    const RecordDecl *RD = TTy->getDecl()->getDefinition();
    const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
    using TBAAStructField = llvm::MDBuilder::TBAAStructField;
    SmallVector<TBAAStructField, 4> Fields;
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      // Handle C++ base classes. Non-virtual bases can treated a kind of
      // field. Virtual bases are more complex and omitted, but avoid an
      // incomplete view for NewStructPathTBAA.
      if (CodeGenOpts.NewStructPathTBAA && CXXRD->getNumVBases() != 0)
        return nullptr;
      for (const CXXBaseSpecifier &B : CXXRD->bases()) {
        if (B.isVirtual())
          continue;
        QualType BaseQTy = B.getType();
        const CXXRecordDecl *BaseRD = BaseQTy->getAsCXXRecordDecl();
        if (BaseRD->isEmpty())
          continue;
        llvm::MDNode *TypeNode = isValidBaseType(BaseQTy, CodeGenOpts)
                                     ? getBaseTypeInfo(BaseQTy)
                                     : getTypeInfo(BaseQTy);
        if (!TypeNode)
          return nullptr;
        uint64_t Offset = Layout.getBaseClassOffset(BaseRD).getQuantity();
        uint64_t Size =
            Context.getASTRecordLayout(BaseRD).getDataSize().getQuantity();
        Fields.push_back(
            llvm::MDBuilder::TBAAStructField(Offset, Size, TypeNode));
      }
      // The order in which base class subobjects are allocated is unspecified,
      // so may differ from declaration order. In particular, Itanium ABI will
      // allocate a primary base first.
      // Since we exclude empty subobjects, the objects are not overlapping and
      // their offsets are unique.
      llvm::sort(Fields,
                 [](const TBAAStructField &A, const TBAAStructField &B) {
                   return A.Offset < B.Offset;
                 });
    }
    for (FieldDecl *Field : RD->fields()) {
      if (Field->isZeroSize(Context) || Field->isUnnamedBitfield())
        continue;
      QualType FieldQTy = Field->getType();
      llvm::MDNode *TypeNode = isValidBaseType(FieldQTy, CodeGenOpts)
                                   ? getBaseTypeInfo(FieldQTy)
                                   : getTypeInfo(FieldQTy);
      if (!TypeNode)
        return nullptr;

      uint64_t BitOffset = Layout.getFieldOffset(Field->getFieldIndex());
      uint64_t Offset = Context.toCharUnitsFromBits(BitOffset).getQuantity();
      uint64_t Size = Context.getTypeSizeInChars(FieldQTy).getQuantity();
      Fields.push_back(llvm::MDBuilder::TBAAStructField(Offset, Size,
                                                        TypeNode));
    }

    SmallString<256> OutName;
    if (Features.CPlusPlus) {
      // Don't use the mangler for C code.
      llvm::raw_svector_ostream Out(OutName);
      MContext.mangleCanonicalTypeName(QualType(Ty, 0), Out);
    } else {
      OutName = RD->getName();
    }

    if (CodeGenOpts.NewStructPathTBAA) {
      llvm::MDNode *Parent = getChar();
      uint64_t Size = Context.getTypeSizeInChars(Ty).getQuantity();
      llvm::Metadata *Id = MDHelper.createString(OutName);
      return MDHelper.createTBAATypeNode(Parent, Size, Id, Fields);
    }

    // Create the struct type node with a vector of pairs (offset, type).
    SmallVector<std::pair<llvm::MDNode*, uint64_t>, 4> OffsetsAndTypes;
    for (const auto &Field : Fields)
        OffsetsAndTypes.push_back(std::make_pair(Field.Type, Field.Offset));
    return MDHelper.createTBAAStructTypeNode(OutName, OffsetsAndTypes);
  }

  return nullptr;
}

llvm::MDNode *CodeGenTBAA::getBaseTypeInfo(QualType QTy) {
  if (!isValidBaseType(QTy, CodeGenOpts))
    return nullptr;

  const Type *Ty = Context.getCanonicalType(QTy).getTypePtr();

  // nullptr is a valid value in the cache, so use find rather than []
  auto I = BaseTypeMetadataCache.find(Ty);
  if (I != BaseTypeMetadataCache.end())
    return I->second;

  // First calculate the metadata, before recomputing the insertion point, as
  // the helper can recursively call us.
  llvm::MDNode *TypeNode = getBaseTypeInfoHelper(Ty);
  LLVM_ATTRIBUTE_UNUSED auto inserted =
      BaseTypeMetadataCache.insert({Ty, TypeNode});
  assert(inserted.second && "BaseType metadata was already inserted");

  return TypeNode;
}

llvm::MDNode *CodeGenTBAA::getAccessTagInfo(TBAAAccessInfo Info) {
  assert(!Info.isIncomplete() && "Access to an object of an incomplete type!");

  if (Info.isMayAlias())
    Info = TBAAAccessInfo(getChar(), Info.Size);

  if (!Info.AccessType)
    return nullptr;

  if (!CodeGenOpts.StructPathTBAA)
    Info = TBAAAccessInfo(Info.AccessType, Info.Size);

  llvm::MDNode *&N = AccessTagMetadataCache[Info];
  if (N)
    return N;

  if (!Info.BaseType) {
    Info.BaseType = Info.AccessType;
    assert(!Info.Offset && "Nonzero offset for an access with no base type!");
  }
  if (CodeGenOpts.NewStructPathTBAA) {
    return N = MDHelper.createTBAAAccessTag(Info.BaseType, Info.AccessType,
                                            Info.Offset, Info.Size);
  }
  return N = MDHelper.createTBAAStructTagNode(Info.BaseType, Info.AccessType,
                                              Info.Offset);
}

TBAAAccessInfo CodeGenTBAA::mergeTBAAInfoForCast(TBAAAccessInfo SourceInfo,
                                                 TBAAAccessInfo TargetInfo) {
  if (SourceInfo.isMayAlias() || TargetInfo.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();
  return TargetInfo;
}

TBAAAccessInfo
CodeGenTBAA::mergeTBAAInfoForConditionalOperator(TBAAAccessInfo InfoA,
                                                 TBAAAccessInfo InfoB) {
  if (InfoA == InfoB)
    return InfoA;

  if (!InfoA || !InfoB)
    return TBAAAccessInfo();

  if (InfoA.isMayAlias() || InfoB.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();

  // TODO: Implement the rest of the logic here. For example, two accesses
  // with same final access types result in an access to an object of that final
  // access type regardless of their base types.
  return TBAAAccessInfo::getMayAliasInfo();
}

TBAAAccessInfo
CodeGenTBAA::mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo DestInfo,
                                            TBAAAccessInfo SrcInfo) {
  if (DestInfo == SrcInfo)
    return DestInfo;

  if (!DestInfo || !SrcInfo)
    return TBAAAccessInfo();

  if (DestInfo.isMayAlias() || SrcInfo.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();

  // TODO: Implement the rest of the logic here. For example, two accesses
  // with same final access types result in an access to an object of that final
  // access type regardless of their base types.
  return TBAAAccessInfo::getMayAliasInfo();
}
