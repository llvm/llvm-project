//===--- CodeGenTypes.cpp - TBAA information for LLVM CodeGen -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CodeGenOptions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Metadata.h"
#include "llvm/Constants.h"
#include "llvm/Type.h"
using namespace clang;
using namespace CodeGen;

CodeGenTBAA::CodeGenTBAA(ASTContext &Ctx, llvm::LLVMContext& VMContext,
                         const CodeGenOptions &CGO,
                         const LangOptions &Features, MangleContext &MContext)
  : Context(Ctx), CodeGenOpts(CGO), Features(Features), MContext(MContext),
    MDHelper(VMContext), Root(0), Char(0) {
}

CodeGenTBAA::~CodeGenTBAA() {
}

llvm::MDNode *CodeGenTBAA::getRoot() {
  // Define the root of the tree. This identifies the tree, so that
  // if our LLVM IR is linked with LLVM IR from a different front-end
  // (or a different version of this front-end), their TBAA trees will
  // remain distinct, and the optimizer will treat them conservatively.
  if (!Root)
    Root = MDHelper.createTBAARoot("Simple C/C++ TBAA");

  return Root;
}

llvm::MDNode *CodeGenTBAA::getChar() {
  // Define the root of the tree for user-accessible memory. C and C++
  // give special powers to char and certain similar types. However,
  // these special powers only cover user-accessible memory, and doesn't
  // include things like vtables.
  if (!Char)
    Char = MDHelper.createTBAANode("omnipotent char", getRoot());

  return Char;
}

static bool TypeHasMayAlias(QualType QTy) {
  // Tagged types have declarations, and therefore may have attributes.
  if (const TagType *TTy = dyn_cast<TagType>(QTy))
    return TTy->getDecl()->hasAttr<MayAliasAttr>();

  // Typedef types have declarations, and therefore may have attributes.
  if (const TypedefType *TTy = dyn_cast<TypedefType>(QTy)) {
    if (TTy->getDecl()->hasAttr<MayAliasAttr>())
      return true;
    // Also, their underlying types may have relevant attributes.
    return TypeHasMayAlias(TTy->desugar());
  }

  return false;
}

llvm::MDNode *
CodeGenTBAA::getTBAAInfo(QualType QTy) {
  // At -O0 TBAA is not emitted for regular types.
  if (CodeGenOpts.OptimizationLevel == 0 || CodeGenOpts.RelaxedAliasing)
    return NULL;

  // If the type has the may_alias attribute (even on a typedef), it is
  // effectively in the general char alias class.
  if (TypeHasMayAlias(QTy))
    return getChar();

  const Type *Ty = Context.getCanonicalType(QTy).getTypePtr();

  if (llvm::MDNode *N = MetadataCache[Ty])
    return N;

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
      return getTBAAInfo(Context.ShortTy);
    case BuiltinType::UInt:
      return getTBAAInfo(Context.IntTy);
    case BuiltinType::ULong:
      return getTBAAInfo(Context.LongTy);
    case BuiltinType::ULongLong:
      return getTBAAInfo(Context.LongLongTy);
    case BuiltinType::UInt128:
      return getTBAAInfo(Context.Int128Ty);

    // Treat all other builtin types as distinct types. This includes
    // treating wchar_t, char16_t, and char32_t as distinct from their
    // "underlying types".
    default:
      return MetadataCache[Ty] =
        MDHelper.createTBAANode(BTy->getName(Features), getChar());
    }
  }

  // Handle pointers.
  // TODO: Implement C++'s type "similarity" and consider dis-"similar"
  // pointers distinct.
  if (Ty->isPointerType())
    return MetadataCache[Ty] = MDHelper.createTBAANode("any pointer",
                                                       getChar());

  // Enum types are distinct types. In C++ they have "underlying types",
  // however they aren't related for TBAA.
  if (const EnumType *ETy = dyn_cast<EnumType>(Ty)) {
    // In C mode, two anonymous enums are compatible iff their members
    // are the same -- see C99 6.2.7p1. For now, be conservative. We could
    // theoretically implement this by combining information about all the
    // members into a single identifying MDNode.
    if (!Features.CPlusPlus &&
        ETy->getDecl()->getTypedefNameForAnonDecl())
      return MetadataCache[Ty] = getChar();

    // In C++ mode, types have linkage, so we can rely on the ODR and
    // on their mangled names, if they're external.
    // TODO: Is there a way to get a program-wide unique name for a
    // decl with local linkage or no linkage?
    if (Features.CPlusPlus &&
        ETy->getDecl()->getLinkage() != ExternalLinkage)
      return MetadataCache[Ty] = getChar();

    // TODO: This is using the RTTI name. Is there a better way to get
    // a unique string for a type?
    SmallString<256> OutName;
    llvm::raw_svector_ostream Out(OutName);
    MContext.mangleCXXRTTIName(QualType(ETy, 0), Out);
    Out.flush();
    return MetadataCache[Ty] = MDHelper.createTBAANode(OutName, getChar());
  }

  // For now, handle any other kind of type conservatively.
  return MetadataCache[Ty] = getChar();
}

llvm::MDNode *CodeGenTBAA::getTBAAInfoForVTablePtr() {
  return MDHelper.createTBAANode("vtable pointer", getRoot());
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
  llvm::MDNode *TBAAInfo = MayAlias ? getChar() : getTBAAInfo(QTy);
  Fields.push_back(llvm::MDBuilder::TBAAStructField(Offset, Size, TBAAInfo));
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
  return StructMetadataCache[Ty] = NULL;
}
