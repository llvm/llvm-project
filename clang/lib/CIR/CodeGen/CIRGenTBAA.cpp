#include "CIRGenTBAA.h"
#include "CIRGenCXXABI.h"
#include "CIRGenTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
namespace clang::CIRGen {

cir::TBAAAttr tbaa_NYI(mlir::MLIRContext *mlirContext) {
  return cir::TBAAAttr::get(mlirContext);
}

CIRGenTBAA::CIRGenTBAA(mlir::MLIRContext *mlirContext,
                       clang::ASTContext &astContext, CIRGenTypes &types,
                       mlir::ModuleOp moduleOp,
                       const clang::CodeGenOptions &codeGenOpts,
                       const clang::LangOptions &features)
    : mlirContext(mlirContext), astContext(astContext), types(types),
      moduleOp(moduleOp), codeGenOpts(codeGenOpts), features(features) {}

cir::TBAAAttr CIRGenTBAA::getChar() {
  return cir::TBAAOmnipotentCharAttr::get(mlirContext);
}

static bool typeHasMayAlias(clang::QualType qty) {
  // Tagged types have declarations, and therefore may have attributes.
  if (auto *td = qty->getAsTagDecl())
    if (td->hasAttr<MayAliasAttr>())
      return true;

  // Also look for may_alias as a declaration attribute on a typedef.
  // FIXME: We should follow GCC and model may_alias as a type attribute
  // rather than as a declaration attribute.
  while (auto *tt = qty->getAs<TypedefType>()) {
    if (tt->getDecl()->hasAttr<MayAliasAttr>())
      return true;
    qty = tt->desugar();
  }
  return false;
}

/// Check if the given type is a valid base type to be used in access tags.
static bool isValidBaseType(clang::QualType qty) {
  if (const clang::RecordType *tty = qty->getAs<clang::RecordType>()) {
    const clang::RecordDecl *rd = tty->getDecl()->getDefinition();
    // Incomplete types are not valid base access types.
    if (!rd)
      return false;
    if (rd->hasFlexibleArrayMember())
      return false;
    // rd can be struct, union, class, interface or enum.
    // For now, we only handle struct and class.
    if (rd->isStruct() || rd->isClass())
      return true;
  }
  return false;
}

cir::TBAAAttr CIRGenTBAA::getScalarTypeInfo(clang::QualType qty) {
  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();
  assert(mlir::isa<clang::BuiltinType>(ty));
  const clang::BuiltinType *bty = mlir::dyn_cast<BuiltinType>(ty);
  return cir::TBAAScalarAttr::get(mlirContext, bty->getName(features),
                                  types.convertType(qty));
}

cir::TBAAAttr CIRGenTBAA::getTypeInfoHelper(clang::QualType qty) {
  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();
  // Handle builtin types.
  if (const clang::BuiltinType *bty = mlir::dyn_cast<BuiltinType>(ty)) {
    switch (bty->getKind()) {
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
      return getScalarTypeInfo(astContext.ShortTy);
    case BuiltinType::UInt:
      return getScalarTypeInfo(astContext.IntTy);
    case BuiltinType::ULong:
      return getScalarTypeInfo(astContext.LongTy);
    case BuiltinType::ULongLong:
      return getScalarTypeInfo(astContext.LongLongTy);
    case BuiltinType::UInt128:
      return getScalarTypeInfo(astContext.Int128Ty);

    case BuiltinType::UShortFract:
      return getScalarTypeInfo(astContext.ShortFractTy);
    case BuiltinType::UFract:
      return getScalarTypeInfo(astContext.FractTy);
    case BuiltinType::ULongFract:
      return getScalarTypeInfo(astContext.LongFractTy);

    case BuiltinType::SatUShortFract:
      return getScalarTypeInfo(astContext.SatShortFractTy);
    case BuiltinType::SatUFract:
      return getScalarTypeInfo(astContext.SatFractTy);
    case BuiltinType::SatULongFract:
      return getScalarTypeInfo(astContext.SatLongFractTy);

    case BuiltinType::UShortAccum:
      return getScalarTypeInfo(astContext.ShortAccumTy);
    case BuiltinType::UAccum:
      return getScalarTypeInfo(astContext.AccumTy);
    case BuiltinType::ULongAccum:
      return getScalarTypeInfo(astContext.LongAccumTy);

    case BuiltinType::SatUShortAccum:
      return getScalarTypeInfo(astContext.SatShortAccumTy);
    case BuiltinType::SatUAccum:
      return getScalarTypeInfo(astContext.SatAccumTy);
    case BuiltinType::SatULongAccum:
      return getScalarTypeInfo(astContext.SatLongAccumTy);

    // Treat all other builtin types as distinct types. This includes
    // treating wchar_t, char16_t, and char32_t as distinct from their
    // "underlying types".
    default:
      return getScalarTypeInfo(qty);
    }
  }
  // C++1z [basic.lval]p10: "If a program attempts to access the stored value of
  // an object through a glvalue of other than one of the following types the
  // behavior is undefined: [...] a char, unsigned char, or std::byte type."
  if (ty->isStdByteType())
    return getChar();

  // Handle pointers and references.
  //
  // C has a very strict rule for pointer aliasing. C23 6.7.6.1p2:
  //     For two pointer types to be compatible, both shall be identically
  //     qualified and both shall be pointers to compatible types.
  //
  // This rule is impractically strict; we want to at least ignore CVR
  // qualifiers. Distinguishing by CVR qualifiers would make it UB to
  // e.g. cast a `char **` to `const char * const *` and dereference it,
  // which is too common and useful to invalidate. C++'s similar types
  // rule permits qualifier differences in these nested positions; in fact,
  // C++ even allows that cast as an implicit conversion.
  //
  // Other qualifiers could theoretically be distinguished, especially if
  // they involve a significant representation difference.  We don't
  // currently do so, however.
  if (ty->isPointerType() || ty->isReferenceType()) {
    if (!codeGenOpts.PointerTBAA) {
      return cir::TBAAScalarAttr::get(mlirContext, "any pointer",
                                      types.convertType(qty));
    }
    assert(!cir::MissingFeatures::tbaaPointer());
    return tbaa_NYI(mlirContext);
  }
  // Accesses to arrays are accesses to objects of their element types.
  if (codeGenOpts.NewStructPathTBAA && ty->isArrayType()) {
    assert(!cir::MissingFeatures::tbaaNewStructPath());
    return tbaa_NYI(mlirContext);
  }
  // Enum types are distinct types. In C++ they have "underlying types",
  // however they aren't related for TBAA.
  if (const EnumType *ety = dyn_cast<EnumType>(ty)) {
    assert(!cir::MissingFeatures::tbaaTagForEnum());
    return tbaa_NYI(mlirContext);
  }
  if (const auto *eit = dyn_cast<BitIntType>(ty)) {
    assert(!cir::MissingFeatures::tbaaTagForBitInt());
    return tbaa_NYI(mlirContext);
  }
  // For now, handle any other kind of type conservatively.
  return getChar();
}

cir::TBAAAttr CIRGenTBAA::getTypeInfo(clang::QualType qty) {
  // At -O0 or relaxed aliasing, TBAA is not emitted for regular types.
  if (codeGenOpts.OptimizationLevel == 0 || codeGenOpts.RelaxedAliasing) {
    return nullptr;
  }

  // If the type has the may_alias attribute (even on a typedef), it is
  // effectively in the general char alias class.
  if (typeHasMayAlias(qty)) {
    assert(!cir::MissingFeatures::tbaaMayAlias());
    return getChar();
  }
  // We need this function to not fall back to returning the "omnipotent char"
  // type node for aggregate and union types. Otherwise, any dereference of an
  // aggregate will result into the may-alias access descriptor, meaning all
  // subsequent accesses to direct and indirect members of that aggregate will
  // be considered may-alias too.
  // function.
  if (isValidBaseType(qty)) {
    assert(!cir::MissingFeatures::tbaaTagForStruct());
    return getValidBaseTypeInfo(qty);
  }

  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();
  if (metadataCache.contains(ty)) {
    return metadataCache[ty];
  }

  // Note that the following helper call is allowed to add new nodes to the
  // cache, which invalidates all its previously obtained iterators. So we
  // first generate the node for the type and then add that node to the
  // cache.
  auto typeNode = getTypeInfoHelper(qty);
  return metadataCache[ty] = typeNode;
}

TBAAAccessInfo CIRGenTBAA::getAccessInfo(clang::QualType accessType) {
  // Pointee values may have incomplete types, but they shall never be
  // dereferenced.
  if (accessType->isIncompleteType()) {
    assert(!cir::MissingFeatures::tbaaIncompleteType());
    return TBAAAccessInfo::getIncompleteInfo();
  }

  if (typeHasMayAlias(accessType)) {
    assert(!cir::MissingFeatures::tbaaMayAlias());
    return TBAAAccessInfo::getMayAliasInfo();
  }

  uint64_t size = astContext.getTypeSizeInChars(accessType).getQuantity();
  return TBAAAccessInfo(getTypeInfo(accessType), size);
}

TBAAAccessInfo CIRGenTBAA::getVTablePtrAccessInfo(mlir::Type vtablePtrType) {
  assert(!cir::MissingFeatures::tbaaVTablePtr());
  return TBAAAccessInfo();
}

mlir::ArrayAttr CIRGenTBAA::getTBAAStructInfo(clang::QualType qty) {
  assert(!cir::MissingFeatures::tbaaStruct() && "tbaa.struct NYI");
  return mlir::ArrayAttr();
}

cir::TBAAAttr CIRGenTBAA::getBaseTypeInfo(clang::QualType qty) {
  return isValidBaseType(qty) ? getValidBaseTypeInfo(qty) : nullptr;
}

cir::TBAAAttr CIRGenTBAA::getValidBaseTypeInfo(clang::QualType qty) {
  assert(isValidBaseType(qty) && "Must be a valid base type");

  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();

  // nullptr is a valid value in the cache, so use find rather than []
  auto iter = baseTypeMetadataCache.find(ty);
  if (iter != baseTypeMetadataCache.end())
    return iter->second;

  // First calculate the metadata, before recomputinyg the insertion point, as
  // the helper can recursively call us.
  auto typeNode = getBaseTypeInfoHelper(ty);
  LLVM_ATTRIBUTE_UNUSED auto inserted =
      baseTypeMetadataCache.insert({ty, typeNode});
  assert(inserted.second && "BaseType metadata was already inserted");

  return typeNode;
}
cir::TBAAAttr CIRGenTBAA::getBaseTypeInfoHelper(const clang::Type *ty) {
  using namespace clang;
  if (auto *tty = mlir::dyn_cast<clang::RecordType>(ty)) {
    const clang::RecordDecl *rd = tty->getDecl()->getDefinition();
    const ASTRecordLayout &layout = astContext.getASTRecordLayout(rd);
    SmallVector<cir::TBAAMemberAttr, 4> fields;
    if (const CXXRecordDecl *cxxrd = dyn_cast<CXXRecordDecl>(rd)) {
      // Handle C++ base classes. Non-virtual bases can treated a kind of
      // field. Virtual bases are more complex and omitted, but avoid an
      // incomplete view for NewStructPathTBAA.
      if (codeGenOpts.NewStructPathTBAA && cxxrd->getNumVBases() != 0)
        return nullptr;
      for (const CXXBaseSpecifier &cxxBaseSpecifier : cxxrd->bases()) {
        if (cxxBaseSpecifier.isVirtual())
          continue;
        QualType baseQTy = cxxBaseSpecifier.getType();
        const CXXRecordDecl *baseRD = baseQTy->getAsCXXRecordDecl();
        if (baseRD->isEmpty())
          continue;
        auto typeNode = isValidBaseType(baseQTy) ? getValidBaseTypeInfo(baseQTy)
                                                 : getTypeInfo(baseQTy);
        if (!typeNode)
          return nullptr;
        uint64_t offset = layout.getBaseClassOffset(baseRD).getQuantity();
        [[maybe_unused]] uint64_t size =
            astContext.getASTRecordLayout(baseRD).getDataSize().getQuantity();
        fields.push_back(
            cir::TBAAMemberAttr::get(mlirContext, typeNode, offset));
      }
      // The order in which base class subobjects are allocated is
      // unspecified, so may differ from declaration order. In particular,
      // Itanium ABI will allocate a primary base first. Since we exclude
      // empty subobjects, the objects are not overlapping and their offsets
      // are unique.
      llvm::sort(fields, [](const cir::TBAAMemberAttr &lhs,
                            const cir::TBAAMemberAttr &rhs) {
        return lhs.getOffset() < rhs.getOffset();
      });
    }
    for (FieldDecl *field : rd->fields()) {
      if (field->isZeroSize(astContext) || field->isUnnamedBitField())
        continue;
      QualType fieldQTy = field->getType();
      auto typeNode = isValidBaseType(fieldQTy) ? getValidBaseTypeInfo(fieldQTy)
                                                : getTypeInfo(fieldQTy);
      if (!typeNode)
        return nullptr;

      uint64_t bitOffset = layout.getFieldOffset(field->getFieldIndex());
      uint64_t offset = astContext.toCharUnitsFromBits(bitOffset).getQuantity();
      [[maybe_unused]] uint64_t size =
          astContext.getTypeSizeInChars(fieldQTy).getQuantity();
      fields.push_back(cir::TBAAMemberAttr::get(mlirContext, typeNode, offset));
    }

    SmallString<256> outName;
    if (features.CPlusPlus) {
      // Don't use the mangler for C code.
      llvm::raw_svector_ostream out(outName);
      types.getCXXABI().getMangleContext().mangleCanonicalTypeName(
          QualType(ty, 0), out);
    } else {
      outName = rd->getName();
    }

    if (codeGenOpts.NewStructPathTBAA) {
      assert(!cir::MissingFeatures::tbaaNewStructPath());
      return nullptr;
    }
    return cir::TBAAStructAttr::get(mlirContext, outName, fields);
  }
  return nullptr;
}
cir::TBAAAttr CIRGenTBAA::getAccessTagInfo(TBAAAccessInfo tbaaInfo) {
  assert(!tbaaInfo.isIncomplete() &&
         "Access to an object of an incomplete type!");

  if (tbaaInfo.isMayAlias()) {
    assert(!cir::MissingFeatures::tbaaMayAlias());
    tbaaInfo = TBAAAccessInfo(getChar(), tbaaInfo.size);
  }
  if (!tbaaInfo.accessType) {
    return nullptr;
  }

  if (!codeGenOpts.StructPathTBAA)
    tbaaInfo = TBAAAccessInfo(tbaaInfo.accessType, tbaaInfo.size);

  if (!tbaaInfo.baseType) {
    tbaaInfo.baseType = tbaaInfo.accessType;
    assert(!tbaaInfo.offset &&
           "Nonzero offset for an access with no base type!");
  }
  if (codeGenOpts.NewStructPathTBAA) {
    assert(!cir::MissingFeatures::tbaaNewStructPath());
    return tbaa_NYI(mlirContext);
  }
  if (tbaaInfo.baseType == tbaaInfo.accessType) {
    return tbaaInfo.accessType;
  }
  return cir::TBAATagAttr::get(mlirContext, tbaaInfo.baseType,
                               tbaaInfo.accessType, tbaaInfo.offset);
}

TBAAAccessInfo CIRGenTBAA::mergeTBAAInfoForCast(TBAAAccessInfo sourceInfo,
                                                TBAAAccessInfo targetInfo) {
  assert(!cir::MissingFeatures::tbaaMergeTBAAInfo());
  return TBAAAccessInfo();
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForConditionalOperator(TBAAAccessInfo infoA,
                                                TBAAAccessInfo infoB) {
  assert(!cir::MissingFeatures::tbaaMergeTBAAInfo());
  return TBAAAccessInfo();
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo destInfo,
                                           TBAAAccessInfo srcInfo) {
  assert(!cir::MissingFeatures::tbaaMergeTBAAInfo());
  return TBAAAccessInfo();
}

} // namespace clang::CIRGen
