//===--- CIRGenTBAA.cpp - TBAA information for CIR CodeGen ----------------===//
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

#include "CIRGenTBAA.h"
#include "CIRGenCXXABI.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CIR/MissingFeatures.h"
#include "clang/CodeGenUtils/TBAAUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

using namespace clang;
using namespace clang::CIRGen;

CIRGenTBAA::CIRGenTBAA(mlir::MLIRContext &mlirContext, ASTContext &astContext,
                       CIRGenCXXABI &cxxABI, const CodeGenOptions &codeGenOpts,
                       const LangOptions &features)
    : mlirContext(mlirContext), astContext(astContext), cxxABI(cxxABI),
      codeGenOpts(codeGenOpts), features(features),
      mangleCtx(ItaniumMangleContext::create(astContext,
                                             astContext.getDiagnostics())) {}

CIRGenTBAA::~CIRGenTBAA() = default;

cir::TBAAScalarAttr CIRGenTBAA::createScalarTypeNode(StringRef name,
                                                     cir::TBAAScalarAttr parent,
                                                     uint64_t size) {
  assert(!cir::MissingFeatures::tbaaNewStructPath());
  return cir::TBAAScalarAttr::get(&mlirContext, name, size, parent);
}

cir::TBAAOmnipotentCharAttr CIRGenTBAA::getChar() {
  return cir::TBAAOmnipotentCharAttr::get(&mlirContext);
}

cir::TBAAScalarAttr CIRGenTBAA::getAnyPtr(unsigned ptrDepth) {
  assert(ptrDepth >= 1 && "Pointer must have some depth");

  // Populate at least ptrDepth elements in anyPtrs. These are the type nodes
  // for "any" pointers of increasing pointer depth, and are organized in the
  // hierarchy: any pointer <- any p2 pointer <- any p3 pointer <- ...
  if (anyPtrs.size() < ptrDepth) {
    anyPtrs.reserve(ptrDepth);
    uint64_t size =
        astContext
            .toCharUnitsFromBits(
                astContext.getTargetInfo().getPointerWidth(LangAS::Default))
            .getQuantity();
    if (anyPtrs.empty())
      anyPtrs.push_back(createScalarTypeNode("any pointer", {}, size));
    for (size_t idx = anyPtrs.size(); idx < ptrDepth; ++idx) {
      std::string name = ("any p" + llvm::Twine(idx + 1) + " pointer").str();
      anyPtrs.push_back(createScalarTypeNode(name, anyPtrs[idx - 1], size));
    }
  }

  return anyPtrs[ptrDepth - 1];
}

/// Returns the name of a scalar type descriptor.
static StringRef getTypeDescriptorName(mlir::Attribute typeDesc) {
  if (auto scalar = mlir::dyn_cast<cir::TBAAScalarAttr>(typeDesc))
    return scalar.getId();
  assert(mlir::isa<cir::TBAAOmnipotentCharAttr>(typeDesc) &&
         "expected a scalar type descriptor");
  return "omnipotent char";
}

mlir::Attribute CIRGenTBAA::getTypeInfoHelper(const clang::Type *ty) {
  uint64_t size = astContext.getTypeSizeInChars(ty).getQuantity();

  // Handle builtin types.
  if (const auto *bty = dyn_cast<BuiltinType>(ty)) {
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
      return getTypeInfo(astContext.ShortTy);
    case BuiltinType::UInt:
      return getTypeInfo(astContext.IntTy);
    case BuiltinType::ULong:
      return getTypeInfo(astContext.LongTy);
    case BuiltinType::ULongLong:
      return getTypeInfo(astContext.LongLongTy);
    case BuiltinType::UInt128:
      return getTypeInfo(astContext.Int128Ty);

    case BuiltinType::UShortFract:
      return getTypeInfo(astContext.ShortFractTy);
    case BuiltinType::UFract:
      return getTypeInfo(astContext.FractTy);
    case BuiltinType::ULongFract:
      return getTypeInfo(astContext.LongFractTy);

    case BuiltinType::SatUShortFract:
      return getTypeInfo(astContext.SatShortFractTy);
    case BuiltinType::SatUFract:
      return getTypeInfo(astContext.SatFractTy);
    case BuiltinType::SatULongFract:
      return getTypeInfo(astContext.SatLongFractTy);

    case BuiltinType::UShortAccum:
      return getTypeInfo(astContext.ShortAccumTy);
    case BuiltinType::UAccum:
      return getTypeInfo(astContext.AccumTy);
    case BuiltinType::ULongAccum:
      return getTypeInfo(astContext.LongAccumTy);

    case BuiltinType::SatUShortAccum:
      return getTypeInfo(astContext.SatShortAccumTy);
    case BuiltinType::SatUAccum:
      return getTypeInfo(astContext.SatAccumTy);
    case BuiltinType::SatULongAccum:
      return getTypeInfo(astContext.SatLongAccumTy);

    // Treat all other builtin types as distinct types. This includes
    // treating wchar_t, char16_t, and char32_t as distinct from their
    // "underlying types".
    default:
      return createScalarTypeNode(bty->getName(features), {}, size);
    }
  }

  // C++1z [basic.lval]p10: "If a program attempts to access the stored value of
  // an object through a glvalue of other than one of the following types the
  // behavior is undefined: [...] a char, unsigned char, or std::byte type."
  if (ty->isStdByteType())
    return getChar();

  // Handle pointers and references. See CodeGenTBAA::getTypeInfoHelper for the
  // rationale: pointers are distinguished by their pointer depth and by their
  // base pointee type, ignoring qualifiers and array types.
  if (ty->isPointerType() || ty->isReferenceType()) {
    if (!codeGenOpts.PointerTBAA)
      return getAnyPtr();

    unsigned ptrDepth = 0;
    do {
      ptrDepth++;
      ty = ty->getPointeeType()->getBaseElementTypeUnsafe();
    } while (ty->isPointerType());

    // While there are no special rules in the standards regarding void pointers
    // and strict aliasing, emitting distinct tags for void pointers break some
    // common idioms and there is no good alternative to re-write the code
    // without strict-aliasing violations.
    if (ty->isVoidType())
      return getAnyPtr(ptrDepth);

    assert(!isa<VariableArrayType>(ty));
    // When the underlying type is a builtin type, we compute the pointee type
    // string recursively, which is implicitly more forgiving than the standards
    // require.
    SmallString<256> tyName;
    if (isa<BuiltinType>(ty)) {
      tyName = getTypeDescriptorName(getTypeInfoHelper(ty));
    } else {
      // Be conservative if the type isn't a RecordType. We are specifically
      // required to do this for member pointers until we implement the
      // similar-types rule.
      const auto *rt = ty->getAsCanonical<RecordType>();
      if (!rt)
        return getAnyPtr(ptrDepth);

      // For unnamed structs or unions C's compatible types rule applies. Two
      // compatible types in different compilation units can have different
      // mangled names, meaning the metadata emitted below would incorrectly
      // mark them as no-alias. Use AnyPtr for such types in both C and C++, as
      // C and C++ types may be visible when doing LTO.
      if (!rt->getDecl()->getDeclName())
        return getAnyPtr(ptrDepth);

      // For non-builtin types use the mangled name of the canonical type.
      llvm::raw_svector_ostream tyOut(tyName);
      mangleCtx->mangleCanonicalTypeName(QualType(ty, 0), tyOut);
    }

    SmallString<256> outName("p");
    outName += std::to_string(ptrDepth);
    outName += " ";
    outName += tyName;
    return createScalarTypeNode(outName, getAnyPtr(ptrDepth), size);
  }

  // Accesses to matrix types are accesses to objects of their element types.
  if (const auto *mty = dyn_cast<MatrixType>(ty)) {
    assert(isa<ConstantMatrixType>(ty) &&
           "only ConstantMatrixType should reach CodeGen");
    return getTypeInfo(mty->getElementType());
  }

  // Accesses to complex types are accesses to objects of their element types.
  if (const auto *cty = dyn_cast<ComplexType>(ty))
    return getTypeInfo(cty->getElementType());

  // Enum types are distinct types. In C++ they have "underlying types",
  // however they aren't related for TBAA.
  if (const auto *ety = dyn_cast<EnumType>(ty)) {
    const EnumDecl *ed = ety->getDecl()->getDefinitionOrSelf();
    if (!features.CPlusPlus)
      return getTypeInfo(ed->getIntegerType());

    // In C++ mode, types have linkage, so we can rely on the ODR and
    // on their mangled names, if they're external.
    if (!ed->isExternallyVisible())
      return getChar();

    SmallString<256> outName;
    llvm::raw_svector_ostream out(outName);
    cxxABI.getMangleContext().mangleCanonicalTypeName(QualType(ety, 0), out);
    return createScalarTypeNode(outName, {}, size);
  }

  if (const auto *eit = dyn_cast<BitIntType>(ty)) {
    SmallString<256> outName;
    llvm::raw_svector_ostream out(outName);
    // Don't specify signed/unsigned since integer types can alias despite sign
    // differences.
    out << "_BitInt(" << eit->getNumBits() << ')';
    return createScalarTypeNode(outName, {}, size);
  }

  // For now, handle any other kind of type conservatively.
  return getChar();
}

mlir::Attribute CIRGenTBAA::getTypeInfo(QualType qty) {
  // If the type has the may_alias attribute (even on a typedef), it is
  // effectively in the general char alias class.
  if (CodeGenUtils::typeHasMayAlias(qty))
    return getChar();

  // We need this function to not fall back to returning the "omnipotent char"
  // type node for aggregate and union types. Otherwise, any dereference of an
  // aggregate will result into the may-alias access descriptor, meaning all
  // subsequent accesses to direct and indirect members of that aggregate will
  // be considered may-alias too.
  if (CodeGenUtils::isValidTBAABaseType(qty))
    return getValidBaseTypeInfo(qty);

  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();
  if (mlir::Attribute typeDesc = metadataCache.lookup(ty))
    return typeDesc;

  // The helper may add new entries to the cache, so compute the type
  // descriptor before inserting it.
  mlir::Attribute typeDesc = getTypeInfoHelper(ty);
  return metadataCache[ty] = typeDesc;
}

TBAAAccessInfo CIRGenTBAA::getAccessInfo(QualType accessType) {
  // Pointee values may have incomplete types, but they shall never be
  // dereferenced.
  if (accessType->isIncompleteType())
    return TBAAAccessInfo::getIncompleteInfo();

  if (CodeGenUtils::typeHasMayAlias(accessType))
    return TBAAAccessInfo::getMayAliasInfo();

  uint64_t size = astContext.getTypeSizeInChars(accessType).getQuantity();
  return TBAAAccessInfo(getTypeInfo(accessType), size);
}

mlir::Attribute CIRGenTBAA::getBaseTypeInfoHelper(const clang::Type *ty) {
  const auto *rt = dyn_cast<RecordType>(ty);
  if (!rt)
    return {};

  const RecordDecl *rd = rt->getDecl()->getDefinition();
  const ASTRecordLayout &layout = astContext.getASTRecordLayout(rd);
  llvm::SmallVector<cir::TBAAMemberAttr, 4> fields;
  if (const auto *cxxrd = dyn_cast<CXXRecordDecl>(rd)) {
    // Handle C++ base classes. Non-virtual bases can treated a kind of
    // field. Virtual bases are more complex and omitted.
    for (const CXXBaseSpecifier &base : cxxrd->bases()) {
      if (base.isVirtual())
        continue;
      QualType baseQTy = base.getType();
      const CXXRecordDecl *baseRD = baseQTy->getAsCXXRecordDecl();
      if (baseRD->isEmpty())
        continue;
      mlir::Attribute typeDesc = CodeGenUtils::isValidTBAABaseType(baseQTy)
                                     ? getValidBaseTypeInfo(baseQTy)
                                     : getTypeInfo(baseQTy);
      if (!typeDesc)
        return {};
      uint64_t offset = layout.getBaseClassOffset(baseRD).getQuantity();
      fields.push_back(
          cir::TBAAMemberAttr::get(&mlirContext, typeDesc, offset));
    }
    // The order in which base class subobjects are allocated is unspecified,
    // so may differ from declaration order. In particular, Itanium ABI will
    // allocate a primary base first. Since we exclude empty subobjects, the
    // objects are not overlapping and their offsets are unique.
    llvm::sort(fields, [](cir::TBAAMemberAttr a, cir::TBAAMemberAttr b) {
      return a.getOffset() < b.getOffset();
    });
  }

  for (FieldDecl *field : rd->fields()) {
    if (field->isZeroSize(astContext) || field->isUnnamedBitField())
      continue;
    QualType fieldQTy = field->getType();
    mlir::Attribute typeDesc = CodeGenUtils::isValidTBAABaseType(fieldQTy)
                                   ? getValidBaseTypeInfo(fieldQTy)
                                   : getTypeInfo(fieldQTy);
    if (!typeDesc)
      return {};

    uint64_t bitOffset = layout.getFieldOffset(field->getFieldIndex());
    uint64_t offset = astContext.toCharUnitsFromBits(bitOffset).getQuantity();
    fields.push_back(cir::TBAAMemberAttr::get(&mlirContext, typeDesc, offset));
  }

  SmallString<256> outName;
  if (features.CPlusPlus) {
    // Don't use the mangler for C code.
    llvm::raw_svector_ostream out(outName);
    cxxABI.getMangleContext().mangleCanonicalTypeName(QualType(ty, 0), out);
  } else {
    outName = rd->getName();
  }

  assert(!cir::MissingFeatures::tbaaNewStructPath());
  return cir::TBAAStructAttr::get(&mlirContext, outName, fields);
}

mlir::Attribute CIRGenTBAA::getValidBaseTypeInfo(QualType qty) {
  assert(CodeGenUtils::isValidTBAABaseType(qty) && "Must be a valid base type");

  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();

  // A null attribute is a valid value in the cache, so use find rather than
  // lookup.
  auto it = baseTypeMetadataCache.find(ty);
  if (it != baseTypeMetadataCache.end())
    return it->second;

  // First calculate the type descriptor, before recomputing the insertion
  // point, as the helper can recursively call us.
  mlir::Attribute typeDesc = getBaseTypeInfoHelper(ty);
  [[maybe_unused]] auto inserted = baseTypeMetadataCache.insert({ty, typeDesc});
  assert(inserted.second && "BaseType metadata was already inserted");

  return typeDesc;
}

mlir::Attribute CIRGenTBAA::getBaseTypeInfo(QualType qty) {
  return CodeGenUtils::isValidTBAABaseType(qty) ? getValidBaseTypeInfo(qty)
                                                : mlir::Attribute();
}

cir::TBAATagAttr CIRGenTBAA::getAccessTagInfo(TBAAAccessInfo info) {
  assert(!info.isIncomplete() && "Access to an object of an incomplete type!");

  if (info.isMayAlias())
    info = TBAAAccessInfo(getChar(), info.size);

  // Only scalar accesses carry access tags.
  if (!info.accessType || mlir::isa<cir::TBAAStructAttr>(info.accessType))
    return {};

  if (!codeGenOpts.StructPathTBAA)
    info = TBAAAccessInfo(info.accessType, info.size);

  if (!info.baseType) {
    info.baseType = info.accessType;
    assert(!info.offset && "Nonzero offset for an access with no base type!");
  }

  assert(!cir::MissingFeatures::tbaaNewStructPath());
  return cir::TBAATagAttr::get(&mlirContext, info.baseType, info.accessType,
                               info.offset);
}

TBAAAccessInfo CIRGenTBAA::mergeTBAAInfoForCast(TBAAAccessInfo sourceInfo,
                                                TBAAAccessInfo targetInfo) {
  if (sourceInfo.isMayAlias() || targetInfo.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();
  return targetInfo;
}

TBAAAccessInfo
CIRGenTBAA::mergeTBAAInfoForConditionalOperator(TBAAAccessInfo infoA,
                                                TBAAAccessInfo infoB) {
  if (infoA == infoB)
    return infoA;

  if (!infoA || !infoB)
    return TBAAAccessInfo();

  if (infoA.isMayAlias() || infoB.isMayAlias())
    return TBAAAccessInfo::getMayAliasInfo();

  // TODO: Implement the rest of the logic here. For example, two accesses
  // with same final access types result in an access to an object of that final
  // access type regardless of their base types.
  return TBAAAccessInfo::getMayAliasInfo();
}
