#include "CIRGenTBAA.h"
#include "CIRGenTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecordLayout.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/ErrorHandling.h"
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
  return cir::TBAAScalarAttr::get(mlirContext,
                                  cir::IntType::get(mlirContext, 1, true));
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
    // TODO(cir): support TBAA with struct
    return tbaa_NYI(mlirContext);
  }

  const clang::Type *ty = astContext.getCanonicalType(qty).getTypePtr();
  if (metadataCache.contains(ty)) {
    return metadataCache[ty];
  }

  // Note that the following helper call is allowed to add new nodes to the
  // cache, which invalidates all its previously obtained iterators. So we
  // first generate the node for the type and then add that node to the
  // cache.
  auto typeNode = cir::TBAAScalarAttr::get(mlirContext, types.ConvertType(qty));
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
  // TODO(cir): support vtable ptr
  assert(!cir::MissingFeatures::tbaaVTablePtr());
  return TBAAAccessInfo();
}

mlir::ArrayAttr CIRGenTBAA::getTBAAStructInfo(clang::QualType qty) {
  assert(!cir::MissingFeatures::tbaaStruct() && "tbaa.struct NYI");
  return mlir::ArrayAttr();
}

cir::TBAAAttr CIRGenTBAA::getBaseTypeInfo(clang::QualType qty) {
  return tbaa_NYI(mlirContext);
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
    llvm_unreachable("NYI");
  }
  if (tbaaInfo.baseType == tbaaInfo.accessType) {
    return tbaaInfo.accessType;
  }
  return tbaa_NYI(mlirContext);
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
