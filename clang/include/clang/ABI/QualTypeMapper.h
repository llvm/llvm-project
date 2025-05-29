#ifndef LLVM_ABI_QUALTYPE_MAPPER_H
#define LLVM_ABI_QUALTYPE_MAPPER_H

#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Allocator.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Decl.h>
#include <clang/AST/Type.h>
#include <llvm/ABI/Types.h>
#include <llvm/ADT/DenseMap.h>

// Specialization for QualType
template <> struct llvm::DenseMapInfo<clang::QualType> {
  static inline clang::QualType getEmptyKey() {
    return clang::QualType::getFromOpaquePtr(
        reinterpret_cast<clang::Type *>(-1));
  }

  static inline clang::QualType getTombstoneKey() {
    return clang::QualType::getFromOpaquePtr(
        reinterpret_cast<clang::Type *>(-2));
  }

  static unsigned getHashValue(const clang::QualType &Val) {
    return (unsigned)((uintptr_t)Val.getAsOpaquePtr()) ^
           ((unsigned)((uintptr_t)Val.getAsOpaquePtr() >> 9));
  }

  static bool isEqual(const clang::QualType &LHS, const clang::QualType &RHS) {
    return LHS == RHS;
  }
};

namespace clang {
namespace mapper {

class QualTypeMapper {
private:
  clang::ASTContext &ASTCtx;
  llvm::abi::TypeBuilder Builder;

  llvm::DenseMap<clang::QualType, const llvm::abi::Type *> TypeCache;

  const llvm::abi::Type *convertBuiltinType(const clang::BuiltinType *BT);
  const llvm::abi::Type *convertPointerType(const clang::PointerType *PT);
  const llvm::abi::Type *convertArrayType(const clang::ArrayType *AT);
  const llvm::abi::Type *convertVectorType(const clang::VectorType *VT);
  const llvm::abi::Type *convertRecordType(const clang::RecordType *RT);
  const llvm::abi::Type *convertEnumType(const clang::EnumType *ET);

  const llvm::abi::StructType *convertStructType(const clang::RecordDecl *RD);
  const llvm::abi::UnionType *convertUnionType(const clang::RecordDecl *RD);
  const llvm::abi::Type *createPointerTypeForPointee(QualType PointeeType);

  void computeFieldInfo(const clang::RecordDecl *RD,
                        SmallVectorImpl<llvm::abi::FieldInfo> &Fields,
                        const clang::ASTRecordLayout &Layout);

  llvm::TypeSize getTypeSize(clang::QualType QT) const;
  llvm::Align getTypeAlign(clang::QualType QT) const;
  uint64_t getPointerSize() const;
  uint64_t getPointerAlign() const;

public:
  explicit QualTypeMapper(clang::ASTContext &Ctx, llvm::BumpPtrAllocator &Alloc)
      : ASTCtx(Ctx), Builder(Alloc) {}

  const llvm::abi::Type *convertType(clang::QualType QT);

  void clearCache() { TypeCache.clear(); }

  llvm::abi::TypeBuilder getTypeBuilder() { return Builder; }
};

} // namespace mapper
} // namespace clang

#endif // !LLVM_ABI_QUALTYPE_MAPPER_H
