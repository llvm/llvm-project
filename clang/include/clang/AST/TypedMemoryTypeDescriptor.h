#ifndef LLVM_CLANG_AST_TYPEDMEMORYTYPEDESCRIPTOR_H
#define LLVM_CLANG_AST_TYPEDMEMORYTYPEDESCRIPTOR_H

#include "clang/AST/TypedMemoryDescriptorBits.h"
#include "clang/AST/TypedMemoryLayoutBuilder.h"
#include "llvm/ADT/SmallSet.h"

namespace clang {

class TypedMemoryTypeDescriptorBase {
public:
  TypedMemoryTypeDescriptorBase(const TypedMemoryTypeDescriptorBase &) = delete;
  TypedMemoryTypeDescriptorBase &
  operator=(const TypedMemoryTypeDescriptorBase &) = delete;
  TypedMemoryTypeDescriptorBase() {}
  virtual ~TypedMemoryTypeDescriptorBase() = default;

  virtual bool initialize(const QualType &QT,
                          const TypedMemoryLayout &Layout) = 0;
};

class TypedMemoryTypeDescriptor : public TypedMemoryTypeDescriptorBase {
public:
  struct LayoutSemanticsSpan {
    uint64_t Offset;
    uint64_t Width;
    TypedMemoryLayoutSemantics Semantics;
  };

  using TypedMemoryTypeDescriptorBase::TypedMemoryTypeDescriptorBase;
  virtual ~TypedMemoryTypeDescriptor() = default;
  virtual bool initialize(const QualType &QT,
                          const TypedMemoryLayout &Layout) override;

  TypedMemoryLayoutSemantics getCoalescedLayoutSemantics() const {
    return LayoutSemantics;
  }
  TypedMemoryTypeFlags getTypeFlags() const { return Flags; }
  uint32_t computeTypeHash() const;

  static uint32_t computeTypeNameHash(const QualType &QT);

private:
  TypedMemoryTypeFlags Flags;
  llvm::SmallVector<LayoutSemanticsSpan> LayoutProperties;
  llvm::SmallVector<LayoutSemanticsSpan> CoalescedLayoutProperties;
  TypedMemoryLayoutSemantics LayoutSemantics;

  TypedMemoryLayoutSemantics
  getFieldSemantics(const TypedMemoryLayoutField *F) const;
  TypedMemoryLayoutSemantics getTypeSemantics(const QualType &QT) const;

  bool initializeLayoutProperties(const TypedMemoryLayout &Layout);
  void initializeCoalescedLayoutProperties(const TypedMemoryLayout &Layout);
};

} // namespace clang

#endif
