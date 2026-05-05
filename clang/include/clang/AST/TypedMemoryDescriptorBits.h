#ifndef LLVM_CLANG_AST_TYPEDMEMORYDESCRIPTORBITS_H
#define LLVM_CLANG_AST_TYPEDMEMORYDESCRIPTORBITS_H

#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"

namespace clang {

// Represent properties of (a region of) the memory layout of a type.
enum class TypedMemoryLayoutSemantics : uint16_t {
  None = 0,
  DataPointer = 1 << 0,
  StructPointer = 1 << 1,
  ImmutablePointer = 1 << 2,
  AnonymousPointer = 1 << 3,
  ReferenceCount = 1 << 4,
  ResourceHandle = 1 << 5,
  SpatialBounds = 1 << 6,
  TaintedData = 1 << 7,
  GenericData = 1 << 8,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ GenericData)
};

// Represent inherent properties of a type, which are not tied to a specific
// region of its memory layout, bur rather convey information about overall
// characteristics of the type itself.
enum class TypedMemoryTypeFlags : uint8_t {
  None = 0,
  IsPolymorphic = 1 << 0,
  HasMixedUnions = 1 << 1,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ HasMixedUnions)
};

enum class TypedMemoryTypeKind : uint8_t {
  KindC = 0,
  KindObjectiveC = 1,
  KindSwift = 2,
  KindCxx = 3
};

enum class TypedMemoryCallsiteFlags : uint8_t {
  None = 0,
  FixedSize = 1 << 0,
  Array = 1 << 1,
  HeaderPrefixedArray = 1 << 2,

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ HeaderPrefixedArray)
};

struct TypedMemorySummary {
  TypedMemorySummary()
      : LayoutSemantics(TypedMemoryLayoutSemantics::None),
        TypeFlags(TypedMemoryTypeFlags::None),
        TypeKind(TypedMemoryTypeKind::KindC),
        CallsiteFlags(TypedMemoryCallsiteFlags::None), Unused(0), Version(0) {}

  static constexpr uint32_t kLayoutSemanticsBits = 16;
  static constexpr uint8_t kTypeFlagsBits = 4;
  static constexpr uint8_t kTypeKindBits = 2;
  static constexpr uint8_t kCallsiteFlagBits = 4;
  static constexpr uint8_t kUnusedBits = 4;
  static constexpr uint8_t kVersionBits = 2;

  static constexpr uint8_t kVersionShift = 0;
  static constexpr uint8_t kUnusedShift = kVersionShift + kVersionBits;
  static constexpr uint8_t kCallsiteFlagsShift = kUnusedShift + kUnusedBits;
  static constexpr uint8_t kTypeKindShift =
      kCallsiteFlagsShift + kCallsiteFlagBits;
  static constexpr uint8_t kTypeFlagsShift = kTypeKindShift + kTypeKindBits;
  static constexpr uint8_t kLayoutSemanticsShift =
      kTypeFlagsShift + kTypeFlagsBits;

  static constexpr uint32_t kSummaryBitSize =
      kLayoutSemanticsShift + kLayoutSemanticsBits;
  static_assert(kSummaryBitSize == sizeof(uint32_t) * 8);

  TypedMemoryLayoutSemantics LayoutSemantics : kLayoutSemanticsBits;
  TypedMemoryTypeFlags TypeFlags : kTypeFlagsBits;
  TypedMemoryTypeKind TypeKind : kTypeKindBits;
  TypedMemoryCallsiteFlags CallsiteFlags : kCallsiteFlagBits;
  uint8_t Unused : kUnusedBits;
  uint8_t Version : kVersionBits;

  uint32_t value() const {
    const uint64_t LayoutSemantics =
        static_cast<uint64_t>(this->LayoutSemantics);
    const uint64_t TypeFlags = static_cast<uint64_t>(this->TypeFlags);
    const uint64_t TypeKind = static_cast<uint64_t>(this->TypeKind);
    const uint64_t CallsiteFlags = static_cast<uint64_t>(this->CallsiteFlags);
    const uint64_t Version = static_cast<uint64_t>(this->Version);
    // Manually construct this to avoid accidental ABI changes from
    // struct modification
    // [llllllllllllllll][tttt][kk][cccc][uuuu][vv]
    return (LayoutSemantics << kLayoutSemanticsShift) |
           (TypeFlags << kTypeFlagsShift) | (TypeKind << kTypeKindShift) |
           (CallsiteFlags << kCallsiteFlagsShift) | (Version << kVersionShift);
  }
} __attribute__((packed));

struct TypedMemoryDescriptorBits {
  TypedMemoryDescriptorBits() : Hash(0) {}

  TypedMemorySummary Summary;
  uint32_t Hash;

  static constexpr uint8_t kHashShift = 0;
  static constexpr uint8_t kHashBitSize = 32;
  static constexpr uint8_t kSummaryShift = kHashShift + kHashBitSize;
  static_assert(kSummaryShift + TypedMemorySummary::kSummaryBitSize == 64);

  uint64_t value() const {
    const uint64_t Summary = static_cast<uint64_t>(this->Summary.value());
    // [ssssssssssssssssssssssssssssssss][hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh]
    return (Summary << kSummaryShift) | (Hash << kHashShift);
  }
};

static_assert(sizeof(TypedMemorySummary) == sizeof(uint32_t),
              "Summary must be 32 bits");
static_assert(sizeof(TypedMemoryDescriptorBits) == sizeof(uint64_t),
              "Descriptor must be 64 bits");

} // namespace clang

#endif
