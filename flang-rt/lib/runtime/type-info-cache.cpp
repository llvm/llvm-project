//===-- lib/runtime/type-info-cache.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/type-info-cache.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"

#ifndef FLANG_RT_PDT_CACHE_MAX_LOAD_FACTOR
#define FLANG_RT_PDT_CACHE_MAX_LOAD_FACTOR 2
#endif

#ifndef FLANG_RT_PDT_CACHE_INITIAL_BUCKET_CNT
#define FLANG_RT_PDT_CACHE_INITIAL_BUCKET_CNT 31
#endif

namespace Fortran::runtime::typeInfo {

#ifdef RT_DEVICE_COMPILATION

// Device stub: PDT LEN parameter instantiation not supported on GPU
RT_API_ATTRS const DerivedType *GetConcreteType(const DerivedType &genericType,
    const Descriptor &instance, runtime::Terminator &terminator) {
  std::size_t numLenParams = genericType.LenParameters();
  // Types without LEN params or already-concrete types can pass through
  const DerivedType *uninst = genericType.uninstantiatedType();
  if ((numLenParams == 0) || (uninst != nullptr && uninst != &genericType)) {
    return &genericType;
  }
  // Cannot instantiate PDT with LEN parameters on device
  terminator.Crash(
      "PDT LEN parameter instantiation not supported in device code");
  return nullptr;
}

#else // !RT_DEVICE_COMPILATION

#include <cstdlib>
#include <cstring>
#include <functional>

// using Fortran::runtime::FreeMemory;
using Fortran::runtime::New;
using Fortran::runtime::OwningPtr;
using Fortran::runtime::Terminator;

namespace {

// Hash table entry for concrete type cache
struct CacheEntry {
  std::uint64_t hash;
  DerivedType *concreteType;
  OwningPtr<CacheEntry> next{nullptr};

  explicit CacheEntry(std::uint64_t h, DerivedType *type)
      : hash{h}, concreteType{type} {}
};

// Dynamically resizing hash table using malloc/free (no C++ allocators)
class ConcreteTypeCache {
public:
  ConcreteTypeCache() {
    numBuckets_ = initialBuckets_;
    buckets_ = static_cast<CacheEntry **>(
        std::calloc(numBuckets_, sizeof(CacheEntry *)));
  }

  ~ConcreteTypeCache() {
    if (buckets_) {
      // Free all chains
      for (std::size_t i = 0; i < numBuckets_; ++i) {
        FreeBucketChain(buckets_[i]);
      }
      std::free(buckets_);
    }
  }

  DerivedType *Find(std::uint64_t hash) {
    if (!buckets_) {
      return nullptr;
    }
    std::size_t index = hash % numBuckets_;
    for (CacheEntry *entry = buckets_[index]; entry;
        entry = entry->next.get()) {
      if (entry->hash == hash) {
        return entry->concreteType;
      }
    }
    return nullptr;
  }

  void Insert(
      std::uint64_t hash, DerivedType *type, const Terminator &terminator) {
    if (!buckets_) {
      return;
    }

    // Check if we need to resize
    if (numEntries_ >= numBuckets_ * maxLoadFactor_) {
      Resize(terminator);
    }

    std::size_t index = hash % numBuckets_;
    CacheEntry *newEntry = New<CacheEntry>{terminator}(hash, type).release();
    newEntry->next.reset(buckets_[index]);
    buckets_[index] = newEntry;
    ++numEntries_;
  }

private:
  static void FreeBucketChain(CacheEntry *head) {
    while (head) {
      CacheEntry *next = head->next.release();
      Fortran::runtime::FreeMemory(head);
      head = next;
    }
  }

  void Resize(const Terminator &terminator) {
    std::size_t oldNumBuckets = numBuckets_;
    CacheEntry **oldBuckets = buckets_;

    // Double the bucket count
    numBuckets_ = oldNumBuckets * 2;
    buckets_ = static_cast<CacheEntry **>(
        std::calloc(numBuckets_, sizeof(CacheEntry *)));

    if (!buckets_) {
      // Allocation failed - restore old state
      buckets_ = oldBuckets;
      numBuckets_ = oldNumBuckets;
      return;
    }

    // Rehash all entries from old buckets
    for (std::size_t i = 0; i < oldNumBuckets; ++i) {
      CacheEntry *entry = oldBuckets[i];
      while (entry) {
        CacheEntry *next = entry->next.release();

        // Insert into new bucket
        std::size_t newIndex = entry->hash % numBuckets_;
        entry->next.reset(buckets_[newIndex]);
        buckets_[newIndex] = entry;

        entry = next;
      }
    }

    // Clean up old bucket array (entries already moved)
    std::free(oldBuckets);
  }

  static constexpr std::size_t initialBuckets_{
      FLANG_RT_PDT_CACHE_INITIAL_BUCKET_CNT};
  static constexpr std::size_t maxLoadFactor_{
      FLANG_RT_PDT_CACHE_MAX_LOAD_FACTOR};

  CacheEntry **buckets_{nullptr};
  std::size_t numBuckets_{0};
  std::size_t numEntries_{0};
};

static ConcreteTypeCache concreteTypeCache;

// Compute hash from generic type pointer and LEN parameter values
// using a hash combining formula based on Boost's hash_combine.
static std::uint64_t ComputeConcreteTypeHash(const DerivedType &genericType,
    const DescriptorAddendum &addendum, std::size_t numLenParams) {
  std::uint64_t hash = reinterpret_cast<std::uintptr_t>(&genericType);
  for (std::size_t i = 0; i < numLenParams; ++i) {
    TypeParameterValue v = addendum.LenParameterValue(i);
    hash ^= std::hash<TypeParameterValue>{}(v) + 0x9e3779b9 + (hash << 6) +
        (hash >> 2);
  }
  return hash;
}

} // anonymous namespace

// Compute allocation size for a concrete type: DerivedType + Component array
static std::size_t ComputeConcreteTypeAllocationSize(
    const DerivedType &generic) {
  std::size_t numComponents = generic.component().Elements();
  // Ensure Component array is properly aligned after DerivedType
  static_assert(sizeof(DerivedType) % alignof(Component) == 0,
      "DerivedType size must be aligned for trailing Component array");
  return sizeof(DerivedType) + numComponents * sizeof(Component);
}

// Copy generic type to concrete, setting up the Component array pointer
static void CopyGenericToConcreteType(DerivedType *concrete,
    const DerivedType &generic, Component *concreteComponents,
    std::size_t numComponents) {
  // NOTE: DerivedType has a user-declared destructor, so it is not trivially
  // copyable. We still rely on this shallow copy because its members are
  // descriptor wrappers/pointers to shared immutable metadata (bindings, name,
  // kindParameter, lenParameterKind, procPtr, special).
  std::memcpy(static_cast<void *>(concrete), &generic, sizeof(DerivedType));

  // Copy the Component array
  const Descriptor &genericCompDesc = generic.component();
  const Component *genericComponents =
      genericCompDesc.OffsetElement<const Component>();
  std::memcpy(
      concreteComponents, genericComponents, numComponents * sizeof(Component));

  concrete->SetComponentBaseAddr(concreteComponents);
  concrete->SetUninstantiatedType(&generic);
}

// Get alignment requirement for a component (stored by compiler)
static std::size_t GetComponentAlignment(const Component &comp) {
  return comp.alignment();
}

// Resolve LEN-dependent offsets in the concrete type's Component array
static std::size_t ResolveComponentOffsets(Component *components,
    std::size_t numComponents, const Descriptor &instance,
    runtime::Terminator &terminator) {
  std::size_t currentOffset = 0;
  std::size_t maxAlignment = 1;

  for (std::size_t j = 0; j < numComponents; ++j) {
    Component &comp = components[j];

    // Use existing method to compute element byte size
    std::size_t elementBytes = comp.GetElementByteSize(instance);
    std::size_t alignment = GetComponentAlignment(comp);

    // Compute element count for array components
    std::size_t numElements = 1;
    if (int rank = comp.rank(); rank > 0) {
      if (const Value *boundValues = comp.bounds()) {
        for (int dim = 0; dim < rank; ++dim) {
          auto lb = boundValues[2 * dim].GetValue(&instance);
          auto ub = boundValues[2 * dim + 1].GetValue(&instance);
          if (lb.has_value() && ub.has_value() && *ub >= *lb) {
            numElements *= (*ub - *lb + 1);
          } else {
            numElements = 0;
            break;
          }
        }
      }
    }

    // Compute component size based on genre
    std::size_t componentSize = 0;

    if (comp.genre() != Component::Genre::Data) {
      // Non-Data genres (Allocatable, Pointer, Automatic): store a Descriptor
      const DerivedType *derivedComp = comp.derivedType();
      componentSize = Descriptor::SizeInBytes(
          comp.rank(), true, derivedComp ? derivedComp->LenParameters() : 0);
      alignment = alignof(Descriptor);
    } else if (const DerivedType *nestedType{comp.derivedType()};
        comp.category() == TypeCategory::Derived && nestedType &&
        nestedType->LenParameters() > 0) {
      // Nested PDT with LEN params: resolve concrete type depth-first
      const Value *lenValues = comp.lenValue();
      RUNTIME_CHECK(terminator, lenValues != nullptr);

      OwningPtr<Descriptor> tempDesc{Descriptor::Create(
          *nestedType, nullptr, 0, nullptr, CFI_attribute_other)};
      DescriptorAddendum *tempAddendum = tempDesc->Addendum();
      RUNTIME_CHECK(terminator, tempAddendum != nullptr);

      std::size_t nestedLenParams = nestedType->LenParameters();
      for (std::size_t i = 0; i < nestedLenParams; ++i) {
        auto value = lenValues[i].GetValue(&instance);
        RUNTIME_CHECK(terminator, value.has_value());
        tempAddendum->SetLenParameterValue(i, *value);
      }

      const DerivedType *nestedConcrete =
          GetConcreteType(*nestedType, *tempDesc, terminator);
      elementBytes = nestedConcrete->sizeInBytes();

      comp.SetDerivedType(nestedConcrete);
      componentSize = elementBytes * numElements;
    } else {
      // Data genre: intrinsic, non-PDT derived, or KIND-only PDT
      componentSize = elementBytes * numElements;
    }

    // Apply alignment; round up to the next multiple, if needed
    if (alignment > 1) {
      currentOffset = (currentOffset + alignment - 1) & ~(alignment - 1);
    }

    comp.SetOffset(currentOffset);

    currentOffset += componentSize;
    if (alignment > maxAlignment) {
      maxAlignment = alignment;
    }
  }

  // Final structure alignment
  if (maxAlignment > 1) {
    currentOffset = (currentOffset + maxAlignment - 1) & ~(maxAlignment - 1);
  }

  return currentOffset; // This becomes sizeInBytes_
}

// Create a new concrete type from a generic type and specific LEN values
static DerivedType *CreateConcreteType(const DerivedType &generic,
    const Descriptor &instance, runtime::Terminator &terminator) {
  std::size_t numComponents = generic.component().Elements();
  std::size_t allocSize = ComputeConcreteTypeAllocationSize(generic);

  void *memory = std::calloc(allocSize, 1);
  RUNTIME_CHECK(terminator, memory != nullptr);

  DerivedType *concrete = static_cast<DerivedType *>(memory);
  Component *components = reinterpret_cast<Component *>(
      static_cast<char *>(memory) + sizeof(DerivedType));

  CopyGenericToConcreteType(concrete, generic, components, numComponents);

  std::size_t sizeInBytes =
      ResolveComponentOffsets(components, numComponents, instance, terminator);

  concrete->SetSizeInBytes(sizeInBytes);

  return concrete;
}

RT_API_ATTRS const DerivedType *GetConcreteType(const DerivedType &genericType,
    const Descriptor &instance, runtime::Terminator &terminator) {
  std::size_t numLenParams = genericType.LenParameters();
  // Concrete types have uninstantiatedType_ pointing to the original generic.
  // Generic types with LEN params have uninstantiatedType_ == nullptr.
  // Generic types without LEN params are caught by the numLenParams == 0 check.
  const DerivedType *uninst = genericType.uninstantiatedType();
  if ((numLenParams == 0) || (uninst != nullptr && uninst != &genericType)) {
    return &genericType; // Already concrete or no LEN params
  }

  // Check that instance has addendum with LEN values
  const DescriptorAddendum *addendum = instance.Addendum();
  RUNTIME_CHECK(terminator, addendum != nullptr);

  // Compute hash directly from LEN values (no heap allocation)
  std::uint64_t hash =
      ComputeConcreteTypeHash(genericType, *addendum, numLenParams);

  // Check cache
  DerivedType *cached = concreteTypeCache.Find(hash);
  if (cached) {
    return cached; // Cache hit
  }

  // Cache miss: create new concrete type
  DerivedType *concrete = CreateConcreteType(genericType, instance, terminator);

  // Insert into cache
  concreteTypeCache.Insert(hash, concrete, terminator);

  return concrete;
}

#endif // RT_DEVICE_COMPILATION

} // namespace Fortran::runtime::typeInfo
