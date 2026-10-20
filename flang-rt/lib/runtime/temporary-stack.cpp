//===-- lib/runtime/temporary-stack.cpp -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements std::vector like storage for a dynamically resizable number of
// temporaries. For use in HLFIR lowering.

#include "flang/Runtime/temporary-stack.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/memory.h"
#include "flang-rt/runtime/terminator.h"
#include "flang/Common/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/assign.h"

RT_OFFLOAD_API_GROUP_BEGIN

namespace {

using namespace Fortran;
using namespace Fortran::runtime;

// the number of elements to allocate when first creating the vector
constexpr size_t INITIAL_ALLOC = 8;

/// To store C style data. Does not run constructors/destructors.
/// Not using std::vector to avoid linking the runtime library to stdc++
template <bool COPY_VALUES> class DescriptorStorage final {
  using size_type = uint64_t; // see checkedMultiply()

  size_type capacity_{0};
  size_type size_{0};
  Descriptor **data_{nullptr};
  Terminator terminator_;

  // return true on overflow
  static bool checkedMultiply(size_type x, size_type y, size_type &res);

  void resize(size_type newCapacity);

  Descriptor *cloneDescriptor(const Descriptor &source);

public:
  DescriptorStorage(const char *sourceFile, int line);
  ~DescriptorStorage();

  // `new` but using the runtime allocation API
  static inline DescriptorStorage *allocate(const char *sourceFile, int line) {
    Terminator term{sourceFile, line};
    void *ptr = AllocateMemoryOrCrash(term, sizeof(DescriptorStorage));
    return new (ptr) DescriptorStorage{sourceFile, line};
  }

  // `delete` but using the runtime allocation API
  static inline void destroy(DescriptorStorage *instance) {
    instance->~DescriptorStorage();
    FreeMemory(instance);
  }

  // clones a descriptor into this storage
  void push(const Descriptor &source);

  // out must be big enough to hold a descriptor of the right rank and addendum
  void pop(Descriptor &out);

  // out must be big enough to hold a descriptor of the right rank and addendum
  void at(size_type i, Descriptor &out);
};

using ValueStack = DescriptorStorage</*COPY_VALUES=*/true>;
using DescriptorStack = DescriptorStorage</*COPY_VALUES=*/false>;
} // namespace

template <bool COPY_VALUES>
bool DescriptorStorage<COPY_VALUES>::checkedMultiply(
    size_type x, size_type y, size_type &res) {
  // TODO: c++20 [[unlikely]]
  if (x > UINT64_MAX / y) {
    return true;
  }
  res = x * y;
  return false;
}

template <bool COPY_VALUES>
void DescriptorStorage<COPY_VALUES>::resize(size_type newCapacity) {
  if (newCapacity <= capacity_) {
    return;
  }
  size_type bytes;
  if (checkedMultiply(newCapacity, sizeof(Descriptor *), bytes)) {
    terminator_.Crash("temporary-stack: out of memory");
  }
  Descriptor **newData =
      static_cast<Descriptor **>(AllocateMemoryOrCrash(terminator_, bytes));
  // "memcpy" in glibc has a "nonnull" attribute on the source pointer.
  // Avoid passing a null pointer, since it would result in an undefined
  // behavior.
  if (data_ != nullptr) {
    runtime::memcpy(newData, data_, capacity_ * sizeof(Descriptor *));
    FreeMemory(data_);
  }
  data_ = newData;
  capacity_ = newCapacity;
}

template <bool COPY_VALUES>
Descriptor *DescriptorStorage<COPY_VALUES>::cloneDescriptor(
    const Descriptor &source) {
  const std::size_t bytes = source.SizeInBytes();
  void *memory = AllocateMemoryOrCrash(terminator_, bytes);
  Descriptor *desc = new (memory) Descriptor{source};
  return desc;
}

template <bool COPY_VALUES>
DescriptorStorage<COPY_VALUES>::DescriptorStorage(
    const char *sourceFile, int line)
    : terminator_{sourceFile, line} {
  resize(INITIAL_ALLOC);
}

template <bool COPY_VALUES>
DescriptorStorage<COPY_VALUES>::~DescriptorStorage() {
  for (size_type i = 0; i < size_; ++i) {
    Descriptor *element = data_[i];
    if constexpr (COPY_VALUES) {
      element->Destroy(false, true);
    }
    FreeMemory(element);
  }
  FreeMemory(data_);
}

template <bool COPY_VALUES>
void DescriptorStorage<COPY_VALUES>::push(const Descriptor &source) {
  if (size_ == capacity_) {
    size_type newSize;
    if (checkedMultiply(capacity_, 2, newSize)) {
      terminator_.Crash("temporary-stack: out of address space");
    }
    resize(newSize);
  }
  data_[size_] = cloneDescriptor(source);
  Descriptor &box = *data_[size_];
  size_ += 1;

  if constexpr (COPY_VALUES) {
    // copy the data pointed to by the box
    box.set_base_addr(nullptr);
    box.Allocate(kNoAsyncObject);
    RTNAME(AssignTemporary)
    (box, source, terminator_.sourceFileName(), terminator_.sourceLine());
  }
}

template <bool COPY_VALUES>
void DescriptorStorage<COPY_VALUES>::pop(Descriptor &out) {
  if (size_ == 0) {
    terminator_.Crash("temporary-stack: pop empty storage");
  }
  size_ -= 1;
  Descriptor *ptr = data_[size_];
  out = *ptr; // Descriptor::operator= handles the different sizes
  FreeMemory(ptr);
}

template <bool COPY_VALUES>
void DescriptorStorage<COPY_VALUES>::at(size_type i, Descriptor &out) {
  if (i >= size_) {
    terminator_.Crash("temporary-stack: out of bounds access");
  }
  Descriptor *ptr = data_[i];
  out = *ptr; // Descriptor::operator= handles the different sizes
}

inline static ValueStack *getValueStorage(void *opaquePtr) {
  return static_cast<ValueStack *>(opaquePtr);
}
inline static DescriptorStack *getDescriptorStorage(void *opaquePtr) {
  return static_cast<DescriptorStack *>(opaquePtr);
}

RT_OFFLOAD_API_GROUP_END

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN
void *RTNAME(CreateValueStack)(const char *sourceFile, int line) {
  return ValueStack::allocate(sourceFile, line);
}

void RTNAME(PushValue)(void *opaquePtr, const Descriptor &value) {
  getValueStorage(opaquePtr)->push(value);
}

void RTNAME(PopValue)(void *opaquePtr, Descriptor &value) {
  getValueStorage(opaquePtr)->pop(value);
}

void RTNAME(ValueAt)(void *opaquePtr, uint64_t i, Descriptor &value) {
  getValueStorage(opaquePtr)->at(i, value);
}

void RTNAME(DestroyValueStack)(void *opaquePtr) {
  ValueStack::destroy(getValueStorage(opaquePtr));
}

void *RTNAME(CreateDescriptorStack)(const char *sourceFile, int line) {
  return DescriptorStack::allocate(sourceFile, line);
}

void RTNAME(PushDescriptor)(void *opaquePtr, const Descriptor &value) {
  getDescriptorStorage(opaquePtr)->push(value);
}

void RTNAME(PopDescriptor)(void *opaquePtr, Descriptor &value) {
  getDescriptorStorage(opaquePtr)->pop(value);
}

void RTNAME(DescriptorAt)(void *opaquePtr, uint64_t i, Descriptor &value) {
  getValueStorage(opaquePtr)->at(i, value);
}

void RTNAME(DestroyDescriptorStack)(void *opaquePtr) {
  DescriptorStack::destroy(getDescriptorStorage(opaquePtr));
}
RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
