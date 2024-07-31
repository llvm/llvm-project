//===--- flang/unittests/Runtime/TemporaryStack.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"
#include "tools.h"
#include "flang/ISO_Fortran_binding_wrapper.h"
#include "flang/Runtime/allocatable.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/temporary-stack.h"
#include "flang/Runtime/type-code.h"
#include <vector>

using namespace Fortran::runtime;

// true if two descriptors are otherwise identical, except for different data
// pointers. The pointed-to elements are bit for bit identical.
static void descriptorAlmostEqual(
    const Descriptor &lhs, const Descriptor &rhs) {
  const Fortran::ISO::CFI_cdesc_t &lhsRaw = lhs.raw();
  const Fortran::ISO::CFI_cdesc_t &rhsRaw = rhs.raw();

  ASSERT_EQ(lhs.ElementBytes() == rhs.ElementBytes(), true);
  ASSERT_EQ(lhsRaw.version == rhsRaw.version, true);
  ASSERT_EQ(lhs.rank() == rhs.rank(), true);
  ASSERT_EQ(lhs.type() == rhs.type(), true);
  ASSERT_EQ(lhsRaw.attribute == rhsRaw.attribute, true);

  ASSERT_EQ(memcmp(lhsRaw.dim, rhsRaw.dim, lhs.rank()) == 0, true);
  const std::size_t bytes = lhs.Elements() * lhs.ElementBytes();
  ASSERT_EQ(memcmp(lhsRaw.base_addr, rhsRaw.base_addr, bytes) == 0, true);

  const DescriptorAddendum *lhsAdd = lhs.Addendum();
  const DescriptorAddendum *rhsAdd = rhs.Addendum();
  if (lhsAdd) {
    ASSERT_NE(rhsAdd, nullptr);
    ASSERT_EQ(lhsAdd->SizeInBytes() == rhsAdd->SizeInBytes(), true);
    ASSERT_EQ(memcmp(lhsAdd, rhsAdd, lhsAdd->SizeInBytes()) == 0, true);
  } else {
    ASSERT_EQ(rhsAdd, nullptr);
  }
}

TEST(TemporaryStack, ValueStackBasic) {
  const TypeCode code{CFI_type_int32_t};
  constexpr size_t elementBytes = 4;
  constexpr size_t rank = 2;
  void *const descriptorPtr = reinterpret_cast<void *>(0xdeadbeef);
  const SubscriptValue extent[rank]{42, 24};

  StaticDescriptor<rank> testDescriptorStorage[3];
  Descriptor &inputDesc{testDescriptorStorage[0].descriptor()};
  Descriptor &outputDesc{testDescriptorStorage[1].descriptor()};
  Descriptor &outputDesc2{testDescriptorStorage[2].descriptor()};
  inputDesc.Establish(code, elementBytes, descriptorPtr, rank, extent);

  inputDesc.Allocate();
  ASSERT_EQ(inputDesc.IsAllocated(), true);
  uint32_t *inputData = static_cast<uint32_t *>(inputDesc.raw().base_addr);
  for (std::size_t i = 0; i < inputDesc.Elements(); ++i) {
    inputData[i] = i;
  }

  void *storage = RTNAME(CreateValueStack)(__FILE__, __LINE__);
  ASSERT_NE(storage, nullptr);

  RTNAME(PushValue)(storage, inputDesc);

  RTNAME(ValueAt)(storage, 0, outputDesc);
  descriptorAlmostEqual(inputDesc, outputDesc);

  RTNAME(PopValue)(storage, outputDesc2);
  descriptorAlmostEqual(inputDesc, outputDesc2);

  RTNAME(DestroyValueStack)(storage);
}

static unsigned max(unsigned x, unsigned y) {
  if (x > y) {
    return x;
  }
  return y;
}

TEST(TemporaryStack, ValueStackMultiSize) {
  constexpr unsigned numToTest = 42;
  const TypeCode code{CFI_type_int32_t};
  constexpr size_t elementBytes = 4;
  SubscriptValue extent[CFI_MAX_RANK];

  std::vector<OwningPtr<Descriptor>> inputDescriptors;
  inputDescriptors.reserve(numToTest);

  void *storage = RTNAME(CreateValueStack)(__FILE__, __LINE__);
  ASSERT_NE(storage, nullptr);

  // create descriptors with and without adendums
  auto getAdendum = [](unsigned i) { return i % 2; };
  // create descriptors with varying ranks
  auto getRank = [](unsigned i) { return max(i % 8, 1); };

  // push descriptors of varying sizes and contents
  for (unsigned i = 0; i < numToTest; ++i) {
    const bool adendum = getAdendum(i);
    const size_t rank = getRank(i);
    for (unsigned dim = 0; dim < rank; ++dim) {
      extent[dim] = ((i + dim) % 8) + 1;
    }

    const OwningPtr<Descriptor> &desc =
        inputDescriptors.emplace_back(Descriptor::Create(code, elementBytes,
            nullptr, rank, extent, CFI_attribute_allocatable, adendum));

    // Descriptor::Establish doesn't initialise the extents if baseaddr is null
    for (unsigned dim = 0; dim < rank; ++dim) {
      Fortran::ISO::CFI_dim_t &boxDims = desc->raw().dim[dim];
      boxDims.lower_bound = 1;
      boxDims.extent = extent[dim];
      boxDims.sm = elementBytes;
    }
    desc->Allocate();

    // fill the array with some data to test
    for (uint32_t i = 0; i < desc->Elements(); ++i) {
      uint32_t *data = static_cast<uint32_t *>(desc->raw().base_addr);
      ASSERT_NE(data, nullptr);
      data[i] = i;
    }

    RTNAME(PushValue)(storage, *desc.get());
  }

  const TypeCode boolCode{CFI_type_Bool};
  // peek and test each descriptor
  for (unsigned i = 0; i < numToTest; ++i) {
    const OwningPtr<Descriptor> &input = inputDescriptors[i];
    const bool adendum = getAdendum(i);
    const size_t rank = getRank(i);

    // buffer to return the descriptor into
    OwningPtr<Descriptor> out = Descriptor::Create(
        boolCode, 1, nullptr, rank, extent, CFI_attribute_other, adendum);

    (void)input;
    RTNAME(ValueAt)(storage, i, *out.get());
    descriptorAlmostEqual(*input, *out);
  }

  // pop and test each descriptor
  for (unsigned i = numToTest; i > 0; --i) {
    const OwningPtr<Descriptor> &input = inputDescriptors[i - 1];
    const bool adendum = getAdendum(i - 1);
    const size_t rank = getRank(i - 1);

    // buffer to return the descriptor into
    OwningPtr<Descriptor> out = Descriptor::Create(
        boolCode, 1, nullptr, rank, extent, CFI_attribute_other, adendum);

    RTNAME(PopValue)(storage, *out.get());
    descriptorAlmostEqual(*input, *out);
  }

  RTNAME(DestroyValueStack)(storage);
}

TEST(TemporaryStack, DescriptorStackBasic) {
  const TypeCode code{CFI_type_Bool};
  constexpr size_t elementBytes = 4;
  constexpr size_t rank = 2;
  void *const descriptorPtr = reinterpret_cast<void *>(0xdeadbeef);
  const SubscriptValue extent[rank]{42, 24};

  StaticDescriptor<rank> testDescriptorStorage[3];
  Descriptor &inputDesc{testDescriptorStorage[0].descriptor()};
  Descriptor &outputDesc{testDescriptorStorage[1].descriptor()};
  Descriptor &outputDesc2{testDescriptorStorage[2].descriptor()};
  inputDesc.Establish(code, elementBytes, descriptorPtr, rank, extent);

  void *storage = RTNAME(CreateDescriptorStack)(__FILE__, __LINE__);
  ASSERT_NE(storage, nullptr);

  RTNAME(PushDescriptor)(storage, inputDesc);

  RTNAME(DescriptorAt)(storage, 0, outputDesc);
  ASSERT_EQ(
      memcmp(&inputDesc, &outputDesc, testDescriptorStorage[0].byteSize), 0);

  RTNAME(PopDescriptor)(storage, outputDesc2);
  ASSERT_EQ(
      memcmp(&inputDesc, &outputDesc2, testDescriptorStorage[0].byteSize), 0);

  RTNAME(DestroyDescriptorStack)(storage);
}

TEST(TemporaryStack, DescriptorStackMultiSize) {
  constexpr unsigned numToTest = 42;
  const TypeCode code{CFI_type_Bool};
  constexpr size_t elementBytes = 4;
  const uintptr_t ptrBase = 0xdeadbeef;
  SubscriptValue extent[CFI_MAX_RANK];

  std::vector<OwningPtr<Descriptor>> inputDescriptors;
  inputDescriptors.reserve(numToTest);

  void *storage = RTNAME(CreateDescriptorStack)(__FILE__, __LINE__);
  ASSERT_NE(storage, nullptr);

  // create descriptors with and without adendums
  auto getAdendum = [](unsigned i) { return i % 2; };
  // create descriptors with varying ranks
  auto getRank = [](unsigned i) { return max(i % CFI_MAX_RANK, 1); };

  // push descriptors of varying sizes and contents
  for (unsigned i = 0; i < numToTest; ++i) {
    const bool adendum = getAdendum(i);
    const size_t rank = getRank(i);
    for (unsigned dim = 0; dim < rank; ++dim) {
      extent[dim] = max(i - dim, 1);
    }

    // varying pointers
    void *const ptr = reinterpret_cast<void *>(ptrBase + i * elementBytes);

    const OwningPtr<Descriptor> &desc =
        inputDescriptors.emplace_back(Descriptor::Create(code, elementBytes,
            ptr, rank, extent, CFI_attribute_other, adendum));
    RTNAME(PushDescriptor)(storage, *desc.get());
  }

  const TypeCode intCode{CFI_type_int8_t};
  // peek and test each descriptor
  for (unsigned i = 0; i < numToTest; ++i) {
    const OwningPtr<Descriptor> &input = inputDescriptors[i];
    const bool adendum = getAdendum(i);
    const size_t rank = getRank(i);

    // buffer to return the descriptor into
    OwningPtr<Descriptor> out = Descriptor::Create(
        intCode, 1, nullptr, rank, extent, CFI_attribute_other, adendum);

    RTNAME(DescriptorAt)(storage, i, *out.get());
    ASSERT_EQ(memcmp(input.get(), out.get(), input->SizeInBytes()), 0);
  }

  // pop and test each descriptor
  for (unsigned i = numToTest; i > 0; --i) {
    const OwningPtr<Descriptor> &input = inputDescriptors[i - 1];
    const bool adendum = getAdendum(i - 1);
    const size_t rank = getRank(i - 1);

    // buffer to return the descriptor into
    OwningPtr<Descriptor> out = Descriptor::Create(
        intCode, 1, nullptr, rank, extent, CFI_attribute_other, adendum);

    RTNAME(PopDescriptor)(storage, *out.get());
    ASSERT_EQ(memcmp(input.get(), out.get(), input->SizeInBytes()), 0);
  }

  RTNAME(DestroyDescriptorStack)(storage);
}
