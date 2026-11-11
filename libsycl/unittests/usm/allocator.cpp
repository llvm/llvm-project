//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/unittests_helper.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/usm_allocator.hpp>

#include <detail/context_impl.hpp>
#include <detail/device_impl.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <list>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

using namespace sycl;
using namespace ::testing;

namespace {

struct USMAllocatorTest : public Test {
protected:
  // The default mock allocations are dummy handles that cannot be written to.
  USMAllocatorTest() {
    mock::MockLiboffload &Mock = Helper.Mock.get();
    ON_CALL(Mock, olMemAllocAlignedHost)
        .WillByDefault([this](ol_context_handle_t, ol_device_handle_t,
                              size_t Size, size_t Alignment,
                              void **AllocationOut) -> ol_result_t {
          *AllocationOut = allocateHostMemory(Size, Alignment);
          return OL_SUCCESS;
        });
    ON_CALL(Mock, olMemAllocAligned)
        .WillByDefault([this](ol_context_handle_t, ol_device_handle_t,
                              ol_alloc_type_t, size_t Size, size_t Alignment,
                              void **AllocationOut) -> ol_result_t {
          *AllocationOut = allocateHostMemory(Size, Alignment);
          return OL_SUCCESS;
        });
    ON_CALL(Mock, olMemFree)
        .WillByDefault(
            [this](ol_context_handle_t, void *Address) -> ol_result_t {
              EXPECT_EQ(LiveAllocations.erase(Address), 1u)
                  << "Freeing memory that was not allocated by the mock";
              std::free(Address);
              return OL_SUCCESS;
            });
  }

  void TearDown() override { EXPECT_THAT(LiveAllocations, IsEmpty()); }

  void *allocateHostMemory(size_t Size, size_t Alignment) {
    // std::aligned_alloc requires the size to be a multiple of the alignment.
    void *Ptr = std::aligned_alloc(Alignment, (Size + Alignment - 1) /
                                                  Alignment * Alignment);
    LiveAllocations.insert(Ptr);
    return Ptr;
  }

  ol_device_handle_t getOLDevice() const {
    return detail::getSyclObjImpl(Q.get_device())->getOLHandle();
  }

  ol_context_handle_t getOLContext() const {
    return detail::getSyclObjImpl(Q.get_context())->getOLHandleRef();
  }

  void expectAllocation(usm::alloc Kind, size_t Size, size_t Alignment) {
    if (Kind == usm::alloc::host)
      EXPECT_CALL(Helper.Mock.get(),
                  olMemAllocAlignedHost(_, getOLDevice(), Size, Alignment, _))
          .WillOnce(DoDefault());
    else
      EXPECT_CALL(Helper.Mock.get(),
                  olMemAllocAligned(_, getOLDevice(), OL_ALLOC_TYPE_MANAGED,
                                    Size, Alignment, _))
          .WillOnce(DoDefault());
  }

  // Keeps satisfied expectations from matching later allocations.
  void verifyAndClearExpectations() {
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&Helper.Mock.get()));
  }

  template <typename T, usm::alloc Kind, size_t Alignment = 0>
  void checkDirectAllocation() {
    usm_allocator<T, Kind, Alignment> Alloc(Q);
    constexpr size_t ExpectedAlignment = std::max(Alignment, alignof(T));

    expectAllocation(Kind, Count * sizeof(T), ExpectedAlignment);
    T *Ptr = Alloc.allocate(Count);
    ASSERT_NE(Ptr, nullptr);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(Ptr) % ExpectedAlignment, 0u);

    EXPECT_CALL(Helper.Mock.get(), olMemFree(getOLContext(), Ptr))
        .WillOnce(DoDefault());
    Alloc.deallocate(Ptr, Count);
    verifyAndClearExpectations();
  }

  template <typename T, usm::alloc Kind, size_t Alignment = 0>
  void checkVectorAllocation() {
    usm_allocator<T, Kind, Alignment> Alloc(Q);
    std::vector<T, decltype(Alloc)> Vec(Alloc);
    for (size_t I = 0; I < Count; ++I)
      Vec.push_back(T{});

    EXPECT_THAT(LiveAllocations, Contains(static_cast<void *>(Vec.data())));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(Vec.data()) % alignof(T), 0u);
  }

protected:
  static constexpr size_t Count = 5;

  // Declared first so it outlives the mock actions that capture it.
  std::set<void *> LiveAllocations;
  unittests::UnittestsHelper Helper;
  queue Q;
};

TEST_F(USMAllocatorTest, OneByteElement) {
  checkDirectAllocation<uint8_t, usm::alloc::host>();
  checkDirectAllocation<uint8_t, usm::alloc::shared>();
  checkVectorAllocation<uint8_t, usm::alloc::host>();
  checkVectorAllocation<uint8_t, usm::alloc::shared>();
}

TEST_F(USMAllocatorTest, EightByteElement) {
  checkDirectAllocation<uint64_t, usm::alloc::host>();
  checkDirectAllocation<uint64_t, usm::alloc::shared>();
  checkVectorAllocation<uint64_t, usm::alloc::host>();
  checkVectorAllocation<uint64_t, usm::alloc::shared>();
}

struct alignas(64) OverAligned {
  int Value;
};

TEST_F(USMAllocatorTest, OverAlignedElement) {
  checkDirectAllocation<OverAligned, usm::alloc::host>();
  checkDirectAllocation<OverAligned, usm::alloc::shared>();
  checkVectorAllocation<OverAligned, usm::alloc::host>();
  checkVectorAllocation<OverAligned, usm::alloc::shared>();

  constexpr size_t SmallAlignment = 16;
  constexpr size_t LargeAlignment = 256;
  checkDirectAllocation<OverAligned, usm::alloc::host, SmallAlignment>();
  checkDirectAllocation<OverAligned, usm::alloc::shared, SmallAlignment>();
  checkDirectAllocation<OverAligned, usm::alloc::host, LargeAlignment>();
  checkDirectAllocation<OverAligned, usm::alloc::shared, LargeAlignment>();
  checkVectorAllocation<OverAligned, usm::alloc::host, SmallAlignment>();
  checkVectorAllocation<OverAligned, usm::alloc::shared, SmallAlignment>();
  checkVectorAllocation<OverAligned, usm::alloc::host, LargeAlignment>();
  checkVectorAllocation<OverAligned, usm::alloc::shared, LargeAlignment>();
}

TEST_F(USMAllocatorTest, ThrowException) {
  mock::MockLiboffload &Mock = Helper.Mock.get();
  EXPECT_CALL(Mock, olMemAllocAlignedHost(_, _, _, _, _))
      .WillRepeatedly(Return(Mock.makeEmptyStrError(OL_ERRC_OUT_OF_RESOURCES)));
  EXPECT_CALL(Mock, olMemAllocAligned(_, _, _, _, _, _))
      .WillRepeatedly(Return(Mock.makeEmptyStrError(OL_ERRC_OUT_OF_RESOURCES)));
  EXPECT_CALL(Mock, olMemFree(_, _)).Times(0);

  auto ExpectMemoryAllocationError = [this](auto &&Allocate) {
    try {
      Allocate();
      FAIL() << "Expected sycl::exception";
    } catch (const sycl::exception &E) {
      EXPECT_EQ(E.code(), make_error_code(errc::memory_allocation));
      ASSERT_TRUE(E.has_context());
      EXPECT_EQ(E.get_context(), Q.get_context());
    }
  };

  usm_allocator<uint64_t, usm::alloc::host> HostAlloc(Q);
  usm_allocator<uint64_t, usm::alloc::shared> SharedAlloc(Q);
  ExpectMemoryAllocationError([&] { HostAlloc.allocate(Count); });
  ExpectMemoryAllocationError([&] { SharedAlloc.allocate(Count); });

  std::vector<uint64_t, decltype(HostAlloc)> HostVec(HostAlloc);
  std::vector<uint64_t, decltype(SharedAlloc)> SharedVec(SharedAlloc);
  ExpectMemoryAllocationError([&] { HostVec.push_back(1); });
  ExpectMemoryAllocationError([&] { SharedVec.push_back(1); });
}

TEST_F(USMAllocatorTest, RebindContainer) {
  using ByteAlloc = usm_allocator<uint8_t, usm::alloc::host>;
  using FloatAlloc = std::allocator_traits<ByteAlloc>::rebind_alloc<float>;
  static_assert(
      std::is_same_v<FloatAlloc, usm_allocator<float, usm::alloc::host>>);

  ByteAlloc Bytes(Q);
  FloatAlloc Floats(Bytes);
  EXPECT_TRUE(Floats == Bytes);

  expectAllocation(usm::alloc::host, Count * sizeof(float), alignof(float));
  float *Ptr = Floats.allocate(Count);
  ASSERT_NE(Ptr, nullptr);
  EXPECT_CALL(Helper.Mock.get(), olMemFree(getOLContext(), Ptr))
      .WillOnce(DoDefault());
  Floats.deallocate(Ptr, Count);
  verifyAndClearExpectations();

  usm_allocator<uint64_t, usm::alloc::shared> ListAlloc(Q);
  std::list<uint64_t, decltype(ListAlloc)> List(ListAlloc);
  for (size_t I = 0; I < Count; ++I)
    List.push_back(I);

  EXPECT_THAT(LiveAllocations, SizeIs(Count));
}

TEST_F(USMAllocatorTest, Equality) {
  usm_allocator<int, usm::alloc::host> Host(Q);
  usm_allocator<int, usm::alloc::host> HostCopy(Host);
  usm_allocator<int, usm::alloc::host> HostFromContext(Q.get_context(),
                                                       Q.get_device());
  EXPECT_TRUE(Host == HostCopy);
  EXPECT_TRUE(Host == HostFromContext);

  usm_allocator<int, usm::alloc::shared> Shared(Q);
  EXPECT_TRUE(Host != Shared);

  usm_allocator<int, usm::alloc::host, 64> Aligned(Q);
  EXPECT_TRUE(Host != Aligned);

  context OtherContext(Q.get_device());
  ASSERT_NE(OtherContext, Q.get_context());
  usm_allocator<int, usm::alloc::host> HostOtherContext(OtherContext,
                                                        Q.get_device());
  EXPECT_TRUE(Host != HostOtherContext);
}

TEST_F(USMAllocatorTest, Propagation) {
  using Alloc = usm_allocator<int, usm::alloc::shared>;
  using Traits = std::allocator_traits<Alloc>;
  static_assert(Traits::propagate_on_container_copy_assignment::value);
  static_assert(Traits::propagate_on_container_move_assignment::value);
  static_assert(Traits::propagate_on_container_swap::value);

  context OtherContext(Q.get_device());
  Alloc AllocA(Q);
  Alloc AllocB(OtherContext, Q.get_device());
  ASSERT_TRUE(AllocA != AllocB);

  std::vector<int, Alloc> Dst({1, 2, 3}, AllocA);
  std::vector<int, Alloc> Src({4, 5}, AllocB);
  int *SrcData = Src.data();
  Dst = std::move(Src);
  EXPECT_TRUE(Dst.get_allocator() == AllocB);
  EXPECT_EQ(Dst.data(), SrcData);
}

TEST_F(USMAllocatorTest, Properties) {
  using property::queue::in_order;

  usm_allocator<int, usm::alloc::host> Alloc(Q);
  EXPECT_FALSE(Alloc.has_property<in_order>());
  try {
    Alloc.get_property<in_order>();
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
  }

  usm_allocator<int, usm::alloc::host> InOrderAlloc(Q, {in_order{}});
  EXPECT_TRUE(InOrderAlloc.has_property<in_order>());
  EXPECT_NO_THROW(InOrderAlloc.get_property<in_order>());
}

TEST_F(USMAllocatorTest, ZeroCount) {
  mock::MockLiboffload &Mock = Helper.Mock.get();
  EXPECT_CALL(Mock, olMemAllocAlignedHost(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock, olMemAllocAligned(_, _, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock, olMemFree(_, _)).Times(0);

  usm_allocator<int, usm::alloc::host> HostAlloc(Q);
  usm_allocator<int, usm::alloc::shared> SharedAlloc(Q);
  EXPECT_EQ(HostAlloc.allocate(0), nullptr);
  EXPECT_EQ(SharedAlloc.allocate(0), nullptr);
  HostAlloc.deallocate(nullptr, 0);
  SharedAlloc.deallocate(nullptr, 0);
}

} // namespace
