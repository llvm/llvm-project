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
#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <new>
#include <tuple>
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
              freeHostMemory(Address);
              return OL_SUCCESS;
            });
  }

  void TearDown() override { EXPECT_THAT(LiveAllocations, IsEmpty()); }

  void *allocateHostMemory(size_t Size, size_t Alignment) {
    std::align_val_t Align{Alignment};
    void *Ptr = ::operator new(Size, Align);
    LiveAllocations.emplace(Ptr, Align);
    return Ptr;
  }

  void freeHostMemory(void *Ptr) {
    auto It = LiveAllocations.find(Ptr);
    ASSERT_TRUE(It != LiveAllocations.end())
        << "Freeing memory that was not allocated by the mock";
    ::operator delete(Ptr, It->second);
    LiveAllocations.erase(It);
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

protected:
  static constexpr size_t Count = 5;

  // Declared first so it outlives the mock actions that capture it.
  std::map<void *, std::align_val_t> LiveAllocations;
  unittests::UnittestsHelper Helper;
  queue Q;
};

class USMAllocatorKindTest : public USMAllocatorTest,
                             public WithParamInterface<usm::alloc> {
protected:
  // Calls F with an allocator of the kind the test is instantiated for.
  template <typename T, size_t Alignment = 0, typename FuncT>
  void withAllocator(FuncT F) {
    if (GetParam() == usm::alloc::host)
      F(usm_allocator<T, usm::alloc::host, Alignment>(Q));
    else
      F(usm_allocator<T, usm::alloc::shared, Alignment>(Q));
  }

  template <typename T, size_t Alignment = 0> void checkAllocate() {
    constexpr size_t ExpectedAlignment = std::max(Alignment, alignof(T));
    withAllocator<T, Alignment>([&](auto Alloc) {
      expectAllocation(GetParam(), Count * sizeof(T), ExpectedAlignment);
      T *Ptr = Alloc.allocate(Count);
      ASSERT_NE(Ptr, nullptr);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(Ptr) % ExpectedAlignment, 0u);

      EXPECT_CALL(Helper.Mock.get(), olMemFree(getOLContext(), Ptr))
          .WillOnce(DoDefault());
      Alloc.deallocate(Ptr, Count);
    });
  }

  template <typename T, size_t Alignment = 0> void checkVectorAllocate() {
    withAllocator<T, Alignment>([&](auto Alloc) {
      std::vector<T, decltype(Alloc)> Vec(Alloc);
      for (size_t I = 0; I < Count; ++I)
        Vec.push_back(T{});

      EXPECT_THAT(LiveAllocations,
                  Contains(Key(static_cast<void *>(Vec.data()))));
      EXPECT_EQ(reinterpret_cast<uintptr_t>(Vec.data()) % alignof(T), 0u);
    });
  }
};

struct alignas(64) OverAligned {
  int Value;
};

constexpr size_t SmallAlignment = 16;
constexpr size_t LargeAlignment = 256;

TEST_P(USMAllocatorKindTest, Allocate) {
  checkAllocate<uint8_t>();
  checkAllocate<uint64_t>();
  checkAllocate<OverAligned>();
  checkAllocate<OverAligned, SmallAlignment>();
  checkAllocate<OverAligned, LargeAlignment>();
}

TEST_P(USMAllocatorKindTest, VectorAllocate) {
  checkVectorAllocate<uint8_t>();
  checkVectorAllocate<uint64_t>();
  checkVectorAllocate<OverAligned>();
  checkVectorAllocate<OverAligned, SmallAlignment>();
  checkVectorAllocate<OverAligned, LargeAlignment>();
}

TEST_P(USMAllocatorKindTest, ThrowException) {
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

  withAllocator<uint64_t>([&](auto Alloc) {
    ExpectMemoryAllocationError([&] { Alloc.allocate(Count); });

    std::vector<uint64_t, decltype(Alloc)> Vec(Alloc);
    ExpectMemoryAllocationError([&] { Vec.push_back(1); });
  });
}

TEST_P(USMAllocatorKindTest, ZeroCount) {
  mock::MockLiboffload &Mock = Helper.Mock.get();
  EXPECT_CALL(Mock, olMemAllocAlignedHost(_, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock, olMemAllocAligned(_, _, _, _, _, _)).Times(0);
  EXPECT_CALL(Mock, olMemFree(_, _)).Times(0);

  withAllocator<int>([](auto Alloc) {
    EXPECT_EQ(Alloc.allocate(0), nullptr);
    Alloc.deallocate(nullptr, 0);
  });
}

INSTANTIATE_TEST_SUITE_P(USMAllocator, USMAllocatorKindTest,
                         Values(usm::alloc::host, usm::alloc::shared),
                         [](const TestParamInfo<usm::alloc> &Info) {
                           return Info.param == usm::alloc::host ? "Host"
                                                                 : "Shared";
                         });

TEST_F(USMAllocatorTest, Rebind) {
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
}

TEST_F(USMAllocatorTest, RebindContainer) {
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
  EXPECT_TRUE(Host == Aligned);

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

class USMAllocatorTwoDevicesTest : public Test {
protected:
  void SetUp() override {
    Platform = mock::createDummyHandle<ol_platform_handle_t>();
    for (ol_device_handle_t &Device : Devices)
      Device = mock::createDummyHandleWithData<ol_device_handle_t>(
          reinterpret_cast<unsigned char *>(&Platform), sizeof(Platform));

    EXPECT_CALL(Helper.Mock.get(), olIterateDevices(_, _))
        .WillRepeatedly([this](ol_device_iterate_cb_t Callback,
                               void *UserData) -> ol_result_t {
          for (ol_device_handle_t Device : Devices)
            std::ignore = Callback(Device, UserData);
          return OL_SUCCESS;
        });
  }

  void TearDown() override {
    mock::releaseDummyHandles(Devices[0], Devices[1], Platform);
  }

  unittests::UnittestsHelper Helper;
  ol_platform_handle_t Platform{};
  std::array<ol_device_handle_t, 2> Devices{};
};

TEST_F(USMAllocatorTwoDevicesTest, DeviceNotInContext) {
  std::vector<platform> Platforms = platform::get_platforms();
  ASSERT_EQ(Platforms.size(), 1u);

  std::vector<device> PlatformDevices = Platforms[0].get_devices();
  ASSERT_EQ(PlatformDevices.size(), 2u);

  using HostAlloc = usm_allocator<int, usm::alloc::host>;
  using SharedAlloc = usm_allocator<int, usm::alloc::shared>;
  context Ctx(PlatformDevices[0]);

  EXPECT_NO_THROW(HostAlloc Alloc(Ctx, PlatformDevices[0]));
  EXPECT_NO_THROW(SharedAlloc Alloc(Ctx, PlatformDevices[0]));
  // The device is ignored for host allocations.
  EXPECT_NO_THROW(HostAlloc Alloc(Ctx, PlatformDevices[1]));

  try {
    SharedAlloc Alloc(Ctx, PlatformDevices[1]);
    FAIL() << "Expected sycl::exception";
  } catch (const sycl::exception &E) {
    EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    ASSERT_TRUE(E.has_context());
    EXPECT_EQ(E.get_context(), Ctx);
  }
}

} // namespace
