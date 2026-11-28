//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <common/scoped_binary_registration.hpp>
#include <mock/helpers.hpp>

#include <sycl/__impl/accessor.hpp>
#include <sycl/__impl/handler.hpp>
#include <sycl/__impl/local_accessor.hpp>
#include <sycl/__impl/properties/queue_properties.hpp>
#include <sycl/__impl/queue.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <functional>
#include <tuple>
#include <vector>

using namespace sycl;
using namespace ::testing;

class sycl::detail::MockQueue : public sycl::queue {
public:
  using sycl::queue::sycl_kernel_launch;
};

namespace {

struct alignas(16) Aligned16 {
  float Data[4];
};

} // namespace

TEST(LocalAccessor, DefaultConstructedIsEmpty) {
  local_accessor<int, 1> Acc;
  EXPECT_TRUE(Acc.empty());
  EXPECT_EQ(Acc.size(), 0u);
  EXPECT_EQ(Acc.byte_size(), 0u);
  EXPECT_EQ(Acc.max_size(), 0u);
  EXPECT_EQ(Acc.get_range(), range<1>(0));
}

TEST(LocalAccessor, SizeQueries) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 2> Acc2;
  local_accessor<double, 0> Acc0;
  local_accessor<char, 1> ZeroSized;
  Q.submit([&](handler &CGH) {
    Acc2 = local_accessor<int, 2>(range<2>{2, 16}, CGH);
    Acc0 = local_accessor<double, 0>(CGH);
    ZeroSized = local_accessor<char, 1>(range<1>{0}, CGH);
  });

  EXPECT_FALSE(Acc2.empty());
  EXPECT_EQ(Acc2.size(), 32u);
  EXPECT_EQ(Acc2.byte_size(), 32 * sizeof(int));
  EXPECT_EQ(Acc2.get_range(), (range<2>{2, 16}));
  EXPECT_GT(Acc2.max_size(), 0u);

  EXPECT_FALSE(Acc0.empty());
  EXPECT_EQ(Acc0.size(), 1u);
  EXPECT_EQ(Acc0.byte_size(), sizeof(double));

  EXPECT_TRUE(ZeroSized.empty());
  EXPECT_EQ(ZeroSized.size(), 0u);
  EXPECT_EQ(ZeroSized.max_size(), 0u);
}

TEST(LocalAccessor, CopiesAreEqualAndHashEqual) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Acc1, Acc3;
  Q.submit([&](handler &CGH) {
    Acc1 = local_accessor<int, 1>(1, CGH);
    Acc3 = local_accessor<int, 1>(1, CGH);
  });
  local_accessor<int, 1> Acc2 = Acc1;

  using HashT = std::hash<local_accessor<int, 1>>;
  EXPECT_EQ(Acc1, Acc2);
  EXPECT_EQ(HashT{}(Acc1), HashT{}(Acc2));
  EXPECT_NE(Acc1, Acc3);
  EXPECT_NE(HashT{}(Acc1), HashT{}(Acc3));
}

// SYCL 2020 4.5.2: instances that are not copies of each other are not equal,
// default constructed ones included.
TEST(LocalAccessor, DefaultConstructedAreNotEqual) {
  using HashT = std::hash<local_accessor<int, 1>>;
  local_accessor<int, 1> Acc1, Acc2;
  const local_accessor<int, 1> Copy = Acc1;
  EXPECT_NE(Acc1, Acc2);
  EXPECT_NE(HashT{}(Acc1), HashT{}(Acc2));
  EXPECT_EQ(Acc1, Copy);
  EXPECT_EQ(HashT{}(Acc1), HashT{}(Copy));
}

TEST(LocalAccessor, ConvertsToConstDataT) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Default;
  EXPECT_EQ((local_accessor<const int, 1>(Default)),
            (local_accessor<const int, 1>(Default)));

  local_accessor<int, 1> Acc;
  Q.submit([&](handler &CGH) { Acc = local_accessor<int, 1>(4, CGH); });
  local_accessor<const int, 1> ConstAcc1(Acc);
  local_accessor<const int, 1> ConstAcc2 = Acc;
  EXPECT_EQ(ConstAcc1, ConstAcc2);
  EXPECT_EQ(ConstAcc1.size(), 4u);
  EXPECT_EQ((std::hash<local_accessor<const int, 1>>{}(ConstAcc1)),
            (std::hash<local_accessor<int, 1>>{}(Acc)));
}

TEST(LocalAccessor, Swap) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Acc1, Acc2;
  Q.submit([&](handler &CGH) {
    Acc1 = local_accessor<int, 1>(8, CGH);
    Acc2 = local_accessor<int, 1>(16, CGH);
  });
  const local_accessor<int, 1> Orig1 = Acc1;
  const local_accessor<int, 1> Orig2 = Acc2;

  Acc1.swap(Acc2);
  EXPECT_EQ(Acc1.size(), 16u);
  EXPECT_EQ(Acc2.size(), 8u);
  EXPECT_EQ(Acc1, Orig2);
  EXPECT_EQ(Acc2, Orig1);
}

TEST(LocalAccessor, HasNoProperties) {
  mock::MockWrapper Mock;
  queue Q;
  local_accessor<int, 1> Acc;
  Q.submit([&](handler &CGH) { Acc = local_accessor<int, 1>(1, CGH); });
  EXPECT_FALSE(Acc.has_property<property::queue::in_order>());
  EXPECT_THROW(Acc.get_property<property::queue::in_order>(), exception);
}

TEST(LocalAccessor, RejectsPropertyOfAnotherClass) {
  mock::MockWrapper Mock;
  queue Q;
  Q.submit([&](handler &CGH) {
    try {
      local_accessor<int, 1> Acc(1, CGH, {property::queue::in_order{}});
      FAIL() << "a queue property must be rejected by a local accessor";
    } catch (const exception &E) {
      EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    }
  });
}

TEST(DeprecatedLocalAccessor, SizeQueriesAndEquality) {
  using AccT = accessor<int, 2, access_mode::read_write, target::local>;
  mock::MockWrapper Mock;
  queue Q;
  AccT Acc1, Acc3;
  Q.submit([&](handler &CGH) {
    Acc1 = AccT(range<2>{2, 3}, CGH);
    Acc3 = AccT(range<2>{2, 3}, CGH);
  });
  AccT Acc2 = Acc1;

  EXPECT_EQ(Acc1.get_count(), 6u);
  EXPECT_EQ(Acc1.get_size(), 6 * sizeof(int));
  EXPECT_EQ(Acc1.size(), 6u);
  EXPECT_EQ(Acc1.get_range(), (range<2>{2, 3}));
  EXPECT_EQ(Acc1, Acc2);
  EXPECT_NE(Acc1, Acc3);
  EXPECT_EQ(std::hash<AccT>{}(Acc1), std::hash<AccT>{}(Acc2));

  AccT Default;
  EXPECT_TRUE(Default.empty());
  EXPECT_EQ(Default.get_count(), 0u);
}

TEST(DeprecatedLocalAccessor, ZeroDimSizeQueries) {
  using AccT = accessor<double, 0, access_mode::read_write, target::local>;
  mock::MockWrapper Mock;
  queue Q;
  AccT Acc;
  Q.submit([&](handler &CGH) { Acc = AccT(CGH); });

  EXPECT_FALSE(Acc.empty());
  EXPECT_EQ(Acc.get_count(), 1u);
  EXPECT_EQ(Acc.get_size(), sizeof(double));
}

TEST(DeprecatedLocalAccessor, AcceptsEmptyPropertyList) {
  using AccT = accessor<int, 1, access_mode::read_write, target::local>;
  mock::MockWrapper Mock;
  queue Q;
  AccT Acc;
  Q.submit([&](handler &CGH) { Acc = AccT(4, CGH, property_list{}); });
  EXPECT_EQ(Acc.get_count(), 4u);
  EXPECT_FALSE(Acc.has_property<property::queue::in_order>());
  EXPECT_THROW(Acc.get_property<property::queue::in_order>(), exception);
}

TEST(DeprecatedLocalAccessor, RejectsPropertyOfAnotherClass) {
  using AccT = accessor<int, 1, access_mode::read_write, target::local>;
  mock::MockWrapper Mock;
  queue Q;
  Q.submit([&](handler &CGH) {
    try {
      AccT Acc(1, CGH, {property::queue::in_order{}});
      FAIL() << "a queue property must be rejected by a local accessor";
    } catch (const exception &E) {
      EXPECT_EQ(E.code(), make_error_code(errc::invalid));
    }
  });
}

// The tests below check the size of the work-group local memory block that is
// requested at launch for the local accessors of a kernel and how the
// accessors share it.

struct LocalMemKernel;
struct LocalMemKernelInfo {
  using kernel_name = LocalMemKernel;
  static constexpr char kernel_entry_point_name[] = "LocalMemKernel";
};

struct KernelData {
  int Value;
};

// getDeviceKernelInfo<KernelName> caches the kernel info for the whole process
// and unregistering the binary destroys it, so the registration must outlive
// every launch of LocalMemKernel.
static void registerLocalMemKernel() {
  static unittests::ScopedKernelRegistration Registration{"LocalMemKernel"};
}

template <detail::KernelLaunchKind Kind>
using KernelObjectT = detail::KernelObject<KernelData, Kind>;

// A kernel function object holding local accessors next to other data.
template <typename... AccTs> struct AccessorKernel {
  int Value;
  std::tuple<AccTs...> Accs;
};

constexpr int KernelValue = 42;

struct LaunchResult {
  std::size_t LocalMemSize = 0;
  // The bytes of each accessor in the kernel argument.
  std::vector<std::vector<unsigned char>> AccessorBytes;
};

template <typename T> static std::vector<unsigned char> bytesOf(const T &Obj) {
  const auto *Begin = reinterpret_cast<const unsigned char *>(&Obj);
  return {Begin, Begin + sizeof(T)};
}

// Launches LocalMemKernel as an nd_range kernel whose kernel object holds Accs,
// passing them to the callable returned by sycl_kernel_launch as the compiler
// does.
template <typename... AccTs>
static LaunchResult launchWithLocalAccessors(mock::MockWrapper &Mock,
                                             detail::MockQueue &Q,
                                             const AccTs &...Accs) {
  registerLocalMemKernel();
  using KernelT = detail::KernelObject<AccessorKernel<AccTs...>,
                                       detail::KernelLaunchKind::NDRange>;
  KernelT Kernel{{KernelValue, {Accs...}}};
  KernelT Received{};
  LaunchResult Result;
  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, 1, _, _))
      .WillOnce([&](ol_queue_handle_t, ol_device_handle_t, ol_symbol_handle_t,
                    const ol_kernel_launch_size_args_t *Args,
                    const ol_kernel_launch_prop_t *, size_t, void **ArgPtrs,
                    const size_t *ArgSizes) -> ol_result_t {
        Result.LocalMemSize = Args->DynSharedMemory;
        EXPECT_EQ(ArgSizes[0], sizeof(KernelT));
        std::memcpy(static_cast<void *>(&Received), ArgPtrs[0],
                    sizeof(KernelT));
        return OL_SUCCESS;
      });
  std::apply(
      [&](AccTs &...Members) {
        Q.sycl_kernel_launch<LocalMemKernelInfo>(Kernel)(Members...);
      },
      Kernel.Func.Accs);
  EXPECT_EQ(Received.Func.Value, KernelValue);
  Result.AccessorBytes = std::apply(
      [](const AccTs &...Members) {
        return std::vector<std::vector<unsigned char>>{bytesOf(Members)...};
      },
      Received.Func.Accs);
  return Result;
}

template <typename... AccTs>
static std::size_t launchAndCaptureLocalMemSize(mock::MockWrapper &Mock,
                                                detail::MockQueue &Q,
                                                const AccTs &...Accs) {
  return launchWithLocalAccessors(Mock, Q, Accs...).LocalMemSize;
}

TEST(LocalAccessorLaunch, NoLocalAccessors) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q), 0u);
}

TEST(LocalAccessorLaunch, SingleAccessor) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 3> Acc;
  Q.submit([&](handler &CGH) {
    Acc = local_accessor<int, 3>(range<3>{2, 3, 4}, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Acc), 24 * sizeof(int));
}

// Regions are aligned to their element type, so an int followed by a 16-byte
// aligned type leaves padding in between.
TEST(LocalAccessorLaunch, RegionsAreAligned) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 1> IntAcc;
  local_accessor<Aligned16, 1> AlignedAcc;
  Q.submit([&](handler &CGH) {
    IntAcc = local_accessor<int, 1>(1, CGH);
    AlignedAcc = local_accessor<Aligned16, 1>(1, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, IntAcc, AlignedAcc),
            2 * sizeof(Aligned16));
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, AlignedAcc, IntAcc),
            sizeof(Aligned16) + sizeof(int));
}

// Default constructed and zero-sized accessors take no local memory, but their
// regions are still aligned so that the device never forms a misaligned
// pointer.
TEST(LocalAccessorLaunch, EmptyAccessorsOnlyAddAlignment) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 1> Default;
  local_accessor<char, 1> ZeroSized, Byte;
  Q.submit([&](handler &CGH) {
    ZeroSized = local_accessor<char, 1>(range<1>{0}, CGH);
    Byte = local_accessor<char, 1>(1, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Default, ZeroSized), 0u);
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, ZeroSized, Byte), 1u);
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Byte, Default, ZeroSized),
            alignof(int));
}

// Copies of one accessor share a region, distinct accessors of the same size
// do not.
TEST(LocalAccessorLaunch, CopiesShareRegion) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<int, 1> Acc, Other;
  local_accessor<char, 1> Byte;
  Q.submit([&](handler &CGH) {
    Acc = local_accessor<int, 1>(4, CGH);
    Other = local_accessor<int, 1>(4, CGH);
    Byte = local_accessor<char, 1>(1, CGH);
  });
  const local_accessor<int, 1> Copy = Acc;
  const local_accessor<const int, 1> ConstCopy = Acc;
  constexpr std::size_t RegionSize = 4 * sizeof(int);

  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Acc, Copy), RegionSize);
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Acc, ConstCopy, Copy),
            RegionSize);
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Acc, Other), 2 * RegionSize);
  // A copy after another accessor reuses the earlier region and appends
  // nothing, so the block ends with the region of Other.
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Acc, Byte, Copy, Other),
            RegionSize + alignof(int) + RegionSize);

  // A copy is identical to the original in the kernel argument only if both
  // got the same region.
  const LaunchResult Result =
      launchWithLocalAccessors(Mock, Q, Acc, Byte, Copy, ConstCopy);
  EXPECT_EQ(Result.AccessorBytes[2], Result.AccessorBytes[0]);
  EXPECT_EQ(Result.AccessorBytes[3], Result.AccessorBytes[0]);
}

// Distinct default constructed accessors get regions of their own, each
// aligned for its DataT, and a copy of one shares its region.
TEST(LocalAccessorLaunch, DefaultConstructedAreDeduplicatedLikeOthers) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  local_accessor<char, 1> Byte1, Byte2;
  Q.submit([&](handler &CGH) {
    Byte1 = local_accessor<char, 1>(1, CGH);
    Byte2 = local_accessor<char, 1>(1, CGH);
  });
  local_accessor<int, 1> DefaultInt;
  local_accessor<double, 1> DefaultDouble;
  const local_accessor<int, 1> DefaultIntCopy = DefaultInt;
  EXPECT_EQ(
      launchAndCaptureLocalMemSize(Mock, Q, Byte1, DefaultInt, DefaultDouble),
      alignof(double));
  // Byte2 ends the block at alignof(int) + 1. The copy reuses the region of
  // DefaultInt, a new region would be aligned past the end of Byte2.
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Byte1, DefaultInt, Byte2,
                                         DefaultIntCopy),
            alignof(int) + 1);
}

TEST(LocalAccessorLaunch, DeprecatedAccessorTakesMemory) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  using DeprecatedAccT =
      accessor<int, 1, access_mode::read_write, target::local>;
  DeprecatedAccT Deprecated;
  local_accessor<char, 0> Scalar;
  Q.submit([&](handler &CGH) {
    Scalar = local_accessor<char, 0>(CGH);
    Deprecated = DeprecatedAccT(3, CGH);
  });
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Scalar, Deprecated),
            alignof(int) + 3 * sizeof(int));
  const DeprecatedAccT Copy = Deprecated;
  EXPECT_EQ(launchAndCaptureLocalMemSize(Mock, Q, Deprecated, Copy),
            3 * sizeof(int));
}

// SYCL 2020 4.7.6.11: a local accessor must not be used in a kernel invoked
// with single_task or with a parallel_for taking a range. SYCL 2020 4.7.6.9.4.7
// gives the deprecated accessor with target::local the same restrictions.
template <detail::KernelLaunchKind Kind, typename AccT>
static void expectKernelArgumentError() {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  registerLocalMemKernel();
  AccT Default;
  AccT Acc;
  Q.submit([&](handler &CGH) { Acc = AccT(1, CGH); });

  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, _, _, _)).Times(0);
  KernelObjectT<Kind> Kernel{};
  for (const AccT &Used : {Acc, Default}) {
    try {
      Q.sycl_kernel_launch<LocalMemKernelInfo>(Kernel)(Used);
      ADD_FAILURE() << "the launch must throw";
    } catch (const exception &E) {
      EXPECT_EQ(E.code(), make_error_code(errc::kernel_argument));
    }
  }
}

using LocalAccT = local_accessor<int, 1>;
using DeprecatedLocalAccT =
    accessor<int, 1, access_mode::read_write, target::local>;

TEST(LocalAccessorLaunch, SingleTaskRejectsLocalAccessor) {
  expectKernelArgumentError<detail::KernelLaunchKind::SingleTask, LocalAccT>();
}

TEST(LocalAccessorLaunch, RangeRejectsLocalAccessor) {
  expectKernelArgumentError<detail::KernelLaunchKind::Range, LocalAccT>();
}

TEST(LocalAccessorLaunch, SingleTaskRejectsDeprecatedLocalAccessor) {
  expectKernelArgumentError<detail::KernelLaunchKind::SingleTask,
                            DeprecatedLocalAccT>();
}

TEST(LocalAccessorLaunch, RangeRejectsDeprecatedLocalAccessor) {
  expectKernelArgumentError<detail::KernelLaunchKind::Range,
                            DeprecatedLocalAccT>();
}

TEST(LocalAccessorLaunch, KernelWithoutLocalAccessorIsAccepted) {
  mock::MockWrapper Mock;
  detail::MockQueue Q;
  registerLocalMemKernel();
  EXPECT_CALL(Mock.get(), olLaunchKernel(_, _, _, _, _, 1, _, _)).Times(1);
  KernelObjectT<detail::KernelLaunchKind::SingleTask> Kernel{};
  EXPECT_NO_THROW(Q.sycl_kernel_launch<LocalMemKernelInfo>(Kernel)());
}
