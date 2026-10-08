//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl -Wno-error=deprecated-declarations %s -o %t.out
// RUN: %t.out

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <iostream>

constexpr std::size_t LocalSize = 8;
constexpr std::size_t WorkGroups = 4;
constexpr std::size_t GlobalSize = WorkGroups * LocalSize;

static bool check(bool Condition, const char *Case) {
  if (!Condition)
    std::cerr << "FAILED: " << Case << std::endl;
  return Condition;
}

// Work-items exchange data through local memory, and each work-group must get
// its own copy of it.
static bool testSharingWithinWorkGroup(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(GlobalSize, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 1> Local(LocalSize, CGH);
     CGH.parallel_for(sycl::nd_range<1>{GlobalSize, LocalSize},
                      [=](sycl::nd_item<1> It) {
                        const std::size_t Lid = It.get_local_id(0);
                        Local[Lid] = static_cast<int>(It.get_global_id(0));
                        sycl::group_barrier(It.get_group());
                        Out[It.get_global_id(0)] = Local[LocalSize - 1 - Lid];
                      });
   }).wait();

  bool Ok = true;
  for (std::size_t I = 0; I < GlobalSize; ++I) {
    const std::size_t Group = I / LocalSize;
    const std::size_t Lid = I % LocalSize;
    Ok &= Out[I] == static_cast<int>(Group * LocalSize + LocalSize - 1 - Lid);
  }
  sycl::free(Out, Q);
  return check(Ok, "sharing within a work-group");
}

// Several accessors of one kernel must not overlap.
static bool testMultipleAccessors(sycl::queue &Q, std::size_t WGSize) {
  const std::size_t Global = 2 * WGSize;
  int *Out = sycl::malloc_shared<int>(Global, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 1> Local1(WGSize, CGH);
     sycl::local_accessor<int, 1> Local2(WGSize, CGH);
     CGH.parallel_for(sycl::nd_range<1>{Global, WGSize},
                      [=](sycl::nd_item<1> It) {
                        const std::size_t Lid = It.get_local_id(0);
                        const int Gid = static_cast<int>(It.get_global_id(0));
                        Local1[Lid] = Gid;
                        Local2[Lid] = 2 * Gid;
                        Out[Gid] = Local1[Lid] + Local2[Lid];
                      });
   }).wait();

  bool Ok = true;
  for (std::size_t I = 0; I < Global; ++I)
    Ok &= Out[I] == static_cast<int>(3 * I);
  sycl::free(Out, Q);
  return check(Ok, "multiple accessors");
}

static bool testSubscript3D(sycl::queue &Q) {
  std::size_t *Out = sycl::malloc_shared<std::size_t>(1, Q);
  *Out = 0;
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<std::size_t, 3> Local(sycl::range<3>(2, 3, 4), CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       for (std::size_t I = 0; I < 2; ++I)
         for (std::size_t J = 0; J < 3; ++J)
           for (std::size_t K = 0; K < 4; ++K)
             Local[I][J][K] = I * 100 + J * 10 + K;
       // The id and the chained subscripts must address the same element.
       *Out =
           Local[sycl::id<3>(1, 2, 3)] == Local[1][2][3] ? Local[1][2][3] : 0;
     });
   }).wait();

  const bool Ok = *Out == 123;
  sycl::free(Out, Q);
  return check(Ok, "3-dimensional subscript");
}

static bool testZeroDim(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(2, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 0> Local(CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       Local = 64;
       Out[0] = Local;

       *Local.begin() = 32;
       int Value = *Local.cbegin();
       Value += *Local.crbegin();
       *Local.rbegin() += Value;
       Out[1] = Local;
     });
   }).wait();

  const bool Ok = Out[0] == 64 && Out[1] == 96;
  sycl::free(Out, Q);
  return check(Ok, "0-dimensional accessor");
}

// Fills the accessor from Data and writes back the elements transformed by
// the iterator members.
template <typename LocalAccT>
static void runIterators(sycl::handler &CGH, LocalAccT Local, int *Data,
                         bool ConstIterators) {
  CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
    if (ConstIterators) {
      std::size_t Idx = 0;
      for (auto &Elem : Local)
        Elem = Data[Idx++];
      Idx = 0;
      for (auto It = Local.cbegin(); It != Local.cend(); ++It)
        Data[Idx++] = *It * 2 + 1;
      Idx = Local.size() - 1;
      for (auto It = Local.crbegin(); It != Local.crend(); ++It)
        Data[Idx--] += *It;
    } else {
      std::size_t Idx = 0;
      for (auto It = Local.begin(); It != Local.end(); ++It)
        *It = Data[Idx++] * 2;
      for (auto &Elem : Local)
        ++Elem;
      for (auto It = Local.rbegin(); It != Local.rend(); ++It) {
        *It *= 2;
        *It += 1;
      }
      Idx = 0;
      for (auto &Elem : Local)
        Data[Idx++] = Elem;
    }
  });
}

static bool testIterators(sycl::queue &Q, bool ConstIterators, bool TwoDim) {
  constexpr int Size = 32;
  int *Data = sycl::malloc_shared<int>(Size, Q);
  for (int I = 0; I < Size; ++I)
    Data[I] = I;

  Q.submit([&](sycl::handler &CGH) {
     if (TwoDim)
       runIterators(CGH, sycl::local_accessor<int, 2>({2, Size / 2}, CGH), Data,
                    ConstIterators);
     else
       runIterators(CGH, sycl::local_accessor<int, 1>(Size, CGH), Data,
                    ConstIterators);
   }).wait();

  bool Ok = true;
  for (int I = 0; I < Size; ++I)
    Ok &= Data[I] == (ConstIterators ? (I * 2 + 1) + I : (I * 2 + 1) * 2 + 1);
  sycl::free(Data, Q);
  return check(Ok, ConstIterators
                       ? (TwoDim ? "2D const iterators" : "1D const iterators")
                       : (TwoDim ? "2D iterators" : "1D iterators"));
}

static bool testSwap(sycl::queue &Q) {
  std::size_t *Out = sycl::malloc_shared<std::size_t>(2, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 1> Local1(8, CGH), Local2(16, CGH);
     Local1.swap(Local2);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       Out[0] = Local1.size();
       Out[1] = Local2.size();
       // The regions must follow the swapped sizes.
       Local1[15] = 1;
       Local2[7] = 2;
     });
   }).wait();

  const bool Ok = Out[0] == 16 && Out[1] == 8;
  sycl::free(Out, Q);
  return check(Ok, "swap");
}

static int readConst(const sycl::local_accessor<const int, 1> &Acc) {
  return Acc[0];
}

static bool testConstDataT(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(2, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 1> Local(1, CGH);
     sycl::local_accessor<const int, 0> ConstScalar(CGH);
     sycl::local_accessor<const int, 1> ConstArray(5, CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       Local[0] = 123;
       Out[0] = readConst(Local);

       const int &RefScalar = ConstScalar;
       const int &RefArray = ConstArray[0];
       Out[1] = &RefScalar != &RefArray;
     });
   }).wait();

  const bool Ok = Out[0] == 123 && Out[1] == 1;
  sycl::free(Out, Q);
  return check(Ok, "const DataT");
}

static bool testZeroSized(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(1, Q);
  *Out = 0;
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<std::uint8_t, 1> ZeroSized(sycl::range<1>(0), CGH);
     sycl::local_accessor<int, 1> Local(1, CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       if (ZeroSized.get_range()[0])
         ZeroSized[0] = 1;
       Local[0] = 7;
       *Out = Local[0] + static_cast<int>(ZeroSized.size());
     });
   }).wait();

  const bool Ok = *Out == 7;
  sycl::free(Out, Q);
  return check(Ok, "zero-sized accessor");
}

struct DefaultConstructedFunctor {
  sycl::local_accessor<int, 1> Acc;
  void operator()(sycl::nd_item<1>) const {}
};

static bool testDefaultConstructed(sycl::queue &Q) {
  Q.submit([&](sycl::handler &CGH) {
     CGH.parallel_for(sycl::nd_range<1>(1, 1), DefaultConstructedFunctor{});
   }).wait();
  return true;
}

// Each accessor's region must be aligned to its element type even if a
// smaller type precedes it in the kernel object.
static bool testAlignment(sycl::queue &Q) {
  struct alignas(16) Aligned16 {
    float Data[4];
  };
  std::uintptr_t *Out = sycl::malloc_shared<std::uintptr_t>(2, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 1> A(1, CGH);
     sycl::local_accessor<Aligned16, 1> B(1, CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [A, B, Out](sycl::nd_item<1>) {
       Out[0] = reinterpret_cast<std::uintptr_t>(&A[0]);
       Out[1] = reinterpret_cast<std::uintptr_t>(&B[0]);
     });
   }).wait();

  const bool Ok =
      Out[0] % alignof(int) == 0 && Out[1] % alignof(Aligned16) == 0;
  sycl::free(Out, Q);
  return check(Ok, "alignment");
}

static bool testDeprecatedAccessor(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(2, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::accessor<int, 1, sycl::access_mode::read_write, sycl::target::local>
         Local(LocalSize, CGH);
     CGH.parallel_for(sycl::nd_range<1>{LocalSize, LocalSize},
                      [=](sycl::nd_item<1> It) {
                        const std::size_t Lid = It.get_local_id(0);
                        Local[Lid] = static_cast<int>(Lid);
                        sycl::group_barrier(It.get_group());
                        if (Lid == 0) {
                          int Sum = 0;
                          for (std::size_t I = 0; I < LocalSize; ++I)
                            Sum += Local[I];
                          Out[0] = Sum;
                          Out[1] = static_cast<int>(Local.get_count());
                        }
                      });
   }).wait();

  const bool Ok = Out[0] == static_cast<int>(LocalSize * (LocalSize - 1) / 2) &&
                  Out[1] == static_cast<int>(LocalSize);
  sycl::free(Out, Q);
  return check(Ok, "deprecated accessor with target::local");
}

static bool testDeprecatedAccessorDims(sycl::queue &Q) {
  using Scalar = sycl::accessor<int, 0, sycl::access_mode::read_write,
                                sycl::target::local>;
  using Array2D = sycl::accessor<int, 2, sycl::access_mode::read_write,
                                 sycl::target::local>;
  int *Out = sycl::malloc_shared<int>(3, Q);
  Q.submit([&](sycl::handler &CGH) {
     Scalar Local0(CGH);
     Array2D Local2(sycl::range<2>(2, 3), CGH);
     CGH.parallel_for(sycl::nd_range<1>{1, 1}, [=](sycl::nd_item<1>) {
       int &Ref = Local0;
       Ref = 5;
       for (std::size_t I = 0; I < 2; ++I)
         for (std::size_t J = 0; J < 3; ++J)
           Local2[I][J] = static_cast<int>(I * 10 + J);
       Out[0] = Local0;
       Out[1] = Local2[sycl::id<2>(1, 2)];
       Out[2] = static_cast<int>(Local2.get_range()[1]);
     });
   }).wait();

  const bool Ok = Out[0] == 5 && Out[1] == 12 && Out[2] == 3;
  sycl::free(Out, Q);
  return check(Ok, "0- and 2-dimensional deprecated accessor");
}

// SYCL 2020 4.5.2: copies of an accessor in one kernel object must refer to
// the same memory.
struct TwoCopiesFunctor {
  sycl::local_accessor<int, 1> Writer;
  sycl::local_accessor<const int, 1> Reader;
  int *Out;
  void operator()(sycl::nd_item<1> It) const {
    const std::size_t Lid = It.get_local_id(0);
    Writer[Lid] = static_cast<int>(Lid) + 1;
    sycl::group_barrier(It.get_group());
    Out[It.get_global_id(0)] = Reader[LocalSize - 1 - Lid];
  }
};

static bool testCopiesShareMemory(sycl::queue &Q) {
  int *Out = sycl::malloc_shared<int>(LocalSize, Q);
  Q.submit([&](sycl::handler &CGH) {
     sycl::local_accessor<int, 1> Local(LocalSize, CGH);
     CGH.parallel_for(sycl::nd_range<1>{LocalSize, LocalSize},
                      TwoCopiesFunctor{Local, Local, Out});
   }).wait();

  bool Ok = true;
  for (std::size_t I = 0; I < LocalSize; ++I)
    Ok &= Out[I] == static_cast<int>(LocalSize - I);
  sycl::free(Out, Q);
  return check(Ok, "copies share memory");
}

// SYCL 2020 4.7.6.11: submitting a kernel that uses a local accessor with
// single_task or with a parallel_for taking a range must throw. SYCL 2020
// 4.7.6.9.4.7 gives the deprecated accessor with target::local the same
// restrictions.
template <typename AccT, typename SubmitT>
static bool expectKernelArgumentError(sycl::queue &Q, SubmitT Submit,
                                      const char *Case) {
  try {
    Q.submit([&](sycl::handler &CGH) {
      AccT Local(1, CGH);
      Submit(CGH, Local);
    });
  } catch (const sycl::exception &E) {
    return check(E.code() == sycl::make_error_code(sycl::errc::kernel_argument),
                 Case);
  }
  return check(false, Case);
}

template <typename AccT> class SingleTaskKernel;

template <typename AccT>
static bool testKernelArgumentErrors(sycl::queue &Q, const char *SingleTaskCase,
                                     const char *RangeCase) {
  bool Ok = expectKernelArgumentError<AccT>(
      Q,
      [](sycl::handler &CGH, AccT Local) {
        CGH.single_task<SingleTaskKernel<AccT>>([=]() { (void)Local; });
      },
      SingleTaskCase);
  Ok &= expectKernelArgumentError<AccT>(
      Q,
      [](sycl::handler &CGH, AccT Local) {
        CGH.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) { (void)Local; });
      },
      RangeCase);
  return Ok;
}

int main() {
  sycl::queue Q;

  bool Ok = true;
  Ok &= testSharingWithinWorkGroup(Q);
  Ok &= testMultipleAccessors(Q, 10);
  Ok &= testMultipleAccessors(Q, 20);
  Ok &= testSubscript3D(Q);
  Ok &= testZeroDim(Q);
  for (bool ConstIterators : {false, true})
    for (bool TwoDim : {false, true})
      Ok &= testIterators(Q, ConstIterators, TwoDim);
  Ok &= testSwap(Q);
  Ok &= testConstDataT(Q);
  Ok &= testZeroSized(Q);
  Ok &= testDefaultConstructed(Q);
  Ok &= testAlignment(Q);
  Ok &= testDeprecatedAccessor(Q);
  Ok &= testDeprecatedAccessorDims(Q);
  Ok &= testCopiesShareMemory(Q);
  Ok &= testKernelArgumentErrors<sycl::local_accessor<int, 1>>(
      Q, "local accessor in single_task",
      "local accessor in parallel_for with a range");
  Ok &= testKernelArgumentErrors<sycl::accessor<
      int, 1, sycl::access_mode::read_write, sycl::target::local>>(
      Q, "deprecated local accessor in single_task",
      "deprecated local accessor in parallel_for with a range");

  return Ok ? 0 : 1;
}
