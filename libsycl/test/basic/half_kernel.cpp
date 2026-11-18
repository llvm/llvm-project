//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: any-device
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %t.out
//
// Checks that sycl::half conversions, arithmetic and numeric_limits in a kernel
// give the same bits as on the host.

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

using sycl::half;
using Limits = std::numeric_limits<half>;

namespace {

std::uint16_t bits(half H) {
  std::uint16_t Result;
  std::memcpy(&Result, &H, sizeof(Result));
  return Result;
}

bool isNaN(half H) { return std::isnan(static_cast<float>(H)); }

// Results of the operations on one input. Members are in the order used by
// computeResults.
struct Results {
  half Converted;
  half Incremented;
  half Sum;
  half Difference;
  half Product;
  half Quotient;
};

// Computes the results for the inputs F (any float) and A, B (operands of the
// arithmetic, B non-zero). The same code runs on the host and in the kernel.
Results computeResults(float F, half A, half B) {
  Results R;
  R.Converted = F;
  R.Incremented = R.Converted;
  ++R.Incremented;
  R.Sum = A + B;
  R.Difference = A - B;
  R.Product = A * B;
  R.Quotient = A / B;
  return R;
}

// NaN results only need to be NaNs: their payload isn't specified.
bool sameBits(half Device, half Host) {
  return bits(Device) == bits(Host) || (isNaN(Device) && isNaN(Host));
}

// Division on the device may differ from the correctly rounded result by one
// unit in the last place.
bool withinOneULP(half Device, half Host) {
  if (sameBits(Device, Host))
    return true;
  if ((bits(Device) ^ bits(Host)) & 0x8000)
    return false;
  int Distance = static_cast<int>(bits(Device) & 0x7FFF) -
                 static_cast<int>(bits(Host) & 0x7FFF);
  return std::abs(Distance) == 1;
}

} // namespace

int main() {
  sycl::queue Q;
  if (!Q.get_device().has(sycl::aspect::fp16)) {
    std::cout << "Skipped: the device doesn't have aspect::fp16\n";
    return 0;
  }

  // Conversion inputs: special values, rounding boundaries and subnormals,
  // then a sweep of inexact values as in DPC++'s half_operators test.
  std::vector<float> Floats = {0.0f,
                               -0.0f,
                               1.0f,
                               1.9999f,
                               2049.0f,
                               2051.0f,
                               65504.0f,
                               65519.0f,
                               65520.0f,
                               75514.0f,
                               -75514.0f,
                               std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity(),
                               std::numeric_limits<float>::quiet_NaN(),
                               6.103515625e-05f,
                               3.8146972e-05f,
                               5.9604644775390625e-08f,
                               2.9802322e-08f,
                               9.8e-45f,
                               -9.8e-45f};
  for (int I = 0; I < 64; ++I)
    Floats.push_back(0.1f * static_cast<float>(I) - 3.0f);
  const size_t N = Floats.size();

  // Arithmetic operands: multiples of at least 0.37, so that no result is
  // subnormal and devices that flush subnormals give the same results.
  std::vector<half> As(N), Bs(N);
  for (size_t I = 0; I < N; ++I) {
    As[I] = static_cast<float>(static_cast<int>(I % 61) - 30) * 0.37f;
    Bs[I] = static_cast<float>(I % 17 + 1) * 0.73f;
  }

  float *InF = sycl::malloc_shared<float>(N, Q);
  half *InA = sycl::malloc_shared<half>(N, Q);
  half *InB = sycl::malloc_shared<half>(N, Q);
  Results *Out = sycl::malloc_shared<Results>(N, Q);
  constexpr size_t NumLimits = 9;
  half *OutLimits = sycl::malloc_shared<half>(NumLimits, Q);
  for (size_t I = 0; I < N; ++I) {
    InF[I] = Floats[I];
    InA[I] = As[I];
    InB[I] = Bs[I];
  }

  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> Idx) {
     size_t I = Idx[0];
     Out[I] = computeResults(InF[I], InA[I], InB[I]);
   }).wait();
  Q.single_task([=] {
     OutLimits[0] = (Limits::min)();
     OutLimits[1] = (Limits::max)();
     OutLimits[2] = Limits::lowest();
     OutLimits[3] = Limits::epsilon();
     OutLimits[4] = Limits::round_error();
     OutLimits[5] = Limits::infinity();
     OutLimits[6] = Limits::quiet_NaN();
     OutLimits[7] = Limits::signaling_NaN();
     OutLimits[8] = Limits::denorm_min();
   }).wait();

  int Failures = 0;
  auto Check = [&](bool Ok, const char *What, size_t I, half Device,
                   half Host) {
    if (Ok)
      return;
    ++Failures;
    std::cout << What << " [" << I << "]: device 0x" << std::hex << bits(Device)
              << ", host 0x" << bits(Host) << std::dec << "\n";
  };
  for (size_t I = 0; I < N; ++I) {
    Results Host = computeResults(InF[I], InA[I], InB[I]);
    const Results &Device = Out[I];
    Check(sameBits(Device.Converted, Host.Converted), "conversion", I,
          Device.Converted, Host.Converted);
    Check(sameBits(Device.Incremented, Host.Incremented), "increment", I,
          Device.Incremented, Host.Incremented);
    Check(sameBits(Device.Sum, Host.Sum), "sum", I, Device.Sum, Host.Sum);
    Check(sameBits(Device.Difference, Host.Difference), "difference", I,
          Device.Difference, Host.Difference);
    Check(sameBits(Device.Product, Host.Product), "product", I, Device.Product,
          Host.Product);
    Check(withinOneULP(Device.Quotient, Host.Quotient), "quotient", I,
          Device.Quotient, Host.Quotient);
  }

  const half HostLimits[NumLimits] = {
      (Limits::min)(),     (Limits::max)(),         Limits::lowest(),
      Limits::epsilon(),   Limits::round_error(),   Limits::infinity(),
      Limits::quiet_NaN(), Limits::signaling_NaN(), Limits::denorm_min()};
  for (size_t I = 0; I < NumLimits; ++I)
    Check(bits(OutLimits[I]) == bits(HostLimits[I]), "numeric_limits", I,
          OutLimits[I], HostLimits[I]);

  sycl::free(InF, Q);
  sycl::free(InA, Q);
  sycl::free(InB, Q);
  sycl::free(Out, Q);
  sycl::free(OutLimits, Q);
  return Failures != 0;
}
