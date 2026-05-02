//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared fixtures for libc++ tests under util.smartptr/atomic/{shared,weak}/:
// heterogeneous value types (built-in, standard library, and user-defined).
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_MEMORY_UTIL_SMARTPTR_ATOMIC_ATOMIC_SMART_PTR_TEST_TYPES_H
#define TEST_STD_UTILITIES_MEMORY_UTIL_SMARTPTR_ATOMIC_ATOMIC_SMART_PTR_TEST_TYPES_H

#include <cstdint>
#include <memory>
#include <string>

// --- User-defined and semi-random scalar-like types ---------------------------------

struct TrackedPod {
  std::uint32_t gen{};
  std::int64_t salt{};
  friend constexpr bool operator==(TrackedPod lhs, TrackedPod rhs) noexcept {
    return lhs.gen == rhs.gen && lhs.salt == rhs.salt;
  }
};

class Handle {
  double coeff_{};

public:
  explicit Handle(double c = 0.0) : coeff_(c) {}
  double coeff() const noexcept { return coeff_; }
  friend bool operator==(Handle const& lhs, Handle const& rhs) noexcept { return lhs.coeff_ == rhs.coeff_; }
};

enum class Flag : std::uint16_t { Off = 0, Stale = 4099, On = 60000 };

// --- Distinct shared states for compare/exchange style tests ------------------------

template <class T>
struct SpValues;

template <>
struct SpValues<int> {
  static std::shared_ptr<int> state_a() { return std::make_shared<int>(-90210); }
  static std::shared_ptr<int> state_b() { return std::make_shared<int>(404); }
  static std::shared_ptr<int> state_c() { return std::make_shared<int>(7331); }
};

template <>
struct SpValues<double> {
  static std::shared_ptr<double> state_a() { return std::make_shared<double>(1.4142135623730951); }
  static std::shared_ptr<double> state_b() { return std::make_shared<double>(2.7182818284590452); }
  static std::shared_ptr<double> state_c() { return std::make_shared<double>(3.1415926535897932); }
};

template <>
struct SpValues<std::string> {
  static std::shared_ptr<std::string> state_a() { return std::make_shared<std::string>("kappa"); }
  static std::shared_ptr<std::string> state_b() { return std::make_shared<std::string>("lambda"); }
  static std::shared_ptr<std::string> state_c() { return std::make_shared<std::string>("mu"); }
};

template <>
struct SpValues<TrackedPod> {
  static std::shared_ptr<TrackedPod> state_a() { return std::make_shared<TrackedPod>(TrackedPod{3u, -77L}); }
  static std::shared_ptr<TrackedPod> state_b() { return std::make_shared<TrackedPod>(TrackedPod{101u, 1L << 20}); }
  static std::shared_ptr<TrackedPod> state_c() { return std::make_shared<TrackedPod>(TrackedPod{255u, -1L}); }
};

template <>
struct SpValues<Handle> {
  static std::shared_ptr<Handle> state_a() { return std::make_shared<Handle>(0.125); }
  static std::shared_ptr<Handle> state_b() { return std::make_shared<Handle>(-4096.5); }
  static std::shared_ptr<Handle> state_c() { return std::make_shared<Handle>(8192.25); }
};

template <>
struct SpValues<Flag> {
  static std::shared_ptr<Flag> state_a() { return std::make_shared<Flag>(Flag::Off); }
  static std::shared_ptr<Flag> state_b() { return std::make_shared<Flag>(Flag::Stale); }
  static std::shared_ptr<Flag> state_c() { return std::make_shared<Flag>(Flag::On); }
};

struct ForEachSmartPtrType {
  template <template <class> class Fn>
  void operator()() const {
    Fn<int>{}();
    Fn<double>{}();
    Fn<std::string>{}();
    Fn<TrackedPod>{}();
    Fn<Handle>{}();
    Fn<Flag>{}();
  }
};

#endif // TEST_STD_UTILITIES_MEMORY_UTIL_SMARTPTR_ATOMIC_ATOMIC_SMART_PTR_TEST_TYPES_H
