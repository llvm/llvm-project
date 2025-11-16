//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23
// UNSUPPORTED: availability-stacktrace-missing

// (19.6.4.4) Comparisons [stacktrace.basic.cmp]
//
//   template<class Allocator2>
//   friend strong_ordering operator<=>(const basic_stacktrace& x,
//                                      const basic_stacktrace<Allocator2>& y) noexcept;
//
//   Returns: x.size() <=> y.size() if x.size() != y.size();
//            lexicographical_compare_three_way(x.begin(), x.end(), y.begin(), y.end()) otherwise.

#include <cassert>
#include <cstdint>
#include <stacktrace>

namespace {

// Create a stacktrace with entries having the given addresses.
std::stacktrace fake_trace(std::vector<uintptr_t> const& addrs) {
  std::stacktrace ret;
  auto& base = *reinterpret_cast<std::__stacktrace::_Trace*>(&ret);
  for (uintptr_t addr : addrs) {
    auto& entry = base.__entry_append_();
    auto& eb    = *reinterpret_cast<std::__stacktrace::_Entry*>(&entry);
    // For strong-ordering purposes we only need to fill in the address.
    eb.__addr_ = addr;
  }
  return ret;
}

} // namespace

int main(int, char**) {
  using vec = std::vector<uintptr_t>;

  auto lt = std::strong_ordering::less;
  auto eq = std::strong_ordering::equal;
  auto gt = std::strong_ordering::greater;

  assert(lt == (fake_trace(vec{}) <=> fake_trace(vec{100})));
  assert(lt == (fake_trace(vec{99}) <=> fake_trace(vec{100})));
  assert(lt == (fake_trace(vec{100}) <=> fake_trace(vec{100, 200})));

  assert(eq == (fake_trace(vec{}) <=> fake_trace(vec{})));
  assert(eq == (fake_trace(vec{100}) <=> fake_trace(vec{100})));

  assert(gt == (fake_trace(vec{100}) <=> fake_trace(vec{})));
  assert(gt == (fake_trace(vec{100}) <=> fake_trace(vec{99})));
  assert(gt == (fake_trace(vec{100, 200}) <=> fake_trace(vec{100})));

  return 0;
}
