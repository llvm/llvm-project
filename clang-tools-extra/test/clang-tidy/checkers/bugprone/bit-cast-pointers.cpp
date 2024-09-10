// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-bit-cast-pointers %t

namespace std
{
template <typename To, typename From>
To bit_cast(From from)
{
  // Dummy implementation for the purpose of the check.
  // We don't want to include <cstring> to get std::memcpy.
  To to{};
  return to;
}
}

void pointer2pointer()
{
  int x{};
  float bad = *std::bit_cast<float*>(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use std::bit_cast on pointers; use reinterpret_cast instead [bugprone-bit-cast-pointers]
  float good = *reinterpret_cast<float*>(&x);
  float good2 = std::bit_cast<float>(x);
}

void int2pointer()
{
  unsigned long long addr{};
  float* bad = std::bit_cast<float*>(addr);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use std::bit_cast on pointers; use reinterpret_cast instead [bugprone-bit-cast-pointers]
  float* good = reinterpret_cast<float*>(addr);
}

void pointer2int()
{
  int x{};
  auto bad = std::bit_cast<unsigned long long>(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: do not use std::bit_cast on pointers; use reinterpret_cast instead [bugprone-bit-cast-pointers]
  auto good = reinterpret_cast<unsigned long long>(&x);
}
