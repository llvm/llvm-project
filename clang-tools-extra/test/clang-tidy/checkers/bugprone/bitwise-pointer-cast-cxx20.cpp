// RUN: %check_clang_tidy -std=c++20 %s bugprone-bitwise-pointer-cast %t

void memcpy(void* to, void* dst, unsigned long long size)
{
  // Dummy implementation for the purpose of the test
}

namespace std
{
template <typename To, typename From>
To bit_cast(From from)
{
  // Dummy implementation for the purpose of the test
  To to{};
  return to;
}

using ::memcpy;
}

void pointer2pointer()
{
  int x{};
  float bad = *std::bit_cast<float*>(&x); // UB, but looks safe due to std::bit_cast
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use 'std::bit_cast' to cast between pointers [bugprone-bitwise-pointer-cast]
  float good = std::bit_cast<float>(x);   // Well-defined

  using IntPtr = int*;
  using FloatPtr = float*;
  IntPtr x2{};
  float bad2 = *std::bit_cast<FloatPtr>(x2);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: do not use 'std::bit_cast' to cast between pointers [bugprone-bitwise-pointer-cast]
}

void pointer2pointer_memcpy()
{
  int x{};
  int* px{};
  float y{};
  float* py{};

  memcpy(&py, &px, sizeof(px));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'memcpy' to cast between pointers [bugprone-bitwise-pointer-cast]
  std::memcpy(&py, &px, sizeof(px));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'memcpy' to cast between pointers [bugprone-bitwise-pointer-cast]

  std::memcpy(&y, &x, sizeof(x));
}

// Pointer-integer conversions are allowed by this check
void int2pointer()
{
  unsigned long long addr{};
  float* p = std::bit_cast<float*>(addr);
  std::memcpy(&p, &addr, sizeof(addr));
}

void pointer2int()
{
  float* p{};
  auto addr = std::bit_cast<unsigned long long>(p);
  std::memcpy(&addr, &p, sizeof(p));
}
