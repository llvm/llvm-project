// RUN: %check_clang_tidy %s bugprone-bitwise-pointer-cast %t

void memcpy(void* to, void* dst, unsigned long long size)
{
  // Dummy implementation for the purpose of the test
}

namespace std
{
using ::memcpy;
}

void pointer2pointer()
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
  float* p{};
  std::memcpy(&p, &addr, sizeof(addr));
}

void pointer2int()
{
  unsigned long long addr{};
  float* p{};
  std::memcpy(&addr, &p, sizeof(p));
}
