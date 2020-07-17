// REQUIRES: clang-11

#include <atomic>

int main(int, char**)
{
  // expected-error@atomic:*1 {{_Atomic cannot be applied to integer type '_ExtInt(32)'}}
  std::atomic<_ExtInt(32)> x {42};

  return 0;
}
