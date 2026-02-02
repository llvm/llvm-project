// Test llvm::Expected<T> data formatters.

#include "llvm/Support/Error.h"
#include <cstdio>

using namespace llvm;

int main() {
  // Test non-reference types (storage is T directly, no template args).
  Expected<int> ExpectedInt = 42;
  (void)static_cast<bool>(ExpectedInt);

  int x = 10;
  Expected<int *> ExpectedPtr = &x;
  (void)static_cast<bool>(ExpectedPtr);

  // Test reference type (storage is std::reference_wrapper<T>).
  int y = 100;
  Expected<int &> ExpectedRef = y;
  (void)static_cast<bool>(ExpectedRef);

  puts("Break here");

  return 0;
}
