// Test llvm::Expected<T> data formatters.

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <cstdio>

using namespace llvm;

int main() {
  // Test primitive type (storage is T directly).
  Expected<int> ExpectedInt = 42;
  (void)static_cast<bool>(ExpectedInt);

  // Test pointer type (storage is T* directly).
  int x = 10;
  Expected<int *> ExpectedPtr = &x;
  (void)static_cast<bool>(ExpectedPtr);

  // Test reference type (storage is std::reference_wrapper<T>).
  int y = 100;
  Expected<int &> ExpectedRef = y;
  (void)static_cast<bool>(ExpectedRef);

  // Test templated type (storage is the template type directly).
  Expected<SmallVector<int, 2>> ExpectedVec = SmallVector<int, 2>{1, 2};
  (void)static_cast<bool>(ExpectedVec);

  // Test templated reference type (storage is std::reference_wrapper<T>).
  SmallVector<int, 2> vec{3, 4};
  Expected<SmallVector<int, 2> &> ExpectedVecRef = vec;
  (void)static_cast<bool>(ExpectedVecRef);

  puts("Break here");

  return 0;
}
