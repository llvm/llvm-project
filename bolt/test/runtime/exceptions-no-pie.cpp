// REQUIRES: system-linux
// RUN: %clangxx -no-pie -Wl,-q %s -o %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt.exe -lite=false
// RUN: not --crash %t.bolt.exe 2>&1 | FileCheck %s --check-prefix=CHECK-FAIL
// CHECK-FAIL: Should pass one argument
// RUN: not %t.bolt.exe -1 | FileCheck %s --check-prefix=CHECK-BAD
// CHECK-BAD: Bad value
// RUN: not %t.bolt.exe 0 | FileCheck %s --check-prefix=CHECK-ZERO
// CHECK-ZERO: Value is zero
// RUN: %t.bolt.exe 1 | FileCheck %s --check-prefix=CHECK-GOOD
// CHECK-GOOD: Good value
#include <exception>
#include <iostream>

struct ValIsZero {
  const char *error = "Value is zero\n";
};
int dummy(int arg) {
  if (arg == 0)
    throw ValIsZero();
  if (arg > 0)
    return 0;
  else
    throw std::out_of_range("Bad value");
}

int main(int argc, char **argv) {
  if (argc != 2)
    throw std::invalid_argument("Should pass one argument");
  try {
    dummy(std::strtol(argv[1], nullptr, 10));
  } catch (std::out_of_range &e) {
    std::cout << e.what() << "\n";
    return 1;
  } catch (ValIsZero &e) {
    std::cout << e.error;
    return 1;
  }
  std::cout << "Good value\n";
  return 0;
}
