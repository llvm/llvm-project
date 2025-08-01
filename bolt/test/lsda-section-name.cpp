// This test check that LSDA section named by .gcc_except_table.main is
// disassembled by BOLT.

// RUN: %clangxx %cxxflags -O3 -no-pie -c %s -o %t.o
// RUN: %clangxx %cxxflags -O3 -no-pie -fuse-ld=lld %t.o -o %t
// RUN: llvm-objcopy --rename-section .gcc_except_table=.gcc_except_table.main %t
// RUN: llvm-readelf -SW %t | FileCheck %s
// RUN: llvm-bolt %t -o %t.bolt

// CHECK: .gcc_except_table.main

#include <iostream>

class MyException : public std::exception {
public:
  const char *what() const throw() {
    return "Custom Exception: an error occurred!";
  }
};

int divide(int a, int b) {
  if (b == 0) {
    throw MyException();
  }
  return a / b;
}

int main() {
  try {
    int result = divide(10, 2); // normal case
    std::cout << "Result: " << result << std::endl;
    result = divide(5, 0); // will cause exception
    std::cout << "Result: " << result << std::endl;
    // this line will not execute
  } catch (const MyException &e) {
    // catch custom exception
    std::cerr << "Caught exception: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    // catch other C++ exceptions
    std::cerr << "Caught exception: " << e.what() << std::endl;
  } catch (...) {
    // catch all other exceptions
    std::cerr << "Caught unknown exception" << std::endl;
  }

  return 0;
}
