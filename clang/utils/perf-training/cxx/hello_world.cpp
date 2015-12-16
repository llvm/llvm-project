// RUN: %clang_cpp -c %s
#include <iostream>

int main(int, char**) {
  std::cout << "Hello, World!";
  return 0;
}
