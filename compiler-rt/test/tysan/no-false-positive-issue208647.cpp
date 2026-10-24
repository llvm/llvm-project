// RUN: %clangxx_tysan -O0 %s -o %t && %run %t

#include <variant>

int main() {
  std::variant<int, double> v;
  v = 1;
  v = 3.5;
  return 0;
}
