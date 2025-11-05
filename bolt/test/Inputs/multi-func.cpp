#include <iostream>

// Multiple functions to test selective dumping
int add(int a, int b) { return a + b; }

int multiply(int a, int b) { return a * b; }

int main_helper() {
  std::cout << "Helper function" << std::endl;
  return 42;
}

int main_secondary() { return add(5, 3); }

void other_function() { std::cout << "Other function" << std::endl; }

int main() {
  int result = add(10, 20);
  result = multiply(result, 2);
  main_helper();
  main_secondary();
  other_function();
  return result;
}
