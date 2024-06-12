#include <exception>
#include <stdexcept>
void foo(int a) {
  if (!a)
    throw std::out_of_range("bad value");
  return;
}

int main() {
  try {
    foo(1);
    foo(0);
  } catch (...) {
  }
  std::terminate();
  return 0;
}
