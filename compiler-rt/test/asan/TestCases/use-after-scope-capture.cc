// RUN: %clangxx_asan -O0 -fsanitize=use-after-scope %s -o %t && %run %t
// XFAIL: *

int main() {
  std::function<int()> f;
  {
    int x = 0;
    f = [&x]() {
      return x;
    }
  }
  return f();  // BOOM
  // CHECK: ERROR: AddressSanitizer: stack-use-after-scope
}
