// RUN: %clangxx_asan -O3 -flto %s -o %t
// RUN: %run %t 2>&1

// REQUIRES: lto

int main(int argc, const char * argv[]) {
  struct { long width, height; } a = {16, 16};
  struct { long width, height; } b = {16, 16};

  // Just to make sure 'a' and 'b' don't get optimized out.
  asm volatile("" : : "r" (&a), "r" (&b));

  return 0;
}
