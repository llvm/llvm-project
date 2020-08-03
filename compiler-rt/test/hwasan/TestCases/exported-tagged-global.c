// RUN: %clang_hwasan %s -o %t
// RUN: %run %t
// RUN: %clang_hwasan -O1 %s -o %t
// RUN: %run %t
// RUN: %clang_hwasan -O1 -mllvm --aarch64-enable-global-isel-at-O=1 %s -o %t
// RUN: %run %t

static int global;

__attribute__((optnone)) int *address_of_global() { return &global; }

int main(int argc, char **argv) {
  int *global_address = address_of_global();
  *global_address = 13;
  return 0;
}
