// RUN: %clangxx_msan -O0 %s -o %t && %run %t

int main(void) {
  int i;
  __sync_lock_test_and_set(&i, 0);
  return i;
}
