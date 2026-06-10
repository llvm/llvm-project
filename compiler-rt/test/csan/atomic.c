// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | count 0

int global;

int main(void) {
  for (int i = 0; i < 1024; ++i)
    __atomic_fetch_add(&global, 1, __ATOMIC_RELAXED);
  return 0;
}
