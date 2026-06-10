// RUN: %clang_csan %s -o %t && %run --threads 1 --blocks 1 %t 2>&1 | count 0

int global;

int main(void) {
  for (int i = 0; i < 1024; ++i)
    global++;
  return 0;
}
