// RUN: %clangxx_lowfat -O0 %s -o %t
// RUN: %clangxx_lowfat -O1 %s -o %t
// RUN: %clangxx_lowfat -O2 %s -o %t
// RUN: %clangxx_lowfat -O3 %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Cross-boundary OOB write must be reported.

extern "C" void *__lf_malloc(unsigned long size);

int main() {
  // Allocate exactly 16 bytes.
  char *buf = (char *)__lf_malloc(16);
  if (!buf)
    return 1;

  // In-bounds write
  buf[0] = 'A';
  buf[15] = 'B';

  // Cross-boundary OOB: write 8 bytes at offset 12
  // ptr + 8 = buf+20 > buf+16: out of bounds.
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  double *cross = (double *)(buf + 12);
  *cross = 3.14;

  return 0;
}
