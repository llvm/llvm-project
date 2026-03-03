// RUN: %clangxx_lowfat %s -o %t
// RUN: not %run %t 2>&1 | FileCheck %s

// Verify that a cross-boundary out-of-bounds access is detected.
// Writing 8 bytes starting at offset 12 of a 16-byte slot crosses
// the slot boundary (bytes 12-19 > 16 bytes).

extern "C" void *__lf_malloc(unsigned long size);

int main() {
  // Allocate exactly 16 bytes → 16-byte size class (smallest slot)
  char *buf = (char *)__lf_malloc(16);
  if (!buf)
    return 1;

  // In-bounds write
  buf[0] = 'A';
  buf[15] = 'B';

  // Cross-boundary OOB: write 8 bytes at offset 12
  // ptr + 8 = buf+20 > buf+16 → out of bounds
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  double *cross = (double *)(buf + 12);
  *cross = 3.14;

  return 0;
}
