#include <stdio.h>

void use_avx512() {
  printf("after first entry\n");
  asm (".byte 0x62, 0xe2, 0xf5, 0x70, 0x2c, 0xda");
  asm ("secondary_entry:");
  printf("after secondary entry\n");
}

int main() {
  printf("about to use avx-512 instruction...\n");
  use_avx512();

  return 0;
}
