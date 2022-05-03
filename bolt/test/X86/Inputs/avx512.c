void use_avx512() {
  asm (".byte 0x62, 0xe2, 0xf5, 0x70, 0x2c, 0xda");
  asm("secondary_entry:");
}

int main() {
  use_avx512();

  return 0;
}
