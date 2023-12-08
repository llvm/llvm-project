// RUN: %clang %cflags -O2 -fPIE -Wl,-q -pie  %s -o %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
// CHECK-NOT: BOLT-WARNING: unable to disassemble instruction at offset

void extra_space() {
  asm volatile(".rept 256\n"
               "    .byte 0xff\n"
               ".endr\n");
  return;
}

int main(int argc, char **argv) {
  void (*fn)(void);
  fn = extra_space + 256;
  fn();
  return 0;
}
