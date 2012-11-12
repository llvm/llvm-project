// Header for the PCH test asm.c

void f() {
  int i;

  asm ("foo\n" : : "a" (i + 2));
  asm ("foo\n" : [symbolic_name] "=a" (i) : "[symbolic_name]" (i));
}

void clobbers() {
  asm ("nop" : : : "ax", "#ax", "%ax");
  asm ("nop" : : : "eax", "rax", "ah", "al");
  asm ("nop" : : : "0", "%0", "#0");
}
