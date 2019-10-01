// Header for the PCH test asm.c

void f() {
  int i,cond;

  asm ("foo\n" : : "a" (i + 2));
  asm ("foo\n" : [symbolic_name] "=a" (i) : "[symbolic_name]" (i));
  asm volatile goto("testl %0, %0; jne %l1;" :: "r"(cond)::label_true, loop);
label_true:
loop:
  return;
}

void clobbers() {
  asm ("nop" : : : "ax", "#ax", "%ax");
  asm ("nop" : : : "eax", "rax", "ah", "al");
  asm ("nop" : : : "0", "%0", "#0");
}
