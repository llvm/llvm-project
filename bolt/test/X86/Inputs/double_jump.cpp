/*
 * A contrived example to test the double jump removal peephole.
 */

extern "C" unsigned long bar(unsigned long count) {
  return count + 1;
}

unsigned long foo(unsigned long count) {
  asm(
      "     cmpq  $1,%0\n"
      "     je    .L7\n"
      "     incq  %0\n"
      "     jmp   .L1\n"
      ".L1: jmp   .L2\n"
      ".L2: incq  %0\n"
      "     cmpq  $2,%0\n"
      "     jne   .L3\n"
      "     jmp   .L4\n"
      ".L3: jmp   .L5\n"
      ".L5: incq  %0\n"
      ".L4: movq  %0,%%rdi\n"
      "     pop   %%rbp\n"
      "     jmp   .L6\n"
      ".L7: pop   %%rbp\n"
      "     incq  %0\n"
      "     jmp   .L6\n"
      ".L6: jmp   bar\n"
      :
      : "m"(count)
      );
  return count;
}

int main(int argc, const char* argv[]) {
  return foo(38);
}
