#include <cstdint>

int main() {
  constexpr uint64_t rax = 0x0102030405060708;
  constexpr uint64_t rbx = 0x1112131415161718;
  constexpr uint64_t rcx = 0x2122232425262728;
  constexpr uint64_t rdx = 0x3132333435363738;
  constexpr uint64_t rsp = 0x4142434445464748;
  constexpr uint64_t rbp = 0x5152535455565758;
  constexpr uint64_t rsi = 0x6162636465666768;
  constexpr uint64_t rdi = 0x7172737475767778;

  asm volatile(
    // save rsp & rbp
    "movq    %%rsp, %%r8\n\t"
    "movq    %%rbp, %%r9\n\t"
    "\n\t"
    "movq    %0, %%rax\n\t"
    "movq    %1, %%rbx\n\t"
    "movq    %2, %%rcx\n\t"
    "movq    %3, %%rdx\n\t"
    "movq    %4, %%rsp\n\t"
    "movq    %5, %%rbp\n\t"
    "movq    %6, %%rsi\n\t"
    "movq    %7, %%rdi\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    // restore rsp & rbp
    "movq    %%r8, %%rsp\n\t"
    "movq    %%r9, %%rbp"
    :
    : "i"(rax), "i"(rbx), "i"(rcx), "i"(rdx), "i"(rsp), "i"(rbp), "i"(rsi),
      "i"(rdi)
    : "%rax", "%rbx", "%rcx", "%rdx", "%rsp", "%rbp", "%rsi", "%rdi", "%r8",
      "%r9"
  );

  return 0;
}
