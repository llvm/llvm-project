#include <cinttypes>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr uint64_t fill = 0x0F0F0F0F0F0F0F0F;

  uint64_t rax, rbx, rcx, rdx, rsp, rbp, rsi, rdi;

  asm volatile(
    // save rsp & rbp
    "movq    %%rsp, %4\n\t"
    "movq    %%rbp, %5\n\t"
    "\n\t"
    "movq    %8, %%rax\n\t"
    "movq    %8, %%rbx\n\t"
    "movq    %8, %%rcx\n\t"
    "movq    %8, %%rdx\n\t"
    "movq    %8, %%rsp\n\t"
    "movq    %8, %%rbp\n\t"
    "movq    %8, %%rsi\n\t"
    "movq    %8, %%rdi\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    // swap saved & current rsp & rbp
    "xchgq    %%rsp, %4\n\t"
    "xchgq    %%rbp, %5\n\t"
    : "=a"(rax), "=b"(rbx), "=c"(rcx), "=d"(rdx), "=r"(rsp), "=r"(rbp),
      "=S"(rsi), "=D"(rdi)
    : "g"(fill)
    :
  );

  printf("rax = 0x%016" PRIx64 "\n", rax);
  printf("rbx = 0x%016" PRIx64 "\n", rbx);
  printf("rcx = 0x%016" PRIx64 "\n", rcx);
  printf("rdx = 0x%016" PRIx64 "\n", rdx);
  printf("rsp = 0x%016" PRIx64 "\n", rsp);
  printf("rbp = 0x%016" PRIx64 "\n", rbp);
  printf("rsi = 0x%016" PRIx64 "\n", rsi);
  printf("rdi = 0x%016" PRIx64 "\n", rdi);

  return 0;
}
