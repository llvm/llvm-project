#include <cinttypes>
#include <cstdint>
#include <cstdio>

int main() {
  constexpr uint64_t fill = 0x0F0F0F0F0F0F0F0F;

  uint64_t rax, rbx, rcx, rdx, rsp, rbp, rsi, rdi;

  asm volatile(
    // save rsp & rbp
    "movq    %%rsp, %%mm0\n\t"
    "movq    %%rbp, %%mm1\n\t"
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
    "movq    %%rax, %0\n\t"
    "movq    %%rbx, %1\n\t"
    "movq    %%rcx, %2\n\t"
    "movq    %%rdx, %3\n\t"
    "movq    %%rsp, %4\n\t"
    "movq    %%rbp, %5\n\t"
    "movq    %%rsi, %6\n\t"
    "movq    %%rdi, %7\n\t"
    "\n\t"
    // restore rsp & rbp
    "movq    %%mm0, %%rsp\n\t"
    "movq    %%mm1, %%rbp\n\t"
    : "=r"(rax), "=r"(rbx), "=r"(rcx), "=r"(rdx), "=r"(rsp), "=r"(rbp),
      "=r"(rsi), "=r"(rdi)
    : "g"(fill)
    : "%rax", "%rbx", "%rcx", "%rdx", "%rsp", "%rbp", "%rsi", "%rdi", "%mm0",
      "%mm1"
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
