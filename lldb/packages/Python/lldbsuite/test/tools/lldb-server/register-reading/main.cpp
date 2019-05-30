#include <cstdint>

struct alignas(16) xmm_t {
  uint64_t a, b;
};

int main() {
  uint64_t r8 = 0x0102030405060708;
  uint64_t r9 = 0x1112131415161718;
  uint64_t r10 = 0x2122232425262728;
  uint64_t r11 = 0x3132333435363738;
  uint64_t r12 = 0x4142434445464748;
  uint64_t r13 = 0x5152535455565758;
  uint64_t r14 = 0x6162636465666768;
  uint64_t r15 = 0x7172737475767778;

  xmm_t xmm8 = {0x020406080A0C0E01, 0x030507090B0D0F00};
  xmm_t xmm9 = {0x121416181A1C1E11, 0x131517191B1D1F10};
  xmm_t xmm10 = {0x222426282A2C2E21, 0x232527292B2D2F20};
  xmm_t xmm11 = {0x323436383A3C3E31, 0x333537393B3D3F30};
  xmm_t xmm12 = {0x424446484A4C4E41, 0x434547494B4D4F40};
  xmm_t xmm13 = {0x525456585A5C5E51, 0x535557595B5D5F50};
  xmm_t xmm14 = {0x626466686A6C6E61, 0x636567696B6D6F60};
  xmm_t xmm15 = {0x727476787A7C7E71, 0x737577797B7D7F70};

  asm volatile("movq    %0, %%r8\n\t"
               "movq    %1, %%r9\n\t"
               "movq    %2, %%r10\n\t"
               "movq    %3, %%r11\n\t"
               "movq    %4, %%r12\n\t"
               "movq    %5, %%r13\n\t"
               "movq    %6, %%r14\n\t"
               "movq    %7, %%r15\n\t"
               "\n\t"
               "movaps  %8, %%xmm8\n\t"
               "movaps  %9, %%xmm9\n\t"
               "movaps  %10, %%xmm10\n\t"
               "movaps  %11, %%xmm11\n\t"
               "movaps  %12, %%xmm12\n\t"
               "movaps  %13, %%xmm13\n\t"
               "movaps  %14, %%xmm14\n\t"
               "movaps  %15, %%xmm15\n\t"
               "\n\t"
               "int3"
               :
               : "g"(r8), "g"(r9), "g"(r10), "g"(r11), "g"(r12), "g"(r13),
                 "g"(r14), "g"(r15), "m"(xmm8), "m"(xmm9), "m"(xmm10),
                 "m"(xmm11), "m"(xmm12), "m"(xmm13), "m"(xmm14), "m"(xmm15)
               : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15",
                 "%xmm8", "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13",
                 "%xmm14", "%xmm15");

  return 0;
}
