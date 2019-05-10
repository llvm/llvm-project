#include <cinttypes>
#include <cstdint>
#include <cstdio>

union alignas(16) xmm_t {
  uint64_t as_uint64[2];
  uint8_t as_uint8[16];
};

int main() {
  constexpr xmm_t xmm_fill = {
    .as_uint64 = { 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F }
  };

  uint64_t r64[8];
  xmm_t xmm[8];

  asm volatile(
    "movq    %2, %%r8\n\t"
    "movq    %2, %%r9\n\t"
    "movq    %2, %%r10\n\t"
    "movq    %2, %%r11\n\t"
    "movq    %2, %%r12\n\t"
    "movq    %2, %%r13\n\t"
    "movq    %2, %%r14\n\t"
    "movq    %2, %%r15\n\t"
    "\n\t"
    "movaps  %2, %%xmm8\n\t"
    "movaps  %2, %%xmm9\n\t"
    "movaps  %2, %%xmm10\n\t"
    "movaps  %2, %%xmm11\n\t"
    "movaps  %2, %%xmm12\n\t"
    "movaps  %2, %%xmm13\n\t"
    "movaps  %2, %%xmm14\n\t"
    "movaps  %2, %%xmm15\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    "movq    %%r8, 0x00(%0)\n\t"
    "movq    %%r9, 0x08(%0)\n\t"
    "movq    %%r10, 0x10(%0)\n\t"
    "movq    %%r11, 0x18(%0)\n\t"
    "movq    %%r12, 0x20(%0)\n\t"
    "movq    %%r13, 0x28(%0)\n\t"
    "movq    %%r14, 0x30(%0)\n\t"
    "movq    %%r15, 0x38(%0)\n\t"
    "\n\t"
    "movaps  %%xmm8, 0x00(%1)\n\t"
    "movaps  %%xmm9, 0x10(%1)\n\t"
    "movaps  %%xmm10, 0x20(%1)\n\t"
    "movaps  %%xmm11, 0x30(%1)\n\t"
    "movaps  %%xmm12, 0x40(%1)\n\t"
    "movaps  %%xmm13, 0x50(%1)\n\t"
    "movaps  %%xmm14, 0x60(%1)\n\t"
    "movaps  %%xmm15, 0x70(%1)\n\t"
    :
    : "a"(r64), "b"(xmm), "m"(xmm_fill)
    : "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14", "%r15", "%xmm8",
      "%xmm9", "%xmm10", "%xmm11", "%xmm12", "%xmm13", "%xmm14", "%xmm15"
  );

  for (int i = 0; i < 8; ++i)
    printf("r%d = 0x%016" PRIx64 "\n", i+8, r64[i]);
  for (int i = 0; i < 8; ++i) {
    printf("xmm%d = { ", i+8);
    for (int j = 0; j < sizeof(xmm->as_uint8); ++j)
      printf("0x%02x ", xmm[i].as_uint8[j]);
    printf("}\n");
  }

  return 0;
}
