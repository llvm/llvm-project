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

  uint64_t mm[8];
  xmm_t xmm[8];

  asm volatile(
    "movq    %2, %%mm0\n\t"
    "movq    %2, %%mm1\n\t"
    "movq    %2, %%mm2\n\t"
    "movq    %2, %%mm3\n\t"
    "movq    %2, %%mm4\n\t"
    "movq    %2, %%mm5\n\t"
    "movq    %2, %%mm6\n\t"
    "movq    %2, %%mm7\n\t"
    "\n\t"
    "movaps  %2, %%xmm0\n\t"
    "movaps  %2, %%xmm1\n\t"
    "movaps  %2, %%xmm2\n\t"
    "movaps  %2, %%xmm3\n\t"
    "movaps  %2, %%xmm4\n\t"
    "movaps  %2, %%xmm5\n\t"
    "movaps  %2, %%xmm6\n\t"
    "movaps  %2, %%xmm7\n\t"
    "\n\t"
    "int3\n\t"
    "\n\t"
    "movq    %%mm0, 0x00(%0)\n\t"
    "movq    %%mm1, 0x08(%0)\n\t"
    "movq    %%mm2, 0x10(%0)\n\t"
    "movq    %%mm3, 0x18(%0)\n\t"
    "movq    %%mm4, 0x20(%0)\n\t"
    "movq    %%mm5, 0x28(%0)\n\t"
    "movq    %%mm6, 0x30(%0)\n\t"
    "movq    %%mm7, 0x38(%0)\n\t"
    "\n\t"
    "movaps  %%xmm0, 0x00(%1)\n\t"
    "movaps  %%xmm1, 0x10(%1)\n\t"
    "movaps  %%xmm2, 0x20(%1)\n\t"
    "movaps  %%xmm3, 0x30(%1)\n\t"
    "movaps  %%xmm4, 0x40(%1)\n\t"
    "movaps  %%xmm5, 0x50(%1)\n\t"
    "movaps  %%xmm6, 0x60(%1)\n\t"
    "movaps  %%xmm7, 0x70(%1)\n\t"
    :
    : "a"(mm), "b"(xmm), "m"(xmm_fill)
    : "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7", "%xmm0",
      "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
  );

  for (int i = 0; i < 8; ++i)
    printf("mm%d = 0x%016" PRIx64 "\n", i, mm[i]);
  for (int i = 0; i < 8; ++i) {
    printf("xmm%d = { ", i);
    for (int j = 0; j < sizeof(xmm->as_uint8); ++j)
      printf("0x%02x ", xmm[i].as_uint8[j]);
    printf("}\n");
  }

  return 0;
}
