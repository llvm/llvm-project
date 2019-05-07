#include <cinttypes>
#include <cstdint>
#include <cstdio>

union alignas(64) zmm_t {
  uint64_t as_uint64[8];
  uint8_t as_uint8[64];
};

int main() {
  constexpr zmm_t zmm_fill = {
    .as_uint64 = { 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F,
                   0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F,
                   0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F,
                   0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F }
  };

  zmm_t zmm[32];

  asm volatile(
    "vmovaps  %1, %%zmm0\n\t"
    "vmovaps  %1, %%zmm1\n\t"
    "vmovaps  %1, %%zmm2\n\t"
    "vmovaps  %1, %%zmm3\n\t"
    "vmovaps  %1, %%zmm4\n\t"
    "vmovaps  %1, %%zmm5\n\t"
    "vmovaps  %1, %%zmm6\n\t"
    "vmovaps  %1, %%zmm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  %1, %%zmm8\n\t"
    "vmovaps  %1, %%zmm9\n\t"
    "vmovaps  %1, %%zmm10\n\t"
    "vmovaps  %1, %%zmm11\n\t"
    "vmovaps  %1, %%zmm12\n\t"
    "vmovaps  %1, %%zmm13\n\t"
    "vmovaps  %1, %%zmm14\n\t"
    "vmovaps  %1, %%zmm15\n\t"
    "vmovaps  %1, %%zmm16\n\t"
    "vmovaps  %1, %%zmm17\n\t"
    "vmovaps  %1, %%zmm18\n\t"
    "vmovaps  %1, %%zmm19\n\t"
    "vmovaps  %1, %%zmm20\n\t"
    "vmovaps  %1, %%zmm21\n\t"
    "vmovaps  %1, %%zmm22\n\t"
    "vmovaps  %1, %%zmm23\n\t"
    "vmovaps  %1, %%zmm24\n\t"
    "vmovaps  %1, %%zmm25\n\t"
    "vmovaps  %1, %%zmm26\n\t"
    "vmovaps  %1, %%zmm27\n\t"
    "vmovaps  %1, %%zmm28\n\t"
    "vmovaps  %1, %%zmm29\n\t"
    "vmovaps  %1, %%zmm30\n\t"
    "vmovaps  %1, %%zmm31\n\t"
#endif
    "\n\t"
    "int3\n\t"
    "\n\t"
    "vmovaps %%zmm0,  0x000(%0)\n\t"
    "vmovaps %%zmm1,  0x040(%0)\n\t"
    "vmovaps %%zmm2,  0x080(%0)\n\t"
    "vmovaps %%zmm3,  0x0C0(%0)\n\t"
    "vmovaps %%zmm4,  0x100(%0)\n\t"
    "vmovaps %%zmm5,  0x140(%0)\n\t"
    "vmovaps %%zmm6,  0x180(%0)\n\t"
    "vmovaps %%zmm7,  0x1C0(%0)\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps %%zmm8,  0x200(%0)\n\t"
    "vmovaps %%zmm9,  0x240(%0)\n\t"
    "vmovaps %%zmm10, 0x280(%0)\n\t"
    "vmovaps %%zmm11, 0x2C0(%0)\n\t"
    "vmovaps %%zmm12, 0x300(%0)\n\t"
    "vmovaps %%zmm13, 0x340(%0)\n\t"
    "vmovaps %%zmm14, 0x380(%0)\n\t"
    "vmovaps %%zmm15, 0x3C0(%0)\n\t"
    "vmovaps %%zmm16, 0x400(%0)\n\t"
    "vmovaps %%zmm17, 0x440(%0)\n\t"
    "vmovaps %%zmm18, 0x480(%0)\n\t"
    "vmovaps %%zmm19, 0x4C0(%0)\n\t"
    "vmovaps %%zmm20, 0x500(%0)\n\t"
    "vmovaps %%zmm21, 0x540(%0)\n\t"
    "vmovaps %%zmm22, 0x580(%0)\n\t"
    "vmovaps %%zmm23, 0x5C0(%0)\n\t"
    "vmovaps %%zmm24, 0x600(%0)\n\t"
    "vmovaps %%zmm25, 0x640(%0)\n\t"
    "vmovaps %%zmm26, 0x680(%0)\n\t"
    "vmovaps %%zmm27, 0x6C0(%0)\n\t"
    "vmovaps %%zmm28, 0x700(%0)\n\t"
    "vmovaps %%zmm29, 0x740(%0)\n\t"
    "vmovaps %%zmm30, 0x780(%0)\n\t"
    "vmovaps %%zmm31, 0x7C0(%0)\n\t"
#endif
    :
    : "b"(zmm), "m"(zmm_fill)
    : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7"
#if defined(__x86_64__) || defined(_M_X64)
    , "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
      "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21",
      "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28",
      "%zmm29", "%zmm30", "%zmm31"
#endif
  );

  for (int i = 0; i < 32; ++i) {
    printf("zmm%d = { ", i);
    for (int j = 0; j < sizeof(zmm->as_uint8); ++j)
      printf("0x%02x ", zmm[i].as_uint8[j]);
    printf("}\n");
  }

  return 0;
}
