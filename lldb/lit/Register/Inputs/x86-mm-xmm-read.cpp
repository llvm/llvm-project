#include <cstdint>

struct alignas(16) xmm_t {
  uint64_t a, b;
};

int main() {
  constexpr uint64_t mm[] = {
    0x0001020304050607,
    0x1011121314151617,
    0x2021222324252627,
    0x3031323334353637,
    0x4041424344454647,
    0x5051525354555657,
    0x6061626364656667,
    0x7071727374757677,
  };

  constexpr xmm_t xmm[] = {
    { 0x0706050403020100, 0x0F0E0D0C0B0A0908, },
    { 0x0807060504030201, 0x100F0E0D0C0B0A09, },
    { 0x0908070605040302, 0x11100F0E0D0C0B0A, },
    { 0x0A09080706050403, 0x1211100F0E0D0C0B, },
    { 0x0B0A090807060504, 0x131211100F0E0D0C, },
    { 0x0C0B0A0908070605, 0x14131211100F0E0D, },
    { 0x0D0C0B0A09080706, 0x1514131211100F0E, },
    { 0x0E0D0C0B0A090807, 0x161514131211100F, },
  };

  asm volatile(
    "movq     0x00(%0), %%mm0\n\t"
    "movq     0x08(%0), %%mm1\n\t"
    "movq     0x10(%0), %%mm2\n\t"
    "movq     0x18(%0), %%mm3\n\t"
    "movq     0x20(%0), %%mm4\n\t"
    "movq     0x28(%0), %%mm5\n\t"
    "movq     0x30(%0), %%mm6\n\t"
    "movq     0x38(%0), %%mm7\n\t"
    "\n\t"
    "movaps   0x00(%1), %%xmm0\n\t"
    "movaps   0x10(%1), %%xmm1\n\t"
    "movaps   0x20(%1), %%xmm2\n\t"
    "movaps   0x30(%1), %%xmm3\n\t"
    "movaps   0x40(%1), %%xmm4\n\t"
    "movaps   0x50(%1), %%xmm5\n\t"
    "movaps   0x60(%1), %%xmm6\n\t"
    "movaps   0x70(%1), %%xmm7\n\t"
    "\n\t"
    "int3\n\t"
    :
    : "a"(mm), "b"(xmm)
    : "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7",
      "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
  );

  return 0;
}
