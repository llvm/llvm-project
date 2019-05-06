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
    "movq     0x00(%%rax), %%mm0\n\t"
    "movq     0x08(%%rax), %%mm1\n\t"
    "movq     0x10(%%rax), %%mm2\n\t"
    "movq     0x18(%%rax), %%mm3\n\t"
    "movq     0x20(%%rax), %%mm4\n\t"
    "movq     0x28(%%rax), %%mm5\n\t"
    "movq     0x30(%%rax), %%mm6\n\t"
    "movq     0x38(%%rax), %%mm7\n\t"
    "\n\t"
    "movaps   0x00(%%rbx), %%xmm0\n\t"
    "movaps   0x10(%%rbx), %%xmm1\n\t"
    "movaps   0x20(%%rbx), %%xmm2\n\t"
    "movaps   0x30(%%rbx), %%xmm3\n\t"
    "movaps   0x40(%%rbx), %%xmm4\n\t"
    "movaps   0x50(%%rbx), %%xmm5\n\t"
    "movaps   0x60(%%rbx), %%xmm6\n\t"
    "movaps   0x70(%%rbx), %%xmm7\n\t"
    "\n\t"
    "int3\n\t"
    :
    : "a"(mm), "b"(xmm)
    : "%mm0", "%mm1", "%mm2", "%mm3", "%mm4", "%mm5", "%mm6", "%mm7",
      "%xmm0", "%xmm1", "%xmm2", "%xmm3", "%xmm4", "%xmm5", "%xmm6", "%xmm7"
  );

  return 0;
}
