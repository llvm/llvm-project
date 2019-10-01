#include <cstdint>

struct alignas(32) ymm_t {
  uint64_t a, b, c, d;
};

int main() {
  constexpr ymm_t ymm[] = {
    { 0x0706050403020100, 0x0F0E0D0C0B0A0908,
      0x1716151413121110, 0x1F1E1D1C1B1A1918, },
    { 0x0807060504030201, 0x100F0E0D0C0B0A09,
      0x1817161514131211, 0x201F1E1D1C1B1A19, },
    { 0x0908070605040302, 0x11100F0E0D0C0B0A,
      0x1918171615141312, 0x21201F1E1D1C1B1A, },
    { 0x0A09080706050403, 0x1211100F0E0D0C0B,
      0x1A19181716151413, 0x2221201F1E1D1C1B, },
    { 0x0B0A090807060504, 0x131211100F0E0D0C,
      0x1B1A191817161514, 0x232221201F1E1D1C, },
    { 0x0C0B0A0908070605, 0x14131211100F0E0D,
      0x1C1B1A1918171615, 0x24232221201F1E1D, },
    { 0x0D0C0B0A09080706, 0x1514131211100F0E,
      0x1D1C1B1A19181716, 0x2524232221201F1E, },
    { 0x0E0D0C0B0A090807, 0x161514131211100F,
      0x1E1D1C1B1A191817, 0x262524232221201F, },
#if defined(__x86_64__) || defined(_M_X64)
    { 0x0F0E0D0C0B0A0908, 0x1716151413121110,
      0x1F1E1D1C1B1A1918, 0x2726252423222120, },
    { 0x100F0E0D0C0B0A09, 0x1817161514131211,
      0x201F1E1D1C1B1A19, 0x2827262524232221, },
    { 0x11100F0E0D0C0B0A, 0x1918171615141312,
      0x21201F1E1D1C1B1A, 0x2928272625242322, },
    { 0x1211100F0E0D0C0B, 0x1A19181716151413,
      0x2221201F1E1D1C1B, 0x2A29282726252423, },
    { 0x131211100F0E0D0C, 0x1B1A191817161514,
      0x232221201F1E1D1C, 0x2B2A292827262524, },
    { 0x14131211100F0E0D, 0x1C1B1A1918171615,
      0x24232221201F1E1D, 0x2C2B2A2928272625, },
    { 0x1514131211100F0E, 0x1D1C1B1A19181716,
      0x2524232221201F1E, 0x2D2C2B2A29282726, },
    { 0x161514131211100F, 0x1E1D1C1B1A191817,
      0x262524232221201F, 0x2E2D2C2B2A292827, },
#endif
  };

  asm volatile(
    "vmovaps  0x000(%0), %%ymm0\n\t"
    "vmovaps  0x020(%0), %%ymm1\n\t"
    "vmovaps  0x040(%0), %%ymm2\n\t"
    "vmovaps  0x060(%0), %%ymm3\n\t"
    "vmovaps  0x080(%0), %%ymm4\n\t"
    "vmovaps  0x0A0(%0), %%ymm5\n\t"
    "vmovaps  0x0C0(%0), %%ymm6\n\t"
    "vmovaps  0x0E0(%0), %%ymm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  0x100(%0), %%ymm8\n\t"
    "vmovaps  0x120(%0), %%ymm9\n\t"
    "vmovaps  0x140(%0), %%ymm10\n\t"
    "vmovaps  0x160(%0), %%ymm11\n\t"
    "vmovaps  0x180(%0), %%ymm12\n\t"
    "vmovaps  0x1A0(%0), %%ymm13\n\t"
    "vmovaps  0x1C0(%0), %%ymm14\n\t"
    "vmovaps  0x1E0(%0), %%ymm15\n\t"
#endif
    "\n\t"
    "int3\n\t"
    :
    : "b"(ymm)
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7"
#if defined(__x86_64__) || defined(_M_X64)
    , "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
      "%ymm15"
#endif
  );

  return 0;
}
