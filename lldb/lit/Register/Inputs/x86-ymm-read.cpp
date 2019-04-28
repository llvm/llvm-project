#include <cstdint>

struct alignas(32) ymm_t {
  uint64_t a, b, c, d;
};

int main() {
  constexpr ymm_t ymm0 = { 0x020406080A0C0E01, 0x030507090B0D0F00,
                           0x838587898B8D8F80, 0x828486888A8C8E81 };
  constexpr ymm_t ymm1 = { 0x121416181A1C1E11, 0x131517191B1D1F10,
                           0x939597999B9D9F90, 0x929496989A9C9E91 };
  constexpr ymm_t ymm2 = { 0x222426282A2C2E21, 0x232527292B2D2F20,
                           0xA3A5A7A9ABADAFA0, 0xA2A4A6A8AAACAEA1 };
  constexpr ymm_t ymm3 = { 0x323436383A3C3E31, 0x333537393B3D3F30,
                           0xB3B5B7B9BBBDBFB0, 0xB2B4B6B8BABCBEB1 };
  constexpr ymm_t ymm4 = { 0x424446484A4C4E41, 0x434547494B4D4F40,
                           0xC3C5C7C9CBCDCFC0, 0xC2C4C6C8CACCCEC1 };
  constexpr ymm_t ymm5 = { 0x525456585A5C5E51, 0x535557595B5D5F50,
                           0xD3D5D7D9DBDDDFD0, 0xD2D4D6D8DADCDED1 };
  constexpr ymm_t ymm6 = { 0x626466686A6C6E61, 0x636567696B6D6F60,
                           0xE3E5E7E9EBEDEFE0, 0xE2E4E6E8EAECEEE1 };
  constexpr ymm_t ymm7 = { 0x727476787A7C7E71, 0x737577797B7D7F70,
                           0xF3F5F7F9FBFDFFF0, 0xF2F4F6F8FAFCFEF1 };
#if defined(__x86_64__) || defined(_M_X64)
  constexpr ymm_t ymm8 = { 0x838587898B8D8F80, 0x828486888A8C8E81,
                           0x020406080A0C0E01, 0x030507090B0D0F00 };
  constexpr ymm_t ymm9 = { 0x939597999B9D9F90, 0x929496989A9C9E91,
                           0x121416181A1C1E11, 0x131517191B1D1F10 };
  constexpr ymm_t ymm10 = { 0xA3A5A7A9ABADAFA0, 0xA2A4A6A8AAACAEA1,
                            0x222426282A2C2E21, 0x232527292B2D2F20 };
  constexpr ymm_t ymm11 = { 0xB3B5B7B9BBBDBFB0, 0xB2B4B6B8BABCBEB1,
                            0x323436383A3C3E31, 0x333537393B3D3F30 };
  constexpr ymm_t ymm12 = { 0xC3C5C7C9CBCDCFC0, 0xC2C4C6C8CACCCEC1,
                            0x424446484A4C4E41, 0x434547494B4D4F40 };
  constexpr ymm_t ymm13 = { 0xD3D5D7D9DBDDDFD0, 0xD2D4D6D8DADCDED1,
                            0x525456585A5C5E51, 0x535557595B5D5F50 };
  constexpr ymm_t ymm14 = { 0xE3E5E7E9EBEDEFE0, 0xE2E4E6E8EAECEEE1,
                            0x626466686A6C6E61, 0x636567696B6D6F60 };
  constexpr ymm_t ymm15 = { 0xF3F5F7F9FBFDFFF0, 0xF2F4F6F8FAFCFEF1,
                            0x727476787A7C7E71, 0x737577797B7D7F70 };
#endif

  asm volatile(
    "vmovaps  %0, %%ymm0\n\t"
    "vmovaps  %1, %%ymm1\n\t"
    "vmovaps  %2, %%ymm2\n\t"
    "vmovaps  %3, %%ymm3\n\t"
    "vmovaps  %4, %%ymm4\n\t"
    "vmovaps  %5, %%ymm5\n\t"
    "vmovaps  %6, %%ymm6\n\t"
    "vmovaps  %7, %%ymm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  %8, %%ymm8\n\t"
    "vmovaps  %9, %%ymm9\n\t"
    "vmovaps  %10, %%ymm10\n\t"
    "vmovaps  %11, %%ymm11\n\t"
    "vmovaps  %12, %%ymm12\n\t"
    "vmovaps  %13, %%ymm13\n\t"
    "vmovaps  %14, %%ymm14\n\t"
    "vmovaps  %15, %%ymm15\n\t"
#endif
    "\n\t"
    "int3"
    :
    : "m"(ymm0), "m"(ymm1), "m"(ymm2), "m"(ymm3), "m"(ymm4), "m"(ymm5),
      "m"(ymm6), "m"(ymm7)
#if defined(__x86_64__) || defined(_M_X64)
      ,
      "m"(ymm8), "m"(ymm9), "m"(ymm10), "m"(ymm11),
      "m"(ymm12), "m"(ymm13), "m"(ymm14), "m"(ymm15)
#endif
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7"
#if defined(__x86_64__) || defined(_M_X64)
      ,
      "%ymm8", "%ymm9", "%ymm10", "%ymm11", "%ymm12", "%ymm13", "%ymm14",
      "%ymm15"
#endif
  );

  return 0;
}
