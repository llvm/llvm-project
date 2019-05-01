#include <cstdint>

struct alignas(64) zmm_t {
  uint64_t a, b, c, d, e, f, g, h;
};

int main() {
  zmm_t zmm0 = { 0x020406080A0C0E01, 0x030507090B0D0F00,
                 0x838587898B8D8F80, 0x828486888A8C8E81,
                 0x424446484A4C4E41, 0x434547494B4D4F40,
                 0xC3C5C7C9CBCDCFC0, 0xC2C4C6C8CACCCEC1 };
  zmm_t zmm1 = { 0x121416181A1C1E11, 0x131517191B1D1F10,
                 0x939597999B9D9F90, 0x929496989A9C9E91,
                 0x525456585A5C5E51, 0x535557595B5D5F50,
                 0xD3D5D7D9DBDDDFD0, 0xD2D4D6D8DADCDED1 };
  zmm_t zmm2 = { 0x222426282A2C2E21, 0x232527292B2D2F20,
                 0xA3A5A7A9ABADAFA0, 0xA2A4A6A8AAACAEA1,
                 0x626466686A6C6E61, 0x636567696B6D6F60,
                 0xE3E5E7E9EBEDEFE0, 0xE2E4E6E8EAECEEE1 };
  zmm_t zmm3 = { 0x323436383A3C3E31, 0x333537393B3D3F30,
                 0xB3B5B7B9BBBDBFB0, 0xB2B4B6B8BABCBEB1,
                 0x727476787A7C7E71, 0x737577797B7D7F70,
                 0xF3F5F7F9FBFDFFF0, 0xF2F4F6F8FAFCFEF1 };
  zmm_t zmm4 = { 0x424446484A4C4E41, 0x434547494B4D4F40,
                 0xC3C5C7C9CBCDCFC0, 0xC2C4C6C8CACCCEC1,
                 0x828486888A8C8E81, 0x838587898B8D8F80,
                 0x030507090B0D0F00, 0x020406080A0C0E01 };
  zmm_t zmm5 = { 0x525456585A5C5E51, 0x535557595B5D5F50,
                 0xD3D5D7D9DBDDDFD0, 0xD2D4D6D8DADCDED1,
                 0x929496989A9C9E91, 0x939597999B9D9F90,
                 0x131517191B1D1F10, 0x121416181A1C1E11 };
  zmm_t zmm6 = { 0x626466686A6C6E61, 0x636567696B6D6F60,
                 0xE3E5E7E9EBEDEFE0, 0xE2E4E6E8EAECEEE1,
                 0xA2A4A6A8AAACAEA1, 0xA3A5A7A9ABADAFA0,
                 0x232527292B2D2F20, 0x222426282A2C2E21 };
  zmm_t zmm7 = { 0x727476787A7C7E71, 0x737577797B7D7F70,
                 0xF3F5F7F9FBFDFFF0, 0xF2F4F6F8FAFCFEF1,
                 0xB2B4B6B8BABCBEB1, 0xB3B5B7B9BBBDBFB0,
                 0x333537393B3D3F30, 0x323436383A3C3E31 };
#if defined(__x86_64__) || defined(_M_X64)
  zmm_t zmm8 = { 0x828486888A8C8E81, 0x838587898B8D8F80,
                 0x030507090B0D0F00, 0x020406080A0C0E01,
                 0xC2C4C6C8CACCCEC1, 0xC3C5C7C9CBCDCFC0,
                 0x434547494B4D4F40, 0x424446484A4C4E41 };
  zmm_t zmm9 = { 0x929496989A9C9E91, 0x939597999B9D9F90,
                 0x131517191B1D1F10, 0x121416181A1C1E11,
                 0xD2D4D6D8DADCDED1, 0xD3D5D7D9DBDDDFD0,
                 0x535557595B5D5F50, 0x525456585A5C5E51 };
  zmm_t zmm10 = { 0xA2A4A6A8AAACAEA1, 0xA3A5A7A9ABADAFA0,
                  0x232527292B2D2F20, 0x222426282A2C2E21,
                  0xE2E4E6E8EAECEEE1, 0xE3E5E7E9EBEDEFE0,
                  0x636567696B6D6F60, 0x626466686A6C6E61 };
  zmm_t zmm11 = { 0xB2B4B6B8BABCBEB1, 0xB3B5B7B9BBBDBFB0,
                  0x333537393B3D3F30, 0x323436383A3C3E31,
                  0xF2F4F6F8FAFCFEF1, 0xF3F5F7F9FBFDFFF0,
                  0x737577797B7D7F70, 0x727476787A7C7E71 };
  zmm_t zmm12 = { 0xC2C4C6C8CACCCEC1, 0xC3C5C7C9CBCDCFC0,
                  0x434547494B4D4F40, 0x424446484A4C4E41,
                  0x020406080A0C0E01, 0x030507090B0D0F00,
                  0x838587898B8D8F80, 0x828486888A8C8E81 };
  zmm_t zmm13 = { 0xD2D4D6D8DADCDED1, 0xD3D5D7D9DBDDDFD0,
                  0x535557595B5D5F50, 0x525456585A5C5E51,
                  0x121416181A1C1E11, 0x131517191B1D1F10,
                  0x939597999B9D9F90, 0x929496989A9C9E91 };
  zmm_t zmm14 = { 0xE2E4E6E8EAECEEE1, 0xE3E5E7E9EBEDEFE0,
                  0x636567696B6D6F60, 0x626466686A6C6E61,
                  0x222426282A2C2E21, 0x232527292B2D2F20,
                  0xA3A5A7A9ABADAFA0, 0xA2A4A6A8AAACAEA1 };
  zmm_t zmm15 = { 0xF2F4F6F8FAFCFEF1, 0xF3F5F7F9FBFDFFF0,
                  0x737577797B7D7F70, 0x727476787A7C7E71,
                  0x323436383A3C3E31, 0x333537393B3D3F30,
                  0xB3B5B7B9BBBDBFB0, 0xB2B4B6B8BABCBEB1 };
  zmm_t zmm16 = { 0x030507090B0D0F00, 0x020406080A0C0E01,
                  0x828486888A8C8E81, 0x838587898B8D8F80,
                  0x434547494B4D4F40, 0x424446484A4C4E41,
                  0xC2C4C6C8CACCCEC1, 0xC3C5C7C9CBCDCFC0 };
  zmm_t zmm17 = { 0x131517191B1D1F10, 0x121416181A1C1E11,
                  0x929496989A9C9E91, 0x939597999B9D9F90,
                  0x535557595B5D5F50, 0x525456585A5C5E51,
                  0xD2D4D6D8DADCDED1, 0xD3D5D7D9DBDDDFD0 };
  zmm_t zmm18 = { 0x232527292B2D2F20, 0x222426282A2C2E21,
                  0xA2A4A6A8AAACAEA1, 0xA3A5A7A9ABADAFA0,
                  0x636567696B6D6F60, 0x626466686A6C6E61,
                  0xE2E4E6E8EAECEEE1, 0xE3E5E7E9EBEDEFE0 };
  zmm_t zmm19 = { 0x333537393B3D3F30, 0x323436383A3C3E31,
                  0xB2B4B6B8BABCBEB1, 0xB3B5B7B9BBBDBFB0,
                  0x737577797B7D7F70, 0x727476787A7C7E71,
                  0xF2F4F6F8FAFCFEF1, 0xF3F5F7F9FBFDFFF0 };
  zmm_t zmm20 = { 0x434547494B4D4F40, 0x424446484A4C4E41,
                  0xC2C4C6C8CACCCEC1, 0xC3C5C7C9CBCDCFC0,
                  0x838587898B8D8F80, 0x828486888A8C8E81,
                  0x020406080A0C0E01, 0x030507090B0D0F00 };
  zmm_t zmm21 = { 0x535557595B5D5F50, 0x525456585A5C5E51,
                  0xD2D4D6D8DADCDED1, 0xD3D5D7D9DBDDDFD0,
                  0x939597999B9D9F90, 0x929496989A9C9E91,
                  0x121416181A1C1E11, 0x131517191B1D1F10 };
  zmm_t zmm22 = { 0x636567696B6D6F60, 0x626466686A6C6E61,
                  0xE2E4E6E8EAECEEE1, 0xE3E5E7E9EBEDEFE0,
                  0xA3A5A7A9ABADAFA0, 0xA2A4A6A8AAACAEA1,
                  0x222426282A2C2E21, 0x232527292B2D2F20 };
  zmm_t zmm23 = { 0x737577797B7D7F70, 0x727476787A7C7E71,
                  0xF2F4F6F8FAFCFEF1, 0xF3F5F7F9FBFDFFF0,
                  0xB3B5B7B9BBBDBFB0, 0xB2B4B6B8BABCBEB1,
                  0x323436383A3C3E31, 0x333537393B3D3F30 };
  zmm_t zmm24 = { 0x838587898B8D8F80, 0x828486888A8C8E81,
                  0x020406080A0C0E01, 0x030507090B0D0F00,
                  0xC3C5C7C9CBCDCFC0, 0xC2C4C6C8CACCCEC1,
                  0x424446484A4C4E41, 0x434547494B4D4F40 };
  zmm_t zmm25 = { 0x939597999B9D9F90, 0x929496989A9C9E91,
                  0x121416181A1C1E11, 0x131517191B1D1F10,
                  0xD3D5D7D9DBDDDFD0, 0xD2D4D6D8DADCDED1,
                  0x525456585A5C5E51, 0x535557595B5D5F50 };
  zmm_t zmm26 = { 0xA3A5A7A9ABADAFA0, 0xA2A4A6A8AAACAEA1,
                  0x222426282A2C2E21, 0x232527292B2D2F20,
                  0xE3E5E7E9EBEDEFE0, 0xE2E4E6E8EAECEEE1,
                  0x626466686A6C6E61, 0x636567696B6D6F60 };
  zmm_t zmm27 = { 0xB3B5B7B9BBBDBFB0, 0xB2B4B6B8BABCBEB1,
                  0x323436383A3C3E31, 0x333537393B3D3F30,
                  0xF3F5F7F9FBFDFFF0, 0xF2F4F6F8FAFCFEF1,
                  0x727476787A7C7E71, 0x737577797B7D7F70 };
  zmm_t zmm28 = { 0xC3C5C7C9CBCDCFC0, 0xC2C4C6C8CACCCEC1,
                  0x424446484A4C4E41, 0x434547494B4D4F40,
                  0x030507090B0D0F00, 0x020406080A0C0E01,
                  0x828486888A8C8E81, 0x838587898B8D8F80 };
  zmm_t zmm29 = { 0xD3D5D7D9DBDDDFD0, 0xD2D4D6D8DADCDED1,
                  0x525456585A5C5E51, 0x535557595B5D5F50,
                  0x131517191B1D1F10, 0x121416181A1C1E11,
                  0x929496989A9C9E91, 0x939597999B9D9F90 };
  zmm_t zmm30 = { 0xE3E5E7E9EBEDEFE0, 0xE2E4E6E8EAECEEE1,
                  0x626466686A6C6E61, 0x636567696B6D6F60,
                  0x232527292B2D2F20, 0x222426282A2C2E21,
                  0xA2A4A6A8AAACAEA1, 0xA3A5A7A9ABADAFA0 };
  zmm_t zmm31 = { 0xF3F5F7F9FBFDFFF0, 0xF2F4F6F8FAFCFEF1,
                  0x727476787A7C7E71, 0x737577797B7D7F70,
                  0x333537393B3D3F30, 0x323436383A3C3E31,
                  0xB2B4B6B8BABCBEB1, 0xB3B5B7B9BBBDBFB0 };
#endif

  asm volatile(
    "vmovaps  %0, %%zmm0\n\t"
    "vmovaps  %1, %%zmm1\n\t"
    "vmovaps  %2, %%zmm2\n\t"
    "vmovaps  %3, %%zmm3\n\t"
    "vmovaps  %4, %%zmm4\n\t"
    "vmovaps  %5, %%zmm5\n\t"
    "vmovaps  %6, %%zmm6\n\t"
    "vmovaps  %7, %%zmm7\n\t"
#if defined(__x86_64__) || defined(_M_X64)
    "vmovaps  %8, %%zmm8\n\t"
    "vmovaps  %9, %%zmm9\n\t"
    "vmovaps  %10, %%zmm10\n\t"
    "vmovaps  %11, %%zmm11\n\t"
    "vmovaps  %12, %%zmm12\n\t"
    "vmovaps  %13, %%zmm13\n\t"
    "vmovaps  %14, %%zmm14\n\t"
    "vmovaps  %15, %%zmm15\n\t"
    "vmovaps  %16, %%zmm16\n\t"
    "vmovaps  %17, %%zmm17\n\t"
    "vmovaps  %18, %%zmm18\n\t"
    "vmovaps  %19, %%zmm19\n\t"
    "vmovaps  %20, %%zmm20\n\t"
    "vmovaps  %21, %%zmm21\n\t"
    "vmovaps  %22, %%zmm22\n\t"
    "vmovaps  %23, %%zmm23\n\t"
    "vmovaps  %24, %%zmm24\n\t"
    "vmovaps  %25, %%zmm25\n\t"
    "vmovaps  %26, %%zmm26\n\t"
    "vmovaps  %27, %%zmm27\n\t"
    "vmovaps  %28, %%zmm28\n\t"
    "vmovaps  %29, %%zmm29\n\t"
    "vmovaps  %30, %%zmm30\n\t"
    "vmovaps  %31, %%zmm31\n\t"
#endif
    "\n\t"
    "int3"
    :
    : "m"(zmm0), "m"(zmm1), "m"(zmm2), "m"(zmm3), "m"(zmm4), "m"(zmm5),
      "m"(zmm6), "m"(zmm7)
#if defined(__x86_64__) || defined(_M_X64)
      , "m"(zmm8), "m"(zmm9), "m"(zmm10), "m"(zmm11),
      "m"(zmm12), "m"(zmm13), "m"(zmm14), "m"(zmm15), "m"(zmm16), "m"(zmm17),
      "m"(zmm18), "m"(zmm19), "m"(zmm20), "m"(zmm21), "m"(zmm22), "m"(zmm23),
      "m"(zmm24), "m"(zmm25), "m"(zmm26), "m"(zmm27), "m"(zmm28), "m"(zmm29),
      "m"(zmm30), "m"(zmm31)
#endif
    : "%zmm0", "%zmm1", "%zmm2", "%zmm3", "%zmm4", "%zmm5", "%zmm6", "%zmm7"
#if defined(__x86_64__) || defined(_M_X64)
      , "%zmm8", "%zmm9", "%zmm10", "%zmm11", "%zmm12", "%zmm13", "%zmm14",
      "%zmm15", "%zmm16", "%zmm17", "%zmm18", "%zmm19", "%zmm20", "%zmm21",
      "%zmm22", "%zmm23", "%zmm24", "%zmm25", "%zmm26", "%zmm27", "%zmm28",
      "%zmm29", "%zmm30", "%zmm31"
#endif
  );

  return 0;
}
