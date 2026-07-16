// clang-format off
// GENERATED - DO NOT EDIT.
//
// Regenerate with amd/comgr/utils/gen-hotswap-fast-stub-inc.sh from
// comgr-hotswap-entry-trampoline-fast-stub.s. Ground-truth encodings from:
//   llvm-mc -triple amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=obj <src> -o <obj>
//   llvm-objcopy -O binary --only-section=.text <obj> <bin>
// The two s_add immediates are assembled with a literal to force the
// 32-bit-literal encoding, then zeroed (the runtime writes the PC-relative
// delta there). The scratch pair is spelled s[100:101]; the runtime
// rewrites the six SGPR register-field bytes per kernel.
//
// HotswapMCTest.StubTemplateMatchesMCOutput proves these bytes still equal
// fresh MC output, so any assembler drift fails the build's tests.

static constexpr uint8_t StubTemplate[] = {
    0x7c, 0x00, 0x0b, 0xee, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x7e, 0x00, 0x47, 0xe4, 0xbe, 0x64, 0xff, 0x64, 0x80,
    0x00, 0x00, 0x00, 0x00, 0x65, 0xff, 0x65, 0x82, 0x00, 0x00, 0x00, 0x00,
    0x64, 0x48, 0x80, 0xbe,
};

static constexpr uint8_t SCodeEnd[] = {
    0x00, 0x00, 0x9f, 0xbf,
};

static constexpr uint8_t SNop[] = {
    0x00, 0x00, 0x80, 0xbf,
};
// clang-format on
