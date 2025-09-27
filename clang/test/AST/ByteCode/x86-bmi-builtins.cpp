// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -target-feature +bmi -target-feature +bmi2 -fexperimental-new-constant-interpreter -fsyntax-only %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -target-feature +bmi -target-feature +bmi2 -fsyntax-only %s

// expected-no-diagnostics

static_assert(__builtin_ia32_pdep_si(0x89ABCDEFu, 0x00000000u) == 0x00000000u);
static_assert(__builtin_ia32_pdep_si(0x89ABCDEFu, 0x000000F0u) == 0x000000F0u);
static_assert(__builtin_ia32_pdep_si(0x89ABCDEFu, 0x000000C0u) == 0x000000C0u);
static_assert(__builtin_ia32_pdep_si(0x89ABCDEFu, 0xF00000F0u) == 0xE00000F0u);
static_assert(__builtin_ia32_pdep_si(0x89ABCDEFu, 0xFFFFFFFFu) == 0x89ABCDEFu);

static_assert(__builtin_ia32_pdep_di(0x0123456789ABCDEFULL, 0x0000000000000000ULL) ==
              0x0000000000000000ULL);
static_assert(__builtin_ia32_pdep_di(0x0123456789ABCDEFULL, 0x00000000000000F0ULL) ==
              0x00000000000000F0ULL);
static_assert(__builtin_ia32_pdep_di(0x0123456789ABCDEFULL, 0xF00000F0F00000F0ULL) ==
              0xC00000D0E00000F0ULL);
static_assert(__builtin_ia32_pdep_di(0x0123456789ABCDEFULL, 0xFFFFFFFFFFFFFFFFULL) ==
              0x0123456789ABCDEFULL);

static_assert(__builtin_ia32_pext_si(0x89ABCDEFu, 0x00000000u) == 0x00000000u);
static_assert(__builtin_ia32_pext_si(0x89ABCDEFu, 0x000000F0u) == 0x0000000Eu);
static_assert(__builtin_ia32_pext_si(0x89ABCDEFu, 0x000000C0u) == 0x00000003u);
static_assert(__builtin_ia32_pext_si(0x89ABCDEFu, 0xF00000F0u) == 0x0000008Eu);
static_assert(__builtin_ia32_pext_si(0x89ABCDEFu, 0xFFFFFFFFu) == 0x89ABCDEFu);

static_assert(__builtin_ia32_pext_di(0x0123456789ABCDEFULL, 0x0000000000000000ULL) ==
              0x0000000000000000ULL);
static_assert(__builtin_ia32_pext_di(0x0123456789ABCDEFULL, 0x00000000000000F0ULL) ==
              0x000000000000000EULL);
static_assert(__builtin_ia32_pext_di(0x0123456789ABCDEFULL, 0xF00000F0F00000F0ULL) ==
              0x000000000000068EULL);
static_assert(__builtin_ia32_pext_di(0x0123456789ABCDEFULL, 0xFFFFFFFFFFFFFFFFULL) ==
              0x0123456789ABCDEFULL);

static_assert(__builtin_ia32_bzhi_si(0x89ABCDEFu, 0u) == 0x00000000u);
static_assert(__builtin_ia32_bzhi_si(0x89ABCDEFu, 16u) == 0x0000CDEFu);
static_assert(__builtin_ia32_bzhi_si(0x89ABCDEFu, 31u) == 0x09ABCDEFu);
static_assert(__builtin_ia32_bzhi_si(0x89ABCDEFu, 32u) == 0x89ABCDEFu);
static_assert(__builtin_ia32_bzhi_si(0x89ABCDEFu, 99u) == 0x89ABCDEFu);
static_assert(__builtin_ia32_bzhi_si(0x89ABCDEFu, 260u) == 0x0000000Fu);

static_assert(__builtin_ia32_bzhi_di(0x0123456789ABCDEFULL, 0ULL) == 0x0000000000000000ULL);
static_assert(__builtin_ia32_bzhi_di(0x0123456789ABCDEFULL, 32ULL) == 0x0000000089ABCDEFULL);
static_assert(__builtin_ia32_bzhi_di(0x0123456789ABCDEFULL, 99ULL) == 0x0123456789ABCDEFULL);
static_assert(__builtin_ia32_bzhi_di(0x0123456789ABCDEFULL, 520ULL) == 0x00000000000000EFULL);

static_assert(__builtin_ia32_bextr_u32(0x89ABCDEFu, 0x0000u) == 0x00000000u);
static_assert(__builtin_ia32_bextr_u32(0x89ABCDEFu, 0x0800u) == 0x000000EFu);
static_assert(__builtin_ia32_bextr_u32(0x89ABCDEFu, 0x0804u) == 0x000000DEu);
static_assert(__builtin_ia32_bextr_u32(0x89ABCDEFu, 0x0414u) == 0x0000000Au);
static_assert(__builtin_ia32_bextr_u32(0x89ABCDEFu, 0x081Cu) == 0x00000008u);
static_assert(__builtin_ia32_bextr_u32(0x89ABCDEFu, 0x0428u) == 0x00000000u);

static_assert(__builtin_ia32_bextri_u32(0x89ABCDEFu, 0x0800u) == 0x000000EFu);
static_assert(__builtin_ia32_bextri_u32(0x89ABCDEFu, 0x0804u) == 0x000000DEu);
static_assert(__builtin_ia32_bextri_u32(0x89ABCDEFu, 0x0414u) == 0x0000000Au);

static_assert(__builtin_ia32_bextr_u64(0x0123456789ABCDEFULL, 0x0000ULL) == 0x0000000000000000ULL);
static_assert(__builtin_ia32_bextr_u64(0x0123456789ABCDEFULL, 0x1008ULL) == 0x000000000000ABCDULL);
static_assert(__builtin_ia32_bextr_u64(0x0123456789ABCDEFULL, 0x0804ULL) == 0x00000000000000DEULL);
static_assert(__builtin_ia32_bextr_u64(0x0123456789ABCDEFULL, 0xC800ULL) == 0x0123456789ABCDEFULL);
static_assert(__builtin_ia32_bextr_u64(0x0123456789ABCDEFULL, 0x1028ULL) == 0x0000000000002345ULL);
static_assert(__builtin_ia32_bextr_u64(0x0123456789ABCDEFULL, 0x0850ULL) == 0x0000000000000000ULL);

static_assert(__builtin_ia32_bextri_u64(0x0123456789ABCDEFULL, 0x1008ULL) == 0x000000000000ABCDULL);
static_assert(__builtin_ia32_bextri_u64(0x0123456789ABCDEFULL, 0x0804ULL) == 0x00000000000000DEULL);
