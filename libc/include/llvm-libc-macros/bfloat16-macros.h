#ifndef LLVM_LIBC_MACROS_BFLOAT16_MACROS_H
#define LLVM_LIBC_MACROS_BFLOAT16_MACROS_H

// Clang supports __bf16 when __is_identifier(__bf16) is 0, except on:
// - x86 targets without SSE2
// - 32-bit ARM targets with soft float or lacking hardware FP registers
#if defined(__clang__)
#if !__is_identifier(__bf16)
#if (!defined(__i386__) || defined(__SSE2__)) &&                               \
    (!defined(__arm__) || ((defined(__ARM_FP) && !defined(__SOFTFP__)) ||      \
                           defined(__ARM_FEATURE_BF16)))
#define LIBC_TYPES_HAS_BUILTIN_BFLOAT16
#endif
#endif
#elif defined(__GNUC__) && (__GNUC__ >= 13)
#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE2__)) ||         \
    defined(__aarch64__) ||                                                    \
    (defined(__arm__) && ((defined(__ARM_FP) && !defined(__SOFTFP__)) ||       \
                          defined(__ARM_FEATURE_BF16)))
#define LIBC_TYPES_HAS_BUILTIN_BFLOAT16
#endif
#endif

#endif // LLVM_LIBC_MACROS_BFLOAT16_MACROS_H
