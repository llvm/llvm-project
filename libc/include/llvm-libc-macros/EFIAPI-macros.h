#ifndef LLVM_LIBC_MACROS_EFIAPI_MACROS_H
#define LLVM_LIBC_MACROS_EFIAPI_MACROS_H

#if defined(__x86_64__) && !defined(__ILP32__)
#define EFIAPI __attribute__((ms_abi))
#else
#define EFIAPI
#endif

#endif // LLVM_LIBC_MACROS_EFIAPI_MACROS_H
