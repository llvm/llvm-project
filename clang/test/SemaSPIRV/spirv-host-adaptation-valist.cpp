/// Tests that getBuiltinVaListKind() delegates to the host target, verified
/// via sizeof(__builtin_va_list) and struct layout containing va_list.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple aarch64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -fsycl-is-device -fsyntax-only -verify %s

// expected-no-diagnostics

struct has_valist { int x; __builtin_va_list ap; int y; };

#if defined(_WIN32)
// Windows x86_64: CharPtrBuiltinVaList (char*)
static_assert(sizeof(__builtin_va_list) == 8, "Windows: va_list is char*");
static_assert(sizeof(has_valist) == 24, "Windows: struct layout with char* va_list");
static_assert(__builtin_offsetof(has_valist, y) == 16,
              "Windows: field after char* va_list");
#elif defined(__aarch64__)
// AArch64 Linux: AArch64ABIBuiltinVaList (struct, 32 bytes)
static_assert(sizeof(__builtin_va_list) == 32, "AArch64: va_list is struct");
static_assert(sizeof(has_valist) == 48, "AArch64: struct layout with 32-byte va_list");
static_assert(__builtin_offsetof(has_valist, y) == 40,
              "AArch64: field after 32-byte va_list");
#elif defined(__x86_64__)
// x86_64 Linux: X86_64ABIBuiltinVaList (struct __va_list_tag, 24 bytes)
static_assert(sizeof(__builtin_va_list) == 24, "Linux x86_64: va_list is struct");
static_assert(sizeof(has_valist) == 40, "Linux x86_64: struct layout with 24-byte va_list");
static_assert(__builtin_offsetof(has_valist, y) == 32,
              "Linux x86_64: field after 24-byte va_list");
#else
// No host: VoidPtrBuiltinVaList (void*)
static_assert(sizeof(__builtin_va_list) == 8, "No host: va_list is void*");
static_assert(sizeof(has_valist) == 24, "No host: struct layout with void* va_list");
static_assert(__builtin_offsetof(has_valist, y) == 16,
              "No host: field after void* va_list");
#endif
