// Make sure that ASan works with non-cdecl default calling conventions.
// Many x86 projects pass `/Gz` to their compiles, so that __stdcall is the default,
// but LLVM is built with __cdecl.
//
// RUN: %clang_cl_asan -Gz %Od %s %Fe%t

// includes a declaration of `_ReturnAddress`
#include <intrin.h>

#include <sanitizer/asan_interface.h>

int main() {
  alignas(8) char buffer[8];
  __asan_poison_memory_region(buffer, sizeof buffer);
}
