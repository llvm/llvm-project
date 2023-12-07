// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t
// RUN: %clangxx_hwasan -mllvm -hwasan-instrument-with-calls=1 -O0 %s -o %t -fsanitize-recover=hwaddress

// Utilizes all flavors of __hwasan_load/store interface functions to verify
// that the instrumentation and the interface provided by HWASan do match.
// In case of a discrepancy, this test fails to link.

#include <sanitizer/hwasan_interface.h>

#define F(T) void f_##T(T *a, T *b) { *a = *b; }

F(uint8_t)
F(uint16_t)
F(uint32_t)
F(uint64_t)

typedef unsigned V32 __attribute__((__vector_size__(32)));
F(V32)

int main() {}
