// Test that HWASan shadow is initialized before .preinit_array functions run.

// RUN: %clang_hwasan %s -o %t
// RUN: %run %t

volatile int Global;
void StoreToGlobal() { Global = 42; }

__attribute__((section(".preinit_array"), used))
void (*__StoreToGlobal_preinit)() = StoreToGlobal;

int main() { return Global != 42; }
