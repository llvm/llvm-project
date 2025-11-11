// Check that when a runtime implementation uses the `bounds_safety_soft_traps.h`
// header that the right calling convention is used.

// x86_64 and arm64 use preserve_all

// RUN: %clang_cc1 -O0 -triple arm64-apple-macos \
// RUN:   -emit-llvm %s -o -  | \
// RUN:   FileCheck --check-prefixes=PRESERVE_ALL_CC %s

// RUN: %clang_cc1 -O0 -triple x86_64-apple-macos \
// RUN:   -emit-llvm %s -o - | \
// RUN:   FileCheck --check-prefixes=PRESERVE_ALL_CC %s

// Other targets use the normal calling convention

// RUN: %clang_cc1 -O0 -triple i686-apple-macos \
// RUN:   -emit-llvm %s -o - | \
// RUN:   FileCheck --check-prefixes=NORMAL_CC %s
#include <bounds_safety_soft_traps.h>

#if __CLANG_BOUNDS_SAFETY_SOFT_TRAP_API_VERSION > 0
#error ABI changed
#endif

// Note the explicit calling convention macro isn't used here. Clang should
// use the attribute from the declaration in the header.

// PRESERVE_ALL_CC: define preserve_allcc void @__bounds_safety_soft_trap_s(ptr {{.+}})
// NORMAL_CC: define void @__bounds_safety_soft_trap_s(ptr {{.+}})
void __bounds_safety_soft_trap_s(const char *reason) {

}

// PRESERVE_ALL_CC: define preserve_allcc void @__bounds_safety_soft_trap()
// NORMAL_CC: define void @__bounds_safety_soft_trap()
void __bounds_safety_soft_trap(void) {
    
}

