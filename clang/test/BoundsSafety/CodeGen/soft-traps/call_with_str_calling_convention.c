// x86_64 and arm64 use preserve_all

// RUN: %clang_cc1 -O0 -fbounds-safety -triple arm64-apple-macos \
// RUN:   -emit-llvm %s -o - -fbounds-safety-soft-traps=call-with-str | \
// RUN:   FileCheck --check-prefixes=PRESERVE_ALL_CC,COMMON %s

// RUN: %clang_cc1 -O0 -fbounds-safety -triple x86_64-apple-macos \
// RUN:   -emit-llvm %s -o - -fbounds-safety-soft-traps=call-with-str | \
// RUN:   FileCheck --check-prefixes=PRESERVE_ALL_CC,COMMON %s

// Other targets use the normal calling convention

// RUN: %clang_cc1 -O0 -fbounds-safety -triple i686-apple-macos \
// RUN:   -emit-llvm %s -o - -fbounds-safety-soft-traps=call-with-str | \
// RUN:   FileCheck --check-prefixes=NORMAL_CC,COMMON %s

// Hidden flag can also switch off custom calling convention

// RUN: %clang_cc1 -O0 -fbounds-safety -triple arm64-apple-macos \
// RUN:   -emit-llvm %s -o - -fbounds-safety-soft-traps=call-with-str \
// RUN:   -mllvm -bounds-safety-soft-trap-preserve-all-cc=false | \
// RUN:   FileCheck --check-prefixes=NORMAL_CC,COMMON %s

#include <ptrcheck.h>

// PRESERVE_ALL_CC: call preserve_allcc void @__bounds_safety_soft_trap_s(ptr @{{.+}})
// PRESERVE_ALL_CC: call preserve_allcc void @__bounds_safety_soft_trap_s(ptr @{{.+}})
// PRESERVE_ALL_CC: call preserve_allcc void @__bounds_safety_soft_trap_s(ptr @{{.+}})
// NORMAL_CC: call void @__bounds_safety_soft_trap_s(ptr @{{.+}})
// NORMAL_CC: call void @__bounds_safety_soft_trap_s(ptr @{{.+}})
// NORMAL_CC: call void @__bounds_safety_soft_trap_s(ptr @{{.+}})
// COMMON-NOT: call {{.*}} void @__bounds_safety_soft_trap_s(ptr @{{.+}})
int read(int* __bidi_indexable ptr, int idx) {
    return ptr[idx];
}

// PRESERVE_ALL_CC: declare preserve_allcc void @__bounds_safety_soft_trap_s(ptr)
// NORMAL_CC: declare void @__bounds_safety_soft_trap_s(ptr) 
