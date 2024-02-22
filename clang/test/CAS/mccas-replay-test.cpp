// RUN: rm -rf %t && mkdir -p %t
// RUN: export LLVM_CACHE_CAS_PATH=%t/cas && %clang-cache \
// RUN:   %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Rcompile-job-cache %s -o %t/tmp.o -g 2>&1 | FileCheck %s -check-prefix=CACHE-MISS
// CACHE-MISS: remark: compile job cache miss

// RUN: llvm-objdump -h %t/tmp.o | FileCheck %s -check-prefix=CHECK-OBJDUMP

// RUN: export LLVM_CACHE_CAS_PATH=%t/cas && %clang-cache \
// RUN:   %clang -target arm64-apple-macosx12.0.0 -c -Xclang -fcas-backend -Rcompile-job-cache %s -o %t/tmp.o -g 2>&1 | FileCheck %s -check-prefix=CACHE-HIT
// CACHE-HIT: remark: compile job cache hit

// RUN: llvm-objdump -h %t/tmp.o 2>&1 | FileCheck %s -check-prefix=CHECK-OBJDUMP

// CHECK-OBJDUMP: Sections:
// CHECK-OBJDUMP-NEXT: Idx Name             Size     VMA              Type
// CHECK-OBJDUMP-NEXT:   0 __text           {{[0-9a-f]+}} {{[0-9a-f]+}} TEXT
// CHECK-OBJDUMP-NEXT:   1 __debug_abbrev   {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   2 __debug_info     {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   3 __debug_str      {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   4 __apple_names    {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   5 __apple_objc     {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   6 __apple_namespac {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   7 __apple_types    {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG
// CHECK-OBJDUMP-NEXT:   8 __compact_unwind {{[0-9a-f]+}} {{[0-9a-f]+}} DATA
// CHECK-OBJDUMP-NEXT:   9 __debug_line     {{[0-9a-f]+}} {{[0-9a-f]+}} DATA, DEBUG

// REQUIRES: aarch64-registered-target

int foo() {
    return 1;
}
