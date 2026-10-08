// Tests the remark for when the Standard library modules manifest is
// not found.

// RUN: split-file %s %t

//--- main.cpp
// empty

// RUN: %clang -std=c++23 -nostdlib \
// RUN:   -fmodules-driver -Rmodules-driver \
// RUN:   %t/main.cpp \
// RUN:   -fmodules-cache-path=%t/modules-cache \
// RUN:   %t/main.cpp -### 2>&1 \
// RUN:   | sed 's:\\\\\?:/:g' \
// RUN:   | FileCheck -DPREFIX=%/t %s

// CHECK: clang: remark: standard modules manifest file not found; import of standard library modules not supported [-Rmodules-driver]
