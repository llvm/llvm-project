// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/b.cppm -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -fmodule-file=b=%t/b.pcm \
// RUN:     -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only \
// RUN:     -Wno-read-modules-implicitly -DNO_DIAG
// RUN: %clang_cc1 -std=c++20 %t/user.cpp -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -DNO_DIAG -verify -fsyntax-only
//
// RUN: %clang_cc1 -std=c++20 %t/a.pcm -S -emit-llvm -o - 2>&1 | FileCheck %t/a.cppm
// RUN: %clang_cc1 -std=c++20 %t/a.pcm -fmodule-file=b=%t/b.pcm -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %t/a.cppm -check-prefix=CHECK-CORRECT
//
// RUN: mkdir -p %t/tmp
// RUN: mv %t/b.pcm %t/tmp/b.pcm
// RUN: not %clang_cc1 -std=c++20 %t/a.pcm -S -emit-llvm -o - 2>&1 \
// RUN:     | FileCheck %t/a.cppm -check-prefix=CHECK-ERROR
// RUN: %clang_cc1 -std=c++20 %t/a.pcm -S -emit-llvm -o - 2>&1 -fmodule-file=b=%t/tmp/b.pcm \
// RUN:     | FileCheck %t/a.cppm -check-prefix=CHECK-CORRECT

//--- b.cppm
export module b;
export int b() {
    return 43;
}

//--- a.cppm
export module a;
import b;
export int a() {
    return b() + 43;
}

// CHECK: it is deprecated to read module 'b' implicitly;

// CHECK-CORRECT-NOT: warning
// CHECK-CORRECT-NOT: error

// CHECK-ERROR: error: module file{{.*}}not found: module file not found


//--- user.cpp
#ifdef NO_DIAG
// expected-no-diagnostics
#else
 // expected-warning@+2 {{it is deprecated to read module 'b' implicitly;}}
#endif
import a;
int use() {
    return a();
}
