// Test output from -module-file-info about C++20 modules.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/mod-info-tu1.cpp \
// RUN:  -o %t/A.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info %t/A.pcm | FileCheck \
// RUN:  --check-prefix=CHECK-A %s

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/mod-info-tu2.cpp \
// RUN:  -o %t/B.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info %t/B.pcm | FileCheck \
// RUN:  --check-prefix=CHECK-B %s

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/mod-info-tu3.cpp \
// RUN:  -fmodule-file=%t/A.pcm -fmodule-file=%t/B.pcm -o %t/Foo.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info %t/Foo.pcm | FileCheck \
// RUN:  --check-prefix=CHECK-FOO %s

// expected-no-diagnostics

//--- mod-info-tu1.cpp
export module A;

void a();

// CHECK-A: ====== C++20
// CHECK-A-NEXT: Interface Unit 'A' is the Primary Module at index #1

//--- mod-info-tu2.cpp
export module B;

void b();

// CHECK-B: ====== C++20
// CHECK-B-NEXT: Interface Unit 'B' is the Primary Module at index #1

//--- mod-info-tu3.cpp
module;

export module Foo;

import A;
export import B;

namespace hello {
export void say(const char *);
}

void foo() {}

// CHECK-FOO: ====== C++20
// CHECK-FOO-NEXT:  Interface Unit 'Foo' is the Primary Module at index #3
// CHECK-FOO-NEXT:   Sub Modules:
// CHECK-FOO-NEXT:    Global Module Fragment '<global>' is at index #4
// CHECK-FOO-NEXT:   Imports:
// CHECK-FOO-NEXT:    Interface Unit 'A' is at index #1
// CHECK-FOO-NEXT:   Exports:
// CHECK-FOO-NEXT:    Interface Unit 'B' is at index #2
