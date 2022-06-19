// Test generation and import of user and system C++20 Header Units.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t

// check user path
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -I user \
// RUN: -xc++-user-header hu-01.h -o hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info hu-01.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface imp-hu-01.cpp \
// RUN:  -I user -fmodule-file=hu-01.pcm -o B.pcm -Rmodule-import \
// RUN: 2>&1  | FileCheck --check-prefix=CHECK-IMP %s -DTDIR=%t

// check system path
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -isystem system \
// RUN: -xc++-system-header hu-02.h -o hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info hu-02.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU2 %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface imp-hu-02.cpp \
// RUN:  -isystem system -fmodule-file=hu-02.pcm -o C.pcm \
// RUN: -Rmodule-import 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-SYS-IMP %s -DTDIR=%t

// check absolute path
// RUN: %clang_cc1 -std=c++20 -emit-header-unit  \
// RUN: -xc++-header-unit-header %t/hu-03.h -o hu-03.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info hu-03.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU3 %s -DTDIR=%t

//--- user/hu-01.h
int foo(int);

// CHECK-HU:  ====== C++20 Module structure ======
// CHECK-HU-NEXT:  Header Unit 'user{{[/\\]}}hu-01.h' is the Primary Module at index #1

//--- imp-hu-01.cpp
export module B;
import "hu-01.h";

int bar(int x) {
  return foo(x);
}
// CHECK-IMP: remark: importing module 'user{{[/\\]}}hu-01.h' from 'hu-01.pcm'
// expected-no-diagnostics

//--- system/hu-02.h
int baz(int);

// CHECK-HU2:  ====== C++20 Module structure ======
// CHECK-HU2-NEXT:  Header Unit 'system{{[/\\]}}hu-02.h' is the Primary Module at index #1

//--- imp-hu-02.cpp
module;
import <hu-02.h>;

export module C;

int bar(int x) {
  return baz(x);
}
// CHECK-SYS-IMP: remark: importing module 'system{{[/\\]}}hu-02.h' from 'hu-02.pcm'
// expected-no-diagnostics

//--- hu-03.h
int curly(int);

// CHECK-HU3:  ====== C++20 Module structure ======
// CHECK-HU3-NEXT:  Header Unit '[[TDIR]]/hu-03.h' is the Primary Module at index #1
// expected-no-diagnostics
