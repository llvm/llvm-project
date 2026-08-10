// Test generation and import of simple C++20 Header Units.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-header %t/hu-01.h \
// RUN:  -o %t/hu-01.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info %t/hu-01.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/imp-hu-01.cpp \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/B.pcm -Rmodule-import 2>&1  | \
// RUN: FileCheck --check-prefix=CHECK-IMP %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/imp-hu-02.cpp \
// RUN:  -fmodule-file=%t/hu-01.pcm -o %t/C.pcm -Rmodule-import 2>&1  | \
// RUN: FileCheck --check-prefix=CHECK-GMF-IMP %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-header %t/hu-02.h \
// RUN:  -o %t/hu-02.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/imp-hu-03.cpp \
// RUN:  -fmodule-file=%t/hu-01.pcm -fmodule-file=%t/hu-02.pcm -o %t/D.pcm \
// RUN: -Rmodule-import 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-BOTH %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-header %t/hu-03.h \
// RUN: -fmodule-file=%t/hu-01.pcm  -o %t/hu-03.pcm

// RUN: %clang_cc1 -std=c++20 -module-file-info %t/hu-03.pcm | \
// RUN: FileCheck --check-prefix=CHECK-HU-HU %s -DTDIR=%t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/imp-hu-04.cpp \
// RUN:  -fmodule-file=%t/hu-03.pcm -o %t/E.pcm -Rmodule-import 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-NESTED %s -DTDIR=%t

//--- hu-01.h
int foo(int);

// CHECK-HU:  ====== C++20 Module structure ======
// CHECK-HU-NEXT:  Header Unit '[[TDIR]]/hu-01.h' is the Primary Module at index #1

//--- imp-hu-01.cpp
export module B;
import "hu-01.h";

int bar(int x) {
  return foo(x);
}
// CHECK-IMP: remark: importing module '[[TDIR]]/hu-01.h' from '[[TDIR]]/hu-01.pcm'
// expected-no-diagnostics

//--- imp-hu-02.cpp
module;
import "hu-01.h";

export module C;

int bar(int x) {
  return foo(x);
}
// CHECK-GMF-IMP: remark: importing module '[[TDIR]]/hu-01.h' from '[[TDIR]]/hu-01.pcm'
// expected-no-diagnostics

//--- hu-02.h
int baz(int);

//--- imp-hu-03.cpp
module;
export import "hu-01.h";

export module D;
import "hu-02.h";

int bar(int x) {
  return foo(x) + baz(x);
}
// CHECK-BOTH: remark: importing module '[[TDIR]]/hu-01.h' from '[[TDIR]]/hu-01.pcm'
// CHECK-BOTH: remark: importing module '[[TDIR]]/hu-02.h' from '[[TDIR]]/hu-02.pcm'
// expected-no-diagnostics

//--- hu-03.h
export import "hu-01.h";
int baz(int);
// CHECK-HU-HU:  ====== C++20 Module structure ======
// CHECK-HU-HU-NEXT:  Header Unit '[[TDIR]]/hu-03.h' is the Primary Module at index #2
// CHECK-HU-HU-NEXT:   Exports:
// CHECK-HU-HU-NEXT:    Header Unit '[[TDIR]]/hu-01.h' is at index #1

// expected-no-diagnostics

//--- imp-hu-04.cpp
module;
import "hu-03.h";

export module E;

int bar(int x) {
  return foo(x) + baz(x);
}
// CHECK-NESTED: remark: importing module '[[TDIR]]/hu-03.h' from '[[TDIR]]/hu-03.pcm'
// expected-no-diagnostics
