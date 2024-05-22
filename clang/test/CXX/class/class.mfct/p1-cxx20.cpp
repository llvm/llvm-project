// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 no-modules.cpp -fsyntax-only -ast-dump | \
// RUN: FileCheck --match-full-lines --check-prefix=CHECK-NM %s
// RUN: %clang_cc1 -std=c++20 -xc++-user-header header-unit.h -ast-dump | \
// RUN: FileCheck --match-full-lines --check-prefix=CHECK-HU %s
// RUN: %clang_cc1 -std=c++20 module.cpp -ast-dump | \
// RUN: FileCheck --match-full-lines --check-prefix=CHECK-MOD %s

//--- no-modules.cpp

class X {
  void x(){};
};

// CHECK-NM: `-CXXRecordDecl {{.*}} <no-modules.cpp:2:1, line:4:1> line:2:7 class X definition
// CHECK-NM:   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class X
// CHECK-NM-NEXT: `-CXXMethodDecl {{.*}} <line:3:3, col:12> col:8 x 'void ()' implicit-inline

// A header unit header
//--- header-unit.h

class Y {
  void y(){};
};

// CHECK-HU: `-CXXRecordDecl {{.*}} <.{{/|\\\\?}}header-unit.h:2:1, line:4:1> line:2:7 class Y definition
// CHECK-HU: |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class Y
// CHECK-HU-NEXT: `-CXXMethodDecl {{.*}} <line:3:3, col:12> col:8 y 'void ()' implicit-inline

// A textually-included header
//--- header.h

class A {
  void a(){};
};

//--- module.cpp
module;
#include "header.h"

export module M;

class Z {
  void z(){};
};

// CHECK-MOD: |-CXXRecordDecl {{.*}} <.{{/|\\\\?}}header.h:2:1, line:4:1> line:2:7 in M.<global> hidden class A definition
// CHECK-MOD: | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M.<global> hidden implicit class A
// CHECK-MOD-NEXT: | `-CXXMethodDecl {{.*}} <line:3:3, col:12> col:8 in M.<global> hidden a 'void ()' implicit-inline

// CHECK-MOD: `-CXXRecordDecl {{.*}} <module.cpp:6:1, line:8:1> line:6:7 in M hidden class Z{{( ReachableWhenImported)?}} definition
// CHECK-MOD: |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M hidden implicit class Z{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: `-CXXMethodDecl {{.*}} <line:7:3, col:12> col:8 in M hidden z 'void ()'{{( ReachableWhenImported)?}}
