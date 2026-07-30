// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 no-modules.cpp -ast-dump | \
// RUN: FileCheck --match-full-lines --check-prefix=CHECK-NM %s
// RUN: %clang_cc1 -std=c++20 -xc++-user-header header-unit.h -ast-dump | \
// RUN: FileCheck --match-full-lines --check-prefix=CHECK-HU %s
// RUN: %clang_cc1 -std=c++20 module.cpp -ast-dump | \
// RUN: FileCheck --match-full-lines --check-prefix=CHECK-MOD %s

//--- no-modules.cpp

class X {
  friend void x(){};
};

// CHECK-NM: `-CXXRecordDecl {{.*}} <no-modules.cpp:2:1, line:4:1> line:2:7 class X definition external-linkage
// CHECK-NM:   |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class X
// CHECK-NM-NEXT: `-FriendDecl {{.*}} <line:3:3, col:19> col:15
// CHECK-NM-NEXT: `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:19> col:15 friend_undeclared x 'void ()' implicit-inline external-linkage

//--- header-unit.h

class Y {
  friend void y(){};
};

// CHECK-HU: `-CXXRecordDecl {{.*}} <.{{/|\\\\?}}header-unit.h:2:1, line:4:1> line:2:7 class Y definition external-linkage
// CHECK-HU: |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 implicit class Y
// CHECK-HU-NEXT: `-FriendDecl {{.*}} <line:3:3, col:19> col:15
// CHECK-HU-NEXT: `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:19> col:15 friend_undeclared y 'void ()' implicit-inline external-linkage

// A textually-included header
//--- header.h

class A {
  friend void a(){};
};

//--- module.cpp
module;
#include "header.h"

export module M;

class Z {
  friend void z1(){};
};

class Inline {
  friend inline void z2(){};
};

class Constexpr {
  friend constexpr void z3(){};
};

class Consteval {
  friend consteval void z4(){};
};

extern "C++" class GlobalModule {
  friend void z5(){};
};

// CHECK-MOD: |-CXXRecordDecl {{.*}} <.{{/|\\\\?}}header.h:2:1, line:4:1> line:2:7 in M.<global> hidden class A definition external-linkage
// CHECK-MOD: | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M.<global> hidden implicit class A
// CHECK-MOD-NEXT: | `-FriendDecl {{.*}} <line:3:3, col:19> col:15 in M.<global>
// CHECK-MOD-NEXT: |   `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:19> col:15 in M.<global> hidden friend_undeclared a 'void ()' implicit-inline external-linkage

// CHECK-MOD: |-CXXRecordDecl {{.*}} <module.cpp:6:1, line:8:1> line:6:7 in M hidden class Z{{( ReachableWhenImported)?}} definition module-linkage
// CHECK-MOD: | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M hidden implicit class Z{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: | `-FriendDecl {{.*}} <line:7:3, col:20> col:15 in M{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: |   `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:20> col:15 in M hidden friend_undeclared z1 'void ()'{{( ReachableWhenImported)?}} module-linkage

// CHECK-MOD: |-CXXRecordDecl {{.*}} <line:10:1, line:12:1> line:10:7 in M hidden class Inline{{( ReachableWhenImported)?}} definition module-linkage
// CHECK-MOD: | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M hidden implicit class Inline{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: | `-FriendDecl {{.*}} <line:11:3, col:27> col:22 in M{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: |   `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:27> col:22 in M hidden friend_undeclared z2 'void ()'{{( ReachableWhenImported)?}} inline module-linkage

// CHECK-MOD: |-CXXRecordDecl {{.*}} <line:14:1, line:16:1> line:14:7 in M hidden class Constexpr{{( ReachableWhenImported)?}} definition module-linkage
// CHECK-MOD: | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M hidden implicit class Constexpr{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: | `-FriendDecl {{.*}} <line:15:3, col:30> col:25 in M{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: |   `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:30> col:25 in M hidden constexpr friend_undeclared z3 'void ()'{{( ReachableWhenImported)?}} implicit-inline module-linkage

// CHECK-MOD: |-CXXRecordDecl {{.*}} <line:18:1, line:20:1> line:18:7 in M hidden class Consteval{{( ReachableWhenImported)?}} definition module-linkage
// CHECK-MOD: | |-CXXRecordDecl {{.*}} <col:1, col:7> col:7 in M hidden implicit class Consteval{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: | `-FriendDecl {{.*}} <line:19:3, col:30> col:25 in M{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: |   `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:30> col:25 in M hidden consteval friend_undeclared z4 'void ()'{{( ReachableWhenImported)?}} implicit-inline module-linkage

// CHECK-MOD: `-CXXRecordDecl {{.*}} <col:14, line:24:1> line:22:20 in M.<implicit global> hidden class GlobalModule{{( ReachableWhenImported)?}} definition external-linkage
// CHECK-MOD: |-CXXRecordDecl {{.*}} <col:14, col:20> col:20 in M.<implicit global> hidden implicit class GlobalModule{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: `-FriendDecl {{.*}} <line:23:3, col:20> col:15 in M.<implicit global>{{( ReachableWhenImported)?}}
// CHECK-MOD-NEXT: `-FunctionDecl {{.*}} parent {{.*}} <col:3, col:20> col:15 in M.<implicit global> hidden friend_undeclared z5 'void ()'{{( ReachableWhenImported)?}} implicit-inline external-linkage
