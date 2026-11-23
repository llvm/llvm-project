// Check that implicit members that a module adds to a class imported from
// another module appear as members of that class once deserialized,
// and appear once even if several modules add the same member.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t/include %t/main.cpp \
// RUN:   -ast-dump-all -ast-dump-filter Copyable \
// RUN:   | FileCheck %s --check-prefix=COPYABLE \
// RUN:       --implicit-check-not=CXXConstructorDecl \
// RUN:       --implicit-check-not=CXXDestructorDecl \
// RUN:       --implicit-check-not=CXXMethodDecl
// RUN: %clang_cc1 -std=c++17 -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t/include %t/main.cpp \
// RUN:   -ast-dump-all -ast-dump-filter Derived \
// RUN:   | FileCheck %s --check-prefix=DERIVED \
// RUN:       --implicit-check-not=CXXConstructorDecl

//--- include/module.modulemap
module A { header "a.h" export * }
module B { header "b.h" export * }
module C { header "c.h" export * }

//--- include/a.h
struct NonTrivial { ~NonTrivial(); };

// All special members are implicit; none are declared in A.
struct Copyable { NonTrivial m; };

struct Base { Base(int); };
struct Derived : Base { using Base::Base; };

//--- include/b.h
#include "a.h"
inline void useB() {
  Copyable a;
  Copyable b(a);
  Copyable c(static_cast<Copyable &&>(a));
  b = a;
  c = static_cast<Copyable &&>(a);
  Derived d(1);
}

//--- include/c.h
#include "a.h"
inline void useC() {
  Copyable a;
  Copyable b(a);
  Copyable c(static_cast<Copyable &&>(a));
  b = a;
  c = static_cast<Copyable &&>(a);
  Derived d(1);
}

//--- main.cpp
#include "b.h"
#include "c.h"

// COPYABLE-LABEL: Dumping Copyable:
// COPYABLE-NEXT:  CXXRecordDecl {{.*}} imported in A {{.*}}struct Copyable definition
// COPYABLE:       CXXConstructorDecl {{.*}} implicit {{.*}}Copyable 'void (const Copyable &){{.*}}'
// COPYABLE:       CXXConstructorDecl {{.*}} implicit {{.*}}Copyable 'void (Copyable &&){{.*}}'
// COPYABLE:       CXXMethodDecl {{.*}} implicit {{.*}}operator= 'Copyable &(Copyable &&){{.*}}'
// COPYABLE:       CXXDestructorDecl {{.*}} implicit {{.*}}~Copyable 'void (){{.*}}'
// COPYABLE:       CXXConstructorDecl {{.*}} implicit {{.*}}Copyable 'void (){{.*}}'
// COPYABLE:       CXXMethodDecl {{.*}} implicit {{.*}}operator= 'Copyable &(const Copyable &){{.*}}'

// DERIVED-LABEL: Dumping Derived:
// DERIVED-NEXT:  CXXRecordDecl {{.*}} imported in A {{.*}}struct Derived definition
// Targets of the ConstructorUsingShadowDecls, declared in A.
// DERIVED:       CXXConstructorDecl {{.*}}Base 'void (int)'
// DERIVED:       CXXConstructorDecl {{.*}} implicit {{.*}}Base 'void (const Base &)'
// DERIVED:       CXXConstructorDecl {{.*}} implicit {{.*}}Base 'void (Base &&)'
// Implicit members added by B.
// DERIVED:       CXXConstructorDecl {{.*}} implicit {{.*}}Derived 'void ()'
// DERIVED:       CXXConstructorDecl {{.*}} implicit {{.*}}Derived 'void (const Derived &)'
// DERIVED:       CXXConstructorDecl {{.*}} implicit {{.*}}Derived 'void (Derived &&)'
// Inheriting constructors are named after the base class.
// DERIVED:       CXXConstructorDecl {{.*}} implicit {{.*}}Base 'void (int){{.*}}'
