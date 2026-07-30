// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -triple x86_64-pc-windows-msvc -I%S/Inputs/inherit-attribute -fmodules-cache-path=%t \
// RUN: -fimplicit-module-maps -fmodules-local-submodule-visibility %s -ast-dump-all \
// RUN: | FileCheck %s

#include "b.h"
#include "c.h"

class Foo;

Foo f;

// CHECK:   CXXRecordDecl {{.*}} imported in b {{.*}} Foo
// CHECK:   MSInheritanceAttr {{[^()]*$}}

// CHECK:   CXXRecordDecl {{.*}} prev {{.*}} imported in c {{.*}} Foo
// CHECK:   MSInheritanceAttr {{.*}} Inherited {{[^()]*$}}

// CHECK:   CXXRecordDecl {{.*}} <line:9:1, col:7> col:7 referenced class Foo
// CHECK:   MSInheritanceAttr {{.*}} Inherited {{[^()]*$}}
