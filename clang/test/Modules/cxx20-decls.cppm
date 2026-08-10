// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -I %t %t/A.cppm -emit-module-interface -o %t/A.pcm -verify
// RUN: %clang_cc1 -std=c++20 -I %t %t/B.cpp -fmodule-file=A=%t/A.pcm -fsyntax-only -verify -ast-dump-all -ast-dump-filter baz | FileCheck %s

//--- foo.h
namespace baz {
  using foo = char;
  using baz::foo;
}

//--- A.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module A;

//--- B.cpp
// expected-no-diagnostics
#include "foo.h"
import A;
// Since modules are loaded lazily, force loading by performing a lookup.
using xxx = baz::foo;

// CHECK-LABEL: Dumping baz:
// CHECK-NEXT: NamespaceDecl 0x[[BAZ_REDECL_ADDR:[^ ]*]] prev 0x[[BAZ_ADDR:[^ ]*]]
// CHECK:      TypeAliasDecl 0x[[ALIAS_REDECL_ADDR:[^ ]*]] prev 0x[[ALIAS_ADDR:[^ ]*]]
// FIXME: UsingShadowDecl should have been merged
// CHECK:      UsingShadowDecl 0x{{[^ ]*}} prev 0x[[SHADOW_ADDR:[^ ]*]] {{.*}} imported in A.<global> {{.*}} 'foo'

// CHECK-LABEL: Dumping baz:
// CHECK-NEXT: NamespaceDecl 0x[[BAZ_ADDR]] <{{.*}}> line:{{.*}} baz
// CHECK:      UsingShadowDecl 0x[[SHADOW_ADDR]] {{.*}} 'foo'
