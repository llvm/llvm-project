// RUN: rm -rf %t
// RUN: split-file %s %t

// The ABI tags of an inline namespace are the union of the tags on all of its
// declarations, so the result must not depend on which module's declaration
// of the namespace happens to be deserialized (and become canonical) first.

// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t/cache -I%t \
// RUN:   -emit-llvm -o - %t/tagged-first.cpp | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-linux-gnu -fmodules \
// RUN:   -fimplicit-module-maps -fmodules-cache-path=%t/cache -I%t \
// RUN:   -emit-llvm -o - %t/untagged-first.cpp | FileCheck %s

// CHECK-DAG: define {{.*}} @_Z10use_taggedB1TB1Uv(
// CHECK-DAG: define {{.*}} @_Z12use_untaggedB1TB1Uv(
// CHECK-DAG: define {{.*}} @_Z9use_otherB1TB1Uv(

//--- module.modulemap
module Tagged { header "tagged.h" export * }
module Untagged { header "untagged.h" export * }
module Other { header "other.h" export * }

//--- tagged.h
inline namespace R __attribute__((abi_tag("T"))) { struct FromTagged {}; }

//--- untagged.h
inline namespace R { struct FromUntagged {}; }

//--- other.h
inline namespace R __attribute__((abi_tag("U"))) { struct FromOther {}; }

//--- tagged-first.cpp
#include "tagged.h"
#include "untagged.h"
#include "other.h"
FromTagged use_tagged() { return {}; }
FromUntagged use_untagged() { return {}; }
FromOther use_other() { return {}; }

//--- untagged-first.cpp
#include "tagged.h"
#include "untagged.h"
#include "other.h"
FromUntagged use_untagged() { return {}; }
FromOther use_other() { return {}; }
FromTagged use_tagged() { return {}; }
