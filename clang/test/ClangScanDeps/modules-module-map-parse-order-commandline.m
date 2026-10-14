// Check that an explicit module's command line orders dependency module maps
// in the valid parse order.

// In this example, Parent and Parent.Sub are declared in separate directories.
// For Client's explicit module, the top level Parent module must come before
// parsing Parent.Sub.
// The required ordering is the opposite of a lexicographical sort of those
// directories.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: clang-scan-deps -format experimental-full -- \
// RUN:   %clang -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:     -I %t/client -I %t/zzz -I %t/aaa -c %t/tu.m -o %t/tu.o \
// RUN:   > %t/result.json

// The map declaring Parent must still be named before the one declaring
// Parent.Sub.
// RUN: %deps-to-rsp %t/result.json --module-name=Client > %t/Client.rsp
// RUN: cat %t/Client.rsp | sed 's:\\\\\?:/:g' | FileCheck %s -DPREFIX=%/t

// CHECK:      -fmodule-map-file=[[PREFIX]]/zzz/module.modulemap
// CHECK-SAME: -fmodule-map-file=[[PREFIX]]/aaa/module.modulemap

// Verify round-trip compilation passes.
// RUN: %deps-to-rsp %t/result.json --module-name=Parent > %t/Parent.rsp
// RUN: %deps-to-rsp %t/result.json --tu-index=0 > %t/tu.rsp
// RUN: %clang @%t/Parent.rsp
// RUN: %clang @%t/Client.rsp
// RUN: %clang @%t/tu.rsp

//--- zzz/module.modulemap
module Parent { header "parent.h" }

//--- zzz/parent.h

//--- aaa/module.modulemap
explicit module Parent.Sub {
  header "sub.h"
  exclude header "sub-excluded.h"
}

//--- aaa/sub.h
//--- aaa/sub-excluded.h

//--- client/module.modulemap
module Client { header "client.h" }

//--- client/client.h
#include "parent.h"
#include "sub-excluded.h"

//--- tu.m
@import Client;
