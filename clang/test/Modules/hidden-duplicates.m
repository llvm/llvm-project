// XFAIL: target={{.*}}-aix{{.*}}, target={{.*}}-zos{{.*}}
// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -I %t/include %t/test.m -verify \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -I %t/include %t/test.m -verify -DTEST_MAKE_HIDDEN_VISIBLE=1 \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -I %t/include %t/test.m -verify -x objective-c++ \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache
// RUN: %clang_cc1 -emit-llvm -o %t/test.bc -I %t/include %t/test.m -verify -DTEST_MAKE_HIDDEN_VISIBLE=1 -x objective-c++ \
// RUN:            -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/modules.cache

// Test parsing duplicate Objective-C entities when a previous entity is defined
// in a hidden [sub]module and cannot be used.
//
// Testing with header guards and includes on purpose as tracking imports in
// modules is a separate issue.

//--- include/textual.h
#ifndef TEXTUAL_H
#define TEXTUAL_H

@protocol TestProtocol
- (void)someMethod;
@end

@protocol ForwardDeclaredProtocolWithoutDefinition;

id<TestProtocol> protocolDefinition(id<TestProtocol> t);
id<ForwardDeclaredProtocolWithoutDefinition> forwardDeclaredProtocol(
    id<ForwardDeclaredProtocolWithoutDefinition> t);

@interface NSObject @end
@class ForwardDeclaredInterfaceWithoutDefinition;

NSObject *interfaceDefinition(NSObject *o);
NSObject *forwardDeclaredInterface(NSObject *o);

#endif

//--- include/empty.h
//--- include/initially_hidden.h
#include "textual.h"

//--- include/module.modulemap
module Piecewise {
  module Empty {
    header "empty.h"
  }
  module InitiallyHidden {
    header "initially_hidden.h"
    export *
  }
}

//--- test.m
// Including empty.h loads the entire module Piecewise but keeps InitiallyHidden hidden.
#include "empty.h"
#include "textual.h"
#ifdef TEST_MAKE_HIDDEN_VISIBLE
#include "initially_hidden.h"
#endif
// expected-no-diagnostics
