// RUN: c-index-test -test-load-source all %s 2>&1 | FileCheck %s

// This test case previously crashed Sema.

extern "C" { @implementation Foo  - (id)initWithBar:(Baz<WozBar>)pepper {

// CHECK: warning: cannot find interface declaration for 'Foo'
