// RUN: c-index-test core -print-source-symbols -- -std=c++20 %s | FileCheck %s

namespace ns { void foo(); }
namespace ns2 { void foo(int); }

namespace exporting {
namespace {
using ns::foo;
// CHECK: [[@LINE-1]]:11 | using/C++ | foo | c:using-decl.cpp@N@exporting@aN@UD@N@ns@foo
using ::ns::foo;
// CHECK: [[@LINE-1]]:13 | using/C++ | foo | c:using-decl.cpp@N@exporting@aN@UD@N@ns@foo
// FIXME: Also put the qualified name for the target decl
using ns2::foo;
// CHECK: [[@LINE-1]]:12 | using/C++ | foo | c:using-decl.cpp@N@exporting@aN@UD@N@ns2@foo
}
}
