// RUN: not %clang_cc1 -std=c++23 -fsyntax-only %s 2>&1 | FileCheck %s
// Regression test: don't crash when an expression requirement becomes a
// substitution failure during template instantiation (see #176402).

void f() {
    auto recursiveLambda = [](auto self, int depth) -> void {
        struct MyClass;
        auto testConcept = []<typename T> {
            return requires(T) { &MyClass::operator0 }
        };
    };
    recursiveLambda(recursiveLambda, 5);
}

// CHECK: error:
// CHECK: expected ';' at end of requirement
// CHECK-NOT: Assertion failed
// CHECK-NOT: Stack dump:

