// RUN: %clang_cc1 -emit-llvm-only -triple x86_64-linux-gnu -verify %s

extern "C" {
__attribute__((used)) static void *resolve_foo() { return 0; }
namespace NS {
__attribute__((used)) static void *resolve_foo() { return 0; }
} // namespace NS

// FIXME: This diagnostic is pretty confusing, the issue is that the existence
// of the two functions suppresses the 'alias' creation, and thus the ifunc
// resolution via the alias as well. In the future we should probably find
// some way to improve this diagnostic (likely by diagnosing when we decide
// this case suppresses alias creation).
__attribute__((ifunc("resolve_foo"))) void foo(); // expected-error{{ifunc must point to a defined function}}
}

