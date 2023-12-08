// Check that the compiler wouldn't crash due to inconsistent namesapce linkage
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 -x c++ -std=c++20 %S/Inputs/p2.cppm -emit-module-interface -o %t/Y.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fprebuilt-module-path=%t -I%S/Inputs %s -fsyntax-only -verify
// expected-no-diagnostics
export module X;
import Y;

export namespace foo {
namespace bar {
void baz();
}
} // namespace foo
