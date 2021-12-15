// Check that the compiler wouldn't crash due to inconsistent namesapce linkage
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang -std=c++20 %S/Inputs/p2.cppm --precompile -o %t/Y.pcm
// RUN: %clang -std=c++20 -fprebuilt-module-path=%t -I%S/Inputs %s -c -Xclang -verify
// expected-no-diagnostics
export module X;
import Y;

export namespace foo {
namespace bar {
void baz();
}
} // namespace foo
