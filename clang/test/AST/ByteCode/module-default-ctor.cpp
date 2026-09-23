// RUN: mkdir -p %t
// RUN: %clang    -std=c++20 %p/Inputs/default-ctor.cppm --precompile -o %t/default-ctor.pcm
// RUN: %clang -c -std=c++20 -fmodule-file=m=%t/default-ctor.pcm -fexperimental-new-constant-interpreter %s -o %t/module-default-ctor.o
import m;


consteval int evaluate() {
  box b;
  return b.get();
}
static_assert(evaluate() == 42);
