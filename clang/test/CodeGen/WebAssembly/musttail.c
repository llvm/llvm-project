// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -target-feature +tail-call -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple wasm64-unknown-unknown -target-feature +tail-call -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -o /dev/null -emit-llvm -verify=notail

int foo(int x) {
  return x;
}

#if __has_attribute(musttail)
// good-warning@+2 {{HAS IT}}
// notail-warning@+1 {{HAS IT}}
#warning HAS IT
#else
#warning DOES NOT HAVE
#endif

int bar(int x)
{
 [[clang::musttail]] return foo(1);
}
