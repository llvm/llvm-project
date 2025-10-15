// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -target-feature +tail-call -o /dev/null -emit-llvm -verify=tail
// RUN: %clang_cc1 %s -triple wasm32-unknown-unknown -o /dev/null -emit-llvm -verify=notail

int foo(int x) {
  return x;
}

#if __has_attribute(musttail)
// tail-warning@+1 {{HAS IT}}
#warning HAS IT
#else
// notail-warning@+1 {{DOES NOT HAVE}}
#warning DOES NOT HAVE
#endif

int bar(int x)
{
  // notail-error@+1 {{'musttail' attribute is not supported on this target without tail-call feature}}
 [[clang::musttail]] return foo(1);
}
