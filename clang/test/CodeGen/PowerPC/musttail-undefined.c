// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -o /dev/null -emit-llvm -verify

int foo(int x);

int bar(int x)
{
  // expected-error@+1 {{'musttail' attribute for this call is impossible because external calls cannot be tail called on PPC}}
  [[clang::musttail]] return foo(x);
}
