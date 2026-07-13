// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify
// RUN: %clang_cc1 %s -triple powerpc-unknown-linux-gnu -o /dev/null -emit-llvm -verify

void name(int *params) {
  auto fn = (void (*)(int *))1;
  // expected-error@+1 {{'musttail' attribute for this call is impossible because indirect calls cannot be tail called on PPC}}
  [[clang::musttail]] return fn(params);
}
