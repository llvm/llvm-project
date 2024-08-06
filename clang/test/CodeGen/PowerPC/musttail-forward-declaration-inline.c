// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -o /dev/null -emit-llvm -verify
// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify

inline int func2(int i);
int external_call2(int i) {
  // expected-error@+1 {{'musttail' attribute for this call is impossible because external calls can not be tail called on PPC}}
  [[clang::musttail]] return func2(i);
}

inline int func2(int i) {
  return 0;
}
