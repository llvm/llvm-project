// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify=good

int func2(int i);
int external_call2(int i) {
  // good-no-diagnostics
  [[clang::musttail]] return func2(i);
}
int func2(int i) {
  return 0;
}
