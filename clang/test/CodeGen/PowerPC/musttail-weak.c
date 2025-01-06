// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -o /dev/null -emit-llvm -verify=aix
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -o /dev/null -emit-llvm -verify=aix
// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify=linux
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -o /dev/null -emit-llvm -verify=linux

__attribute__((weak)) int func2(int i) {
  return 0;
}
int external_call2(int i) {
  // linux-error@+2 {{'musttail' attribute for this call is impossible because external calls cannot be tail called on PPC}}
  // aix-error@+1 {{'musttail' attribute is not supported on AIX}}
  [[clang::musttail]] return func2(i);
}
