// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -o /dev/null -emit-llvm -verify=aix
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -o /dev/null -emit-llvm -verify=aix
// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify=linux
// RUN: %clang_cc1 %s -triple powerpc-unknown-linux-gnu -o /dev/null -emit-llvm -verify=linux
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -o /dev/null -emit-llvm -verify=linux

int Func();
int Func1() {
  // linux-warning@+2 {{'musttail' attribute may be ignored on ppc targets}}
  // aix-error@+1 {{'musttail' attribute is not supported on AIX}}
  [[clang::musttail]] return Func();
}
