// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -fsyntax-only -verify=aix
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -fsyntax-only -verify=aix
// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -fsyntax-only -verify=linux
// RUN: %clang_cc1 %s -triple powerpc-unknown-linux-gnu -fsyntax-only -verify=linux
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -fsyntax-only -verify=linux
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -target-feature +pcrelative-memops -fsyntax-only -verify=good
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -target-feature +longcall -fsyntax-only -verify=longcall
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -target-feature +pcrelative-memops -target-feature +longcall -fsyntax-only -verify=good

int good_callee() {
  return 0;
}
int good_caller() {
  // good-no-diagnostics
  // longcall-error@+2 {{'musttail' attribute for this call is impossible because long calls can not be tail called}}
  // aix-error@+1 {{'musttail' attribute is not supported on AIX}}
  [[clang::musttail]] return good_callee();
}

int func();
int external_call() {
  // good-no-diagnostics
  // longcall-error@+3 {{'musttail' attribute for this call is impossible because long calls can not be tail called}}
  // linux-error@+2 {{'musttail' attribute for this call is impossible because external calls can not be tail called}}
  // aix-error@+1 {{'musttail' attribute is not supported on AIX}}
  [[clang::musttail]] return func();
}

void indirect_call(int r) {
  auto Fn = (void (*)(int))1;
  // good-no-diagnostics
  // longcall-error@+3 {{'musttail' attribute for this call is impossible because long calls can not be tail called}}
  // linux-error@+2 {{'musttail' attribute for this call is impossible because indirect calls can not be tail called}}
  // aix-error@+1 {{'musttail' attribute is not supported on AIX}}
  [[clang::musttail]] return Fn(r);
}
