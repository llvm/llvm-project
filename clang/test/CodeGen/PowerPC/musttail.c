// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix-xcoff -o /dev/null -emit-llvm -verify=aix
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix-xcoff -o /dev/null -emit-llvm -verify=aix
// RUN: %clang_cc1 %s -triple powerpc64-unknown-linux-gnu -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple powerpc-unknown-linux-gnu -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -target-feature +pcrelative-memops -o /dev/null -emit-llvm -verify=good
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -target-feature +longcall -o /dev/null -emit-llvm -verify=longcall
// RUN: %clang_cc1 %s -triple powerpc64le-unknown-linux-gnu -target-feature +pcrelative-memops -target-feature +longcall -o /dev/null -emit-llvm -verify=good

int foo(int x) {
  return x;
}

int bar(int x)
{
  // good-no-diagnostics
  // longcall-error@+2 {{'musttail' attribute for this call is impossible because long calls cannot be tail called on PPC}}
  // aix-error@+1 {{'musttail' attribute is not supported on AIX}}
 [[clang::musttail]] return foo(1);
}
