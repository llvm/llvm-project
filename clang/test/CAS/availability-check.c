// RUN: rm -rf %t && mkdir %t

// RUN: %clang_cc1 -triple x86_64-apple-ios14-macabi -isysroot %S/Inputs/MacOSX11.0.sdk -fsyntax-only -verify %s

// RUN: %clang -cc1depscan -o %t/cmd.rsp -fdepscan=inline -fdepscan-include-tree -cc1-args \
// RUN:     -cc1 -fcas-path %t/cas -triple x86_64-apple-ios14-macabi -isysroot %S/Inputs/MacOSX11.0.sdk %s

// FIXME: `-verify` should work with a CAS invocation.
// RUN: not %clang @%t/cmd.rsp -fsyntax-only 2> %t/out.txt
// RUN: FileCheck -input-file %t/out.txt %s
// CHECK: error: 'fUnavail' is unavailable

void fUnavail(void) __attribute__((availability(macOS, obsoleted = 10.15))); // expected-note {{marked unavailable here}}

void test() {
  fUnavail(); // expected-error {{unavailable}}
}
