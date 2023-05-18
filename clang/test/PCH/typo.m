// RUN: %clang_cc1 -x objective-c-header -emit-pch -o %t %S/Inputs/typo.h
// RUN: %clang_cc1 -include-pch %t -verify %s

void f(void) {
  [NSstring alloc]; // expected-error{{unknown receiver 'NSstring'; did you mean 'NSString'?}}
                    // expected-note@Inputs/typo.h:3{{declared here}}
}
