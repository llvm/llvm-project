// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -x objective-c-header %S/Inputs/typo.h -emit-pch -o %t.pch
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -include-pch %t.pch %s -verify

void test() {
  [Nsstring alloc]; // expected-error {{unknown receiver 'Nsstring'; did you mean 'NSString'}}
                    // expected-note@typo.h:* {{here}}
}
