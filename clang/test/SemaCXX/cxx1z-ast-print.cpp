//RUN: %clang_cc1 -std=c++1z -verify -ast-print %s | FileCheck %s

struct TypeSuffix {
  template <long> static int x; // expected-note {{forward declaration of template entity is here}}
  template <auto> static int y; // expected-note {{forward declaration of template entity is here}}
};
// CHECK: int k = TypeSuffix().x<0L> + TypeSuffix().y<0L>;
int k = TypeSuffix().x<0L> + TypeSuffix().y<0L>; // expected-warning {{instantiation of variable 'TypeSuffix::x<0>' required here, but no definition is available}} \
                                                 // expected-note {{add an explicit instantiation declaration to suppress this warning if 'TypeSuffix::x<0>' is explicitly instantiated in another translation unit}} \
                                                 // expected-warning {{instantiation of variable 'TypeSuffix::y<0L>' required here, but no definition is available}} \
                                                 // expected-note {{add an explicit instantiation declaration to suppress this warning if 'TypeSuffix::y<0L>' is explicitly instantiated in another translation unit}}
