// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

[[clang::readnone]] void func_readnone(); // expected-error {{'clang::readnone' attribute only applies to parameters and implicit object parameters}}
void func_readnone(int [[clang::readnone]]); // expected-error {{'clang::readnone' attribute cannot be applied to types}}
void func_readnone([[clang::readnone]] int); // expected-error {{'clang::readnone' attribute can only be applied to pointers and references}}
void func_readnone([[clang::readnone]] int*);
void func_readnone([[clang::readnone]] int&);

[[clang::readonly]] void func_readonly(); // expected-error {{'clang::readonly' attribute only applies to parameters and implicit object parameters}}
void func_readonly(int [[clang::readonly]]); // expected-error {{'clang::readonly' attribute cannot be applied to types}}
void func_readonly([[clang::readonly]] int); // expected-error {{'clang::readonly' attribute can only be applied to pointers and references}}
void func_readonly([[clang::readonly]] int*);
void func_readonly([[clang::readonly]] int&);

[[clang::writeonly]] void func_writeonly(); // expected-error {{'clang::writeonly' attribute only applies to parameters and implicit object parameters}}
void func_writeonly(int [[clang::writeonly]]); // expected-error {{'clang::writeonly' attribute cannot be applied to types}}
void func_writeonly([[clang::writeonly]] int); // expected-error {{'clang::writeonly' attribute can only be applied to pointers and references}}
void func_writeonly([[clang::writeonly]] int*);
void func_writeonly([[clang::writeonly]] int&);

void func_mutex1([[clang::readnone, clang::readonly]] int*); // expected-error {{'clang::readonly' and 'clang::readnone' attributes are not compatible}} expected-note {{here}}
void func_mutex2([[clang::readnone, clang::writeonly]] int*); // expected-error {{'clang::writeonly' and 'clang::readnone' attributes are not compatible}} expected-note {{here}}
void func_mutex3([[clang::readonly, clang::writeonly]] int*); // expected-error {{'clang::writeonly' and 'clang::readonly' attributes are not compatible}} expected-note {{here}}

void func_arg_mismatch([[clang::readnone]] int*);
void func_arg_mismatch([[clang::readnone]] int*); // expected-note {{here}}
void func_arg_mismatch([[clang::readonly]] int*); // expected-error {{conflicting types for 'func_arg_mismatch'}}

struct S {
  void func1() [[clang::readnone]];
  void func2() [[clang::readonly]];
  void func3() [[clang::writeonly]];
  void func4() [[clang::readnone, clang::readonly]];
  void func5() [[clang::readnone, clang::writeonly]];
  void func6() [[clang::readonly, clang::writeonly]];
};
