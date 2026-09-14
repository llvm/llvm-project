// RUN: %clang_cc1 %s -I %S/Inputs -std=c++2c -fsyntax-only -verify

int wibble(); // #wibble_decl

void foo1() {
  template for (auto x : {1}) { // #foo1_instantiation
    void wibble();
    // expected-error@-1 {{functions that differ only in their return type cannot be overloaded}}
    // expected-note@#wibble_decl {{previous declaration is here}}
    // expected-note@#foo1_instantiation {{in instantiation of expansion statement requested here}}
  }
}

void foo2() {
  template for (auto x : {1}) { // #foo2_instantiation
    template for (auto x : {1}) {
      void wibble();
      // expected-error@-1 {{functions that differ only in their return type cannot be overloaded}}
      // expected-note@#wibble_decl {{previous declaration is here}}
      // expected-note@#foo2_instantiation {{in instantiation of expansion statement requested here}}
    }
  }
}

int woffle; // #woffle_decl

void foo3() {
  template for (auto x : {1}) { // #foo3_instantiation
    extern double woffle;
    // expected-error@-1 {{redeclaration of 'woffle' with a different type: 'double' vs 'int'}}
    // expected-note@#woffle_decl {{previous definition is here}}
    // expected-note@#foo3_instantiation {{in instantiation of expansion statement requested here}}
  }
}

void foo4() {
  template for (auto x : {1}) { // #foo4_instantiation
    template for (auto x : {1}) {
      extern double woffle;
      // expected-error@-1 {{redeclaration of 'woffle' with a different type: 'double' vs 'int'}}
      // expected-note@#woffle_decl {{previous definition is here}}
      // expected-note@#foo4_instantiation {{in instantiation of expansion statement requested here}}
    }
  }
}

void foo5() {
  template for (constexpr auto x : {1,2,3}) { // #foo5_instantiation
    extern int duplicated_global_var_decl[x]; // #duplicate_global_var_decl
    // expected-error@#duplicate_global_var_decl {{redeclaration of 'duplicated_global_var_decl' with a different type: 'int[2]' vs 'int[1]'}}
    // expected-note@#foo5_instantiation {{in instantiation of expansion statement requested here}}
    // expected-note@#duplicate_global_var_decl {{previous declaration is here}}

    // expected-error@#duplicate_global_var_decl {{redeclaration of 'duplicated_global_var_decl' with a different type: 'int[3]' vs 'int[1]'}}
    // expected-note@#foo5_instantiation {{in instantiation of expansion statement requested here}}
    // expected-note@#duplicate_global_var_decl {{previous declaration is here}}
  }
}

template <int n> struct StandinType {};

void foo6() {
  template for (constexpr auto x : {1,2,3}) { // #foo6_instantiation
    extern StandinType<x> duplicated_global_function_decl(); // #duplicate_global_function_decl
    // expected-error@#duplicate_global_function_decl {{functions that differ only in their return type cannot be overloaded}}
    // expected-note@#foo6_instantiation {{in instantiation of expansion statement requested here}}
    // expected-note@#duplicate_global_function_decl {{previous declaration is here}}

    // expected-error@#duplicate_global_function_decl {{functions that differ only in their return type cannot be overloaded}}
    // expected-note@#foo6_instantiation {{in instantiation of expansion statement requested here}}
    // expected-note@#duplicate_global_function_decl {{previous declaration is here}}
  }
}
void foo7() {
  template for (constexpr auto x : {true, false}) { // #foo7_instantiation
    if constexpr (x) {
      extern int qualified_decl; // #first_qualified_decl
    } else {
      extern thread_local int qualified_decl; // #mismatched_qualified_decl
      // expected-error@#mismatched_qualified_decl {{thread-local declaration of 'qualified_decl' follows non-thread-local declaration}}
      // expected-note@#foo7_instantiation {{in instantiation of expansion statement requested here}}
      // expected-note@#first_qualified_decl {{previous declaration is here}}
    }
  }
}

int foo8_decl; // #foo8_decl
void foo8() {
  template for (constexpr auto x : {true, false}) { // #foo8_instantiation
    if constexpr (x) {
      extern int foo8_decl;
    } else {
      extern thread_local int foo8_decl; // #mismatched_foo8_decl
      // expected-error@#mismatched_foo8_decl {{thread-local declaration of 'foo8_decl' follows non-thread-local declaration}}
      // expected-note@#foo8_instantiation {{in instantiation of expansion statement requested here}}
      // expected-note@#foo8_decl {{previous definition is here}}
    }
  }
}
