// RUN: %clang_cc1 %s -verify -fopenacc

namespace NS {
  void foo(); // expected-note{{declared here}}

  template<typename T>
  void templ(); // expected-note 2{{declared here}}

  class C { // #CDef
    void private_mem_func(); // #PrivateMemFunc
    public:
    void public_mem_func();
  };
}

// expected-error@+2{{use of undeclared identifier 'foo'; did you mean 'NS::foo'?}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(foo)
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(NS::foo)

// expected-error@+2{{use of undeclared identifier 'templ'; did you mean 'NS::templ'?}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(templ)
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(NS::templ)

// expected-error@+2{{use of undeclared identifier 'templ'; did you mean 'NS::templ'?}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(templ<int>)
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(NS::templ<int>)

// expected-error@+2{{use of undeclared identifier 'T'}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(templ<T>)
// expected-error@+2{{use of undeclared identifier 'T'}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(NS::templ<T>)

// expected-error@+3{{expected ')'}}
// expected-note@+2{{to match this '('}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (NS::foo())

// expected-error@+2 {{expected unqualified-id}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine()

// expected-error@+2 {{expected unqualified-id}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine(int)

// expected-error@+3{{'C' does not refer to a value}}
// expected-note@#CDef{{declared here}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (NS::C)
// expected-error@+3{{'private_mem_func' is a private member of 'NS::C'}}
// expected-note@#PrivateMemFunc{{implicitly declared private here}}
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (NS::C::private_mem_func)
// expected-warning@+1{{OpenACC directives not yet implemented, pragma ignored}}
#pragma acc routine (NS::C::public_mem_func)
