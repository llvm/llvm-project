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

// expected-error@+1{{use of undeclared identifier 'foo'; did you mean 'NS::foo'?}}
#pragma acc routine(foo)
#pragma acc routine(NS::foo)

// expected-error@+2{{use of undeclared identifier 'templ'; did you mean 'NS::templ'?}}
// expected-error@+1{{OpenACC routine name 'NS::templ' names a set of overloads}}
#pragma acc routine(templ)
// expected-error@+1{{OpenACC routine name 'NS::templ' names a set of overloads}}
#pragma acc routine(NS::templ)

// expected-error@+2{{use of undeclared identifier 'templ'; did you mean 'NS::templ'?}}
// expected-error@+1{{OpenACC routine name 'NS::templ' names a set of overloads}}
#pragma acc routine(templ<int>)
// expected-error@+1{{OpenACC routine name 'NS::templ<int>' names a set of overloads}}
#pragma acc routine(NS::templ<int>)

// expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(templ<T>)
// expected-error@+1{{use of undeclared identifier 'T'}}
#pragma acc routine(NS::templ<T>)

// expected-error@+2{{expected ')'}}
// expected-note@+1{{to match this '('}}
#pragma acc routine (NS::foo())

// expected-error@+1 {{expected unqualified-id}}
#pragma acc routine()

// expected-error@+1 {{expected unqualified-id}}
#pragma acc routine(int)

// expected-error@+2{{'C' does not refer to a value}}
// expected-note@#CDef{{declared here}}
#pragma acc routine (NS::C)
// expected-error@+2{{'private_mem_func' is a private member of 'NS::C'}}
// expected-note@#PrivateMemFunc{{implicitly declared private here}}
#pragma acc routine (NS::C::private_mem_func)
#pragma acc routine (NS::C::public_mem_func)
