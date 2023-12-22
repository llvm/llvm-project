// RUN: %clang_cc1 %s -verify -fopenacc

namespace NS {
  void foo(); // expected-note{{declared here}}

  template<typename T>
  void templ(); // expected-note 2{{declared here}}
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
