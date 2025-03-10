// RUN: %clang_cc1 --std=c++1y -fsyntax-only -verify %s
// RUN: cp %s %t
// RUN: not %clang_cc1 --std=c++1y -x c++ -fixit %t -DFIXING
// RUN: %clang_cc1 --std=c++1y -x c++ %t -DFIXING

template<typename T> // expected-note {{template parameter is declared here}}
T pi = T(3.1415926535897932385);

template int pi<int>;

#ifndef FIXING
template float pi<>; // expected-error {{missing template argument for template parameter}}
template double pi_var0; // expected-error {{explicit instantiation of 'pi_var0' does not refer to a function template, variable template, member function, member class, or static data member}}
#endif

// Should recover as if definition
template double pi_var = 5; // expected-error {{variable cannot be defined in an explicit instantiation; if this declaration is meant to be a variable definition, remove the 'template' keyword}}
#ifndef FIXING
template<typename T>
T pi0 = T(3.1415926535897932385); // expected-note {{previous definition is here}}

template int pi0 = 10; // expected-error {{variable cannot be defined in an explicit instantiation; if this declaration is meant to be a variable definition, remove the 'template' keyword}} \
                          expected-error{{redefinition of 'pi0' as different kind of symbol}}
#endif

template<typename T>
T pi1 = T(3.1415926535897932385); // expected-note 0-2 {{here}}

// Should recover as if specialization
template float pi1<float> = 1.0;  // expected-error {{explicit template instantiation cannot have a definition; if this definition is meant to be an explicit specialization, add '<>' after the 'template' keyword}}
namespace expected_global {
#ifndef FIXING
  template int pi1<int> = 10;  // expected-error {{explicit template instantiation cannot have a definition; if this definition is meant to be an explicit specialization, add '<>' after the 'template' keyword}} expected-error {{must occur at global scope}}
#endif
}
