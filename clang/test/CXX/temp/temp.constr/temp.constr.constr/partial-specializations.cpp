// RUN: %clang_cc1 -std=c++2a -x c++ -verify %s

namespace class_templates
{
  template<typename T, typename U> requires (sizeof(T) >= 4) // expected-note {{because 'sizeof(char) >= 4' (1 >= 4) evaluated to false}}
  struct is_same { static constexpr bool value = false; };

  template<typename T> requires (sizeof(T*) >= 4 && sizeof(T) >= 4)
  struct is_same<T*, T*> { static constexpr bool value = true; };

  static_assert(!is_same<char*, char*>::value);
  static_assert(!is_same<short*, short*>::value);
  static_assert(is_same<int*, int*>::value);
  static_assert(is_same<char, char>::value); // expected-error {{constraints not satisfied for class template 'is_same' [with T = char, U = char]}}

  template<typename T>
  struct A { using type = typename T::type; }; // expected-error{{type 'int *' cannot be used prior to '::' because it has no members}}

  template<typename T>
  struct B {};

  template<typename T> requires A<T>::type // expected-note{{in instantiation of template class 'class_templates::A<int *>' requested here}}
                                           // expected-note@-1{{while substituting template arguments into constraint expression here}}
  struct B<T*> {};

  template<typename T> requires (T{}) // expected-error{{atomic constraint must be of type 'bool' (found 'int')}}
  struct B<T**> {};

  static_assert(((void)B<int**>{}, true)); // expected-note{{while checking constraint satisfaction for class template partial specialization 'B<int *>' required here}}
  // expected-note@-1{{while checking constraint satisfaction for class template partial specialization 'B<int>' required here}}
  // expected-note@-2{{during template argument deduction for class template partial specialization 'B<T *>' [with T = int *]}}
  // expected-note@-3{{during template argument deduction for class template partial specialization 'B<T **>' [with T = int]}}
  // expected-note@-4 2{{in instantiation of template class 'class_templates::B<int **>' requested here}}

  template<typename T, typename U = double>
  concept same_as = is_same<T, U>::value;

  template<same_as<bool> T> requires A<T>::type
  struct B<T*> {};
  // expected-note@-1{{previous}}

  template<same_as<bool> T> requires A<T>::type
  struct B<T*> {};
  // expected-error@-1{{redefinition}}

  template<same_as T> requires A<T>::type
  struct B<T*> {};

  template<same_as<int> T> requires A<T>::type
  struct B<T*> {};
}

namespace variable_templates
{
  template<typename T, typename U> requires (sizeof(T) >= 4)
  constexpr bool is_same_v = false;

  template<typename T> requires (sizeof(T*) >= 4 && sizeof(T) >= 4)
  constexpr bool is_same_v<T*, T*> = true;

  static_assert(!is_same_v<char*, char*>);
  static_assert(!is_same_v<short*, short*>);
  static_assert(is_same_v<int*, int*>);

  template<typename T>
  struct A { using type = typename T::type; }; // expected-error{{type 'int *' cannot be used prior to '::' because it has no members}}

  template<typename T>
  constexpr bool v1 = false;

  template<typename T> requires A<T>::type // expected-note{{in instantiation of template class 'variable_templates::A<int *>' requested here}}
                                           // expected-note@-1{{while substituting template arguments into constraint expression here}}
  constexpr bool v1<T*> = true;

  template<typename T> requires (T{}) // expected-error{{atomic constraint must be of type 'bool' (found 'int')}}
  constexpr bool v1<T**> = true;

  static_assert(v1<int**>); // expected-note{{while checking constraint satisfaction for variable template partial specialization 'v1<int *>' required here}}
  // expected-note@-1{{while checking constraint satisfaction for variable template partial specialization 'v1<int>' required here}}
  // expected-note@-2{{during template argument deduction for variable template partial specialization 'v1<T *>' [with T = int *]}}
  // expected-note@-3{{during template argument deduction for variable template partial specialization 'v1<T **>' [with T = int]}}
  // expected-error@-4{{static_assert failed due to requirement 'v1<int **>'}}

}