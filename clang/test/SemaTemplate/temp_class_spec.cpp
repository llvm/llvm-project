// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct is_pointer {
  static const bool value = false;
};

template<typename T>
struct is_pointer<T*> {
  static const bool value = true;
};

template<typename T>
struct is_pointer<const T*> {
  static const bool value = true;
};

int array0[is_pointer<int>::value? -1 : 1];
int array1[is_pointer<int*>::value? 1 : -1];
int array2[is_pointer<const int*>::value? 1 : -1];

template<typename T>
struct is_lvalue_reference {
  static const bool value = false;
};

template<typename T>
struct is_lvalue_reference<T&> {
  static const bool value = true;
};

int lvalue_ref0[is_lvalue_reference<int>::value? -1 : 1];
int lvalue_ref1[is_lvalue_reference<const int&>::value? 1 : -1];

template<typename T>
struct is_const {
  static const bool value = false;
};

template<typename T>
struct is_const<const T> {
  static const bool value = true;
};

int is_const0[is_const<int>::value? -1 : 1];
int is_const1[is_const<const int>::value? 1 : -1];
int is_const2[is_const<const volatile int>::value? 1 : -1];
int is_const3[is_const<const int [3]>::value? 1 : -1];
int is_const4[is_const<const volatile int[3]>::value? 1 : -1];
int is_const5[is_const<volatile int[3]>::value? -1 : 1];

template<typename T>
struct is_volatile {
  static const bool value = false;
};

template<typename T>
struct is_volatile<volatile T> {
  static const bool value = true;
};

int is_volatile0[is_volatile<int>::value? -1 : 1];
int is_volatile1[is_volatile<volatile int>::value? 1 : -1];
int is_volatile2[is_volatile<const volatile int>::value? 1 : -1];
int is_volatile3[is_volatile<volatile char[3]>::value? 1 : -1];
                 
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

typedef int INT;
typedef INT* int_ptr;

int is_same0[is_same<int, int>::value? 1 : -1];
int is_same1[is_same<int, INT>::value? 1 : -1];
int is_same2[is_same<const int, int>::value? -1 : 1];
int is_same3[is_same<int_ptr, int>::value? -1 : 1];

template<typename T>
struct remove_reference {
  typedef T type;
};

template<typename T>
struct remove_reference<T&> {
  typedef T type;
};

int remove_ref0[is_same<remove_reference<int>::type, int>::value? 1 : -1];
int remove_ref1[is_same<remove_reference<int&>::type, int>::value? 1 : -1];

template<typename T>
struct remove_const {
  typedef T type;
};

template<typename T>
struct remove_const<const T> {
  typedef T type;
};

int remove_const0[is_same<remove_const<const int>::type, int>::value? 1 : -1];
int remove_const1[is_same<remove_const<const int[3]>::type, int[3]>::value? 1 : -1];

template<typename T>
struct is_incomplete_array {
  static const bool value = false;
};

template<typename T>
struct is_incomplete_array<T[]> {
  static const bool value = true;
};

int incomplete_array0[is_incomplete_array<int>::value ? -1 : 1];
int incomplete_array1[is_incomplete_array<int[1]>::value ? -1 : 1];
int incomplete_array2[is_incomplete_array<bool[]>::value ? 1 : -1];
int incomplete_array3[is_incomplete_array<int[]>::value ? 1 : -1];

template<typename T>
struct is_array_with_4_elements {
  static const bool value = false;
};

template<typename T>
struct is_array_with_4_elements<T[4]> {
  static const bool value = true;
};

int array_with_4_elements0[is_array_with_4_elements<int[]>::value ? -1 : 1];
int array_with_4_elements1[is_array_with_4_elements<int[1]>::value ? -1 : 1];
int array_with_4_elements2[is_array_with_4_elements<int[4]>::value ? 1 : -1];
int array_with_4_elements3[is_array_with_4_elements<int[4][2]>::value ? 1 : -1];

template<typename T>
struct get_array_size;

template<typename T, unsigned N>
struct get_array_size<T[N]> {
  static const unsigned value = N;
};

int array_size0[get_array_size<int[12]>::value == 12? 1 : -1];

template<typename T>
struct remove_extent {
  typedef T type;
};

template<typename T>
struct remove_extent<T[]> { 
  typedef T type;
};

template<typename T, unsigned N>
struct remove_extent<T[N]> { 
  typedef T type;
};

int remove_extent0[is_same<remove_extent<int[][5]>::type, int[5]>::value? 1 : -1];
int remove_extent1[is_same<remove_extent<const int[][5]>::type, const int[5]>::value? 1 : -1];

template<typename T>
struct is_unary_function {
  static const bool value = false;
};

template<typename T, typename U>
struct is_unary_function<T (*)(U)> {
  static const bool value = true;
};

int is_unary_function0[is_unary_function<int>::value ? -1 : 1];
int is_unary_function1[is_unary_function<int (*)()>::value ? -1 : 1];
int is_unary_function2[is_unary_function<int (*)(int, bool)>::value ? -1 : 1];
int is_unary_function3[is_unary_function<int (*)(bool)>::value ? 1 : -1];
int is_unary_function4[is_unary_function<int (*)(int)>::value ? 1 : -1];

template<typename T>
struct is_unary_function_with_same_return_type_as_argument_type {
  static const bool value = false;
};

template<typename T>
struct is_unary_function_with_same_return_type_as_argument_type<T (*)(T)> {
  static const bool value = true;
};

int is_unary_function5[is_unary_function_with_same_return_type_as_argument_type<int>::value ? -1 : 1];
int is_unary_function6[is_unary_function_with_same_return_type_as_argument_type<int (*)()>::value ? -1 : 1];
int is_unary_function7[is_unary_function_with_same_return_type_as_argument_type<int (*)(int, bool)>::value ? -1 : 1];
int is_unary_function8[is_unary_function_with_same_return_type_as_argument_type<int (*)(bool)>::value ? -1 : 1];
int is_unary_function9[is_unary_function_with_same_return_type_as_argument_type<int (*)(int)>::value ? 1 : -1];
int is_unary_function10[is_unary_function_with_same_return_type_as_argument_type<int (*)(int, ...)>::value ? -1 : 1];
int is_unary_function11[is_unary_function_with_same_return_type_as_argument_type<int (* const)(int)>::value ? -1 : 1];

template<typename T>
struct is_binary_function {
  static const bool value = false;
};

template<typename R, typename T1, typename T2>
struct is_binary_function<R(T1, T2)> {
  static const bool value = true;
};

int is_binary_function0[is_binary_function<int(float, double)>::value? 1 : -1];

template<typename T>
struct is_member_pointer {
  static const bool value = false;
};

template<typename T, typename Class>
struct is_member_pointer<T Class::*> {
  static const bool value = true;
};

struct X { };

int is_member_pointer0[is_member_pointer<int X::*>::value? 1 : -1];
int is_member_pointer1[is_member_pointer<const int X::*>::value? 1 : -1];
int is_member_pointer2[is_member_pointer<int (X::*)()>::value? 1 : -1];
int is_member_pointer3[is_member_pointer<int (X::*)(int) const>::value? 1 : -1];
int is_member_pointer4[is_member_pointer<int (X::**)(int) const>::value? -1 : 1];
int is_member_pointer5[is_member_pointer<int>::value? -1 : 1];

template<typename T>
struct is_member_function_pointer {
  static const bool value = false;
};

template<typename T, typename Class>
struct is_member_function_pointer<T (Class::*)()> {
  static const bool value = true;
};

template<typename T, typename Class>
struct is_member_function_pointer<T (Class::*)() const> {
  static const bool value = true;
};

template<typename T, typename Class>
struct is_member_function_pointer<T (Class::*)() volatile> {
  static const bool value = true;
};

template<typename T, typename Class>
struct is_member_function_pointer<T (Class::*)() const volatile> {
  static const bool value = true;
};

template<typename T, typename Class, typename A1>
struct is_member_function_pointer<T (Class::*)(A1)> {
  static const bool value = true;
};

template<typename T, typename Class, typename A1>
struct is_member_function_pointer<T (Class::*)(A1) const> {
  static const bool value = true;
};

template<typename T, typename Class, typename A1>
struct is_member_function_pointer<T (Class::*)(A1) volatile> {
  static const bool value = true;
};

template<typename T, typename Class, typename A1>
struct is_member_function_pointer<T (Class::*)(A1) const volatile> {
  static const bool value = true;
};

int is_member_function_pointer0[
                          is_member_function_pointer<int X::*>::value? -1 : 1];
int is_member_function_pointer1[
                      is_member_function_pointer<int (X::*)()>::value? 1 : -1];
int is_member_function_pointer2[
                      is_member_function_pointer<X (X::*)(X&)>::value? 1 : -1];
int is_member_function_pointer3[
           is_member_function_pointer<int (X::*)() const>::value? 1 : -1];
int is_member_function_pointer4[
           is_member_function_pointer<int (X::*)(float) const>::value? 1 : -1];

// Test substitution of non-dependent arguments back into the template
// argument list of the class template partial specialization.
template<typename T, typename ValueType = T>
struct is_nested_value_type_identity {
  static const bool value = false;
};

template<typename T>
struct is_nested_value_type_identity<T, typename T::value_type> {
  static const bool value = true;
};

template<typename T>
struct HasValueType {
  typedef T value_type;
};

struct HasIdentityValueType {
  typedef HasIdentityValueType value_type;
};

struct NoValueType { };

int is_nested_value_type_identity0[
            is_nested_value_type_identity<HasValueType<int> >::value? -1 : 1];
int is_nested_value_type_identity1[
          is_nested_value_type_identity<HasIdentityValueType>::value? 1 : -1];
int is_nested_value_type_identity2[
                   is_nested_value_type_identity<NoValueType>::value? -1 : 1];


// C++ [temp.class.spec]p4:
template<class T1, class T2, int I> class A { }; //#1 
template<class T, int I> class A<T, T*, I> { }; //#2 
template<class T1, class T2, int I> class A<T1*, T2, I> { }; //#3 
template<class T> class A<int, T*, 5> { }; //#4 
template<class T1, class T2, int I> class A<T1, T2*, I> { }; //#5 

// Redefinition of class template partial specializations
template<typename T, T N, typename U> class A0;

template<typename T, T N> class A0<T, N, int> { }; // expected-note{{here}}
template<typename T, T N> class A0<T, N, int>;
template<typename T, T N> class A0<T, N, int> { }; // expected-error{{redef}}

namespace PR6025 {
  template< int N > struct A;

  namespace N 
  {
    template< typename F > 
    struct B;
  }

  template< typename Protect, typename Second > 
  struct C;

  template <class T>
  struct C< T, A< N::B<T>::value > >
  {
  };
}

namespace PR6181 {
  template <class T>
  class a;
  
  class s;
  
  template <class U>
  class a<s> // expected-error{{partial specialization of 'a' does not use any of its template parameters}}
  {
  };
  
}
