// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

typedef int __attribute__((address_space(1))) int_1;;
typedef int __attribute__((address_space(2))) int_2;;
typedef int __attribute__((address_space(1))) *int_1_ptr;
typedef int_2 *int_2_ptr;

// Check that we maintain address spaces through template argument
// deduction from a type.
template<typename T>
struct remove_pointer {
  typedef T type;
};

template<typename T>
struct remove_pointer<T *> {
  typedef T type;
};

int check_remove0[is_same<remove_pointer<int_1_ptr>::type, int_1>::value? 1 : -1];
int check_remove1[is_same<remove_pointer<int_2_ptr>::type, int_2>::value? 1 : -1];
int check_remove2[is_same<remove_pointer<int_2_ptr>::type, int>::value? -1 : 1];
int check_remove3[is_same<remove_pointer<int_2_ptr>::type, int_1>::value? -1 : 1];

template<typename T>
struct is_pointer_in_address_space_1 {
  static const bool value = false;
};

template<typename T>
struct is_pointer_in_address_space_1<T __attribute__((address_space(1))) *> {
  static const bool value = true;
};
                
int check_ptr_in_as1[is_pointer_in_address_space_1<int_1_ptr>::value? 1 : -1];
int check_ptr_in_as2[is_pointer_in_address_space_1<int_2_ptr>::value? -1 : 1];
int check_ptr_in_as3[is_pointer_in_address_space_1<int*>::value? -1 : 1];

// Check that we maintain address spaces through template argument
// deduction for a call.
template<typename T>
void accept_any_pointer(T*) {
  T *x = 1; // expected-error{{cannot initialize a variable of type '__attribute__((address_space(1))) int *' with an rvalue of type 'int'}} \
  // expected-error{{cannot initialize a variable of type '__attribute__((address_space(3))) int *' with an rvalue of type 'int'}}
}

void test_accept_any_pointer(int_1_ptr ip1, int_2_ptr ip2) {
  static __attribute__((address_space(3))) int array[17];
  accept_any_pointer(ip1); // expected-note{{in instantiation of}}
  accept_any_pointer(array); // expected-note{{in instantiation of}}
}

template<typename T> struct identity {};

template<typename T>
identity<T> accept_arg_in_address_space_1(__attribute__((address_space(1))) T &ir1);

template<typename T>
identity<T> accept_any_arg(T &ir1);

void test_arg_in_address_space_1() {
  static int __attribute__((address_space(1))) int_1;
  identity<int> ii = accept_arg_in_address_space_1(int_1);
  identity<int __attribute__((address_space(1)))> ii2 = accept_any_arg(int_1);
}

// Partial ordering
template<typename T> int &order1(__attribute__((address_space(1))) T&);
template<typename T> float &order1(T&);

void test_order1() {
  static __attribute__((address_space(1))) int i1;
  int i;
  int &ir = order1(i1);
  float &fr = order1(i);
}
