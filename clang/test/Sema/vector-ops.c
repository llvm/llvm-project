// RUN: %clang_cc1 %s -verify -fsyntax-only -Wvector-conversion
typedef unsigned int v2u __attribute__ ((vector_size (8)));
typedef int v2s __attribute__ ((vector_size (8)));
typedef float v2f __attribute__ ((vector_size(8)));

void test1(v2u v2ua, v2s v2sa, v2f v2fa) {
  // Bitwise binary operators
  (void)(v2ua & v2ua);
  (void)(v2fa & v2fa); // expected-error{{invalid operands to binary expression}}

  // Unary operators
  (void)(~v2ua);
  (void)(~v2fa); // expected-error{{invalid argument type 'v2f' to unary}}

  // Comparison operators
  v2ua = (v2ua==v2sa); // expected-warning{{incompatible vector types assigning to 'v2u' from 'int __attribute__((ext_vector_type(2)))'}}
  v2sa = (v2ua==v2sa);
  
  // Arrays
  int array1[v2ua]; // expected-error{{size of array has non-integer type 'v2u'}}
  int array2[17];
  // FIXME: error message below needs type!
  (void)(array2[v2ua]); // expected-error{{array subscript is not an integer}}

  v2u *v2u_ptr = 0;
  v2s *v2s_ptr;
  v2s_ptr = v2u_ptr; // expected-warning{{converts between pointers to integer types with different sign}}
}
 
