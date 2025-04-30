// RUN: %clang_cc1 -verify=BEFORE,BOTH -std=c++23 %s
// RUN: %clang_cc1 -verify=AFTER,BOTH -std=c++23 %s -fimplicit-constexpr 

inline char read_byte(const char * ptr, int offset) {
  return *(ptr+offset);
  // AFTER-note@-1 {{read of dereferenced one-past-the-end pointer is not allowed in a constant expression}}
  
}

inline char normal_function(int offset) {
  // BEFORE-note@-1 {{declared here}}
  char array[8] = {'a','b','c','d','e','f','g','h'};
  return read_byte(array, offset);
  // AFTER-note@-1 {{read_byte(&array[0], 8)}}
}

constexpr char off_by_one_error = normal_function(8);
// BOTH-error@-1 {{constexpr variable 'off_by_one_error' must be initialized by a constant expression}}
// BEFORE-note@-2 {{non-constexpr function 'normal_function' cannot be used in a constant expression}}
// AFTER-note@-3 {{in call to 'normal_function(8)'}}



