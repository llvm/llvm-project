// RUN: %check_clang_tidy %s misc-misplaced-const %t -- -- -DUSING
// RUN: %check_clang_tidy %s misc-misplaced-const %t -- -- -DTYPEDEF

#ifdef TYPEDEF
typedef int int_;
typedef int *ptr_to_int;
typedef const int *ptr_to_const_int;
#endif
#ifdef USING
using int_ = int;
using ptr_to_int = int *;
using ptr_to_const_int = const int *;
#endif

void const_pointers() {
  if (const int *i = 0) {
    i = 0;
    // *i = 0;
  }

  if (const int_ *i = 0) {
    i = 0;
    // *i = 0;
  }

  if (const ptr_to_const_int i = 0) {
    // i = 0;
    // *i = 0;
  }

  // Potentially quite unexpectedly the int can be modified here
  // CHECK-MESSAGES: :[[@LINE+1]]:24: warning: 'i' declared with a const-qualified {{.*}}; results in the type being 'int *const' instead of 'const int *'
  if (const ptr_to_int i = 0) {
    //i = 0;

    *i = 0;
  }
}

template <typename Ty>
struct S {
  const Ty *i;
  const Ty &i2;
};

template struct S<int>;
template struct S<ptr_to_int>; // ok
template struct S<ptr_to_const_int>;

template <typename Ty>
struct U {
  const Ty *i;
  const Ty &i2;
};

template struct U<int *>; // ok

struct T {
  typedef void (T::*PMF)();

  void f() {
    const PMF val = &T::f; // ok
  }
};
