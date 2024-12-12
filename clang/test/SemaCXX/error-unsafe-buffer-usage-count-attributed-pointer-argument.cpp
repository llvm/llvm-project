// RUN: %clang_cc1 -fsyntax-only -std=c++20 -Wno-all -fexperimental-bounds-safety-attributes -verify %s

#include <ptrcheck.h>
#include <stddef.h>

namespace output_param_test {
  void cb_out_ptr(int * __counted_by(n) * p, size_t n);
  void cb_out_count(int * __counted_by(*n) p, size_t * n);
  void cb_out_both(int * __counted_by(*n) * p, size_t * n);

  void test_no_attr(int *p, size_t n, int **q, size_t *m) {
    cb_out_ptr(&p, n);   // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
    cb_out_ptr(q, *m);   // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
    cb_out_count(p, &n); // not output ptr no error but will be warned by -Wunsafe-buffer-usage
    cb_out_count(*q, m); // not output ptr no error but will be warned by -Wunsafe-buffer-usage

    size_t local_n = n;
    int * local_p = p;

    // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
    cb_out_ptr(&local_p, local_n);
    // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}
    cb_out_both(&local_p, &local_n);
  }

  void test(int * __counted_by(n) p, size_t n, int * __counted_by(*m) *q, size_t *m) {
    cb_out_ptr(&p, *m); // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
    cb_out_ptr(q, n);   // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
    cb_out_both(&p, m); // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}
    cb_out_both(q, &n); // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}

    size_t local_n = n;
    int * __counted_by(local_n) local_p = p;

    // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
    cb_out_ptr(&local_p, n);
    // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}
    cb_out_both(&local_p, &n);
  }

  class TestClassMemberFunctions {
    void test_no_attr(int *p, size_t n, int **q, size_t *m) {
      cb_out_ptr(&p, n);   // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
      cb_out_ptr(q, *m);   // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
      cb_out_count(p, &n); // not output ptr no error but will be warned by -Wunsafe-buffer-usage
      cb_out_count(*q, m); // not output ptr no error but will be warned by -Wunsafe-buffer-usage

      size_t local_n = n;
      int * local_p = p;

      // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
      cb_out_ptr(&local_p, local_n);
      // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}
      cb_out_both(&local_p, &local_n);
    }

    void test(int * __counted_by(n) p, size_t n, int * __counted_by(*m) *q, size_t *m) {
      cb_out_ptr(&p, *m); // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
      cb_out_ptr(q, n);   // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
      cb_out_both(&p, m); // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}
      cb_out_both(q, &n); // expected-error{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}

      size_t local_n = n;
      int * __counted_by(local_n) local_p = p;

      // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(n)*' (aka 'int **')}}
      cb_out_ptr(&local_p, n);
      // expected-error@+1{{incompatible dynamic count pointer argument to parameter of type 'int * __counted_by(*n)*' (aka 'int **')}}
      cb_out_both(&local_p, &n);
    }
  };
} // namespace output_param_test
