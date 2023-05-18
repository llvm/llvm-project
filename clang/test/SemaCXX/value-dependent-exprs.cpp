// RUN: %clang_cc1 -verify %s
// expected-no-diagnostics

template <unsigned I>
class C0 {
  static const int iv0 = 1 << I;

  enum {
    A = I,
    B = I + 1
  };

  struct s0 {
    int a : I;
    int b[I];
  };

  // FIXME: I'm unclear where the right place to handle this is.
#if 0
  void f0(int *p) {
    if (p == I) {
    }
  }
#endif

#if 0
  // FIXME: Not sure whether we care about these.
  void f1(int *a)
    __attribute__((nonnull(1 + I)))
    __attribute__((constructor(1 + I)))
    __attribute__((destructor(1 + I)))
    __attribute__((sentinel(1 + I, 2 + I))),
    __attribute__((reqd_work_group_size(1 + I, 2 + I, 3 + I))),
    __attribute__((format_arg(1 + I))),
    __attribute__((aligned(1 + I))),
    __attribute__((regparm(1 + I)));

  typedef int int_a0 __attribute__((address_space(1 + B)));
#endif

#if 0
  // FIXME: This doesn't work. PR4996.
  int f2() {
    return __builtin_choose_expr(I, 1, 2);
  }
#endif

};
