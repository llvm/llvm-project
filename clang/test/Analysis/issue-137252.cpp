// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus -verify %s -DEMPTY_CLASS
// UNSUPPORTED: system-windows
// expected-no-diagnostics

// This test reproduces the issue that previously the static analyzer
// initialized an [[no_unique_address]] empty field to zero,
// over-writing a non-empty field with the same offset.

namespace std {
#ifdef EMPTY_CLASS

  struct default_delete {};
  template <class _Tp, class _Dp = default_delete >
#else
  // Class with methods and static members is still empty:
  template <typename T>
  class default_delete {
    T dump();
    static T x;
  };
  template <class _Tp, class _Dp = default_delete<_Tp> >
#endif
  class unique_ptr {
    [[no_unique_address]]  _Tp * __ptr_;
    [[no_unique_address]] _Dp __deleter_;

  public:
    explicit unique_ptr(_Tp* __p) noexcept
      : __ptr_(__p),
        __deleter_() {}

    ~unique_ptr() {
      delete __ptr_;
    }
  };
}

struct X {};

int main()
{
  // Previously a leak falsely reported here.  It was because the
  // Static Analyzer engine simulated the initialization of
  // `__deleter__` incorrectly.  The engine assigned zero to
  // `__deleter__`--an empty record sharing offset with `__ptr__`.
  // The assignment over wrote `__ptr__`.
  std::unique_ptr<X> a(new X()); 
  return 0;
}
