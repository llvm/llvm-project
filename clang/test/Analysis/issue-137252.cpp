// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=cplusplus -verify %s -DEMPTY_CLASS

// expected-no-diagnostics

// This test reproduces the issue that previously the static analyzer
// initialized an [[__no_unique_address__]] empty field to zero,
// over-writing a non-empty field with the same offset.

namespace std {
#ifdef EMPTY_CLASS

  template <typename T>
  class default_delete {
    T dump();
    static T x;
  };
  template <class _Tp, class _Dp = default_delete<_Tp> >
#else

  struct default_delete {};
  template <class _Tp, class _Dp = default_delete >
#endif
  class unique_ptr {
    [[__no_unique_address__]]  _Tp * __ptr_;
    [[__no_unique_address__]] _Dp __deleter_;

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
    std::unique_ptr<X> a(new X());          // previously leak falsely reported
    return 0;
}
