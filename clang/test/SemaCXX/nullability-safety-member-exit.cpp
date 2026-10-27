// A _Nonnull smart pointer member that reset(), release() or a move leaves
// null on every path to the end of a method is reported once, at the end of
// the method. Those operations do not warn where they happen (the member may
// be refilled before returning), and a raw pointer member already warns at
// the assignment that nulls it. Destructors and &&-qualified methods are
// exempt: leaving the object empty is their job.

// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -verify
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify

namespace std {
template <typename T> struct unique_ptr {
  T *ptr;
  unique_ptr();
  unique_ptr(T *p);
  unique_ptr(unique_ptr &&);
  unique_ptr &operator=(unique_ptr &&);
  T *operator->() const;
  T *release();
  void reset();
  void reset(T *p);
  explicit operator bool() const;
};
template <typename T> T &&move(T &t) noexcept;
template <typename T, typename... A> unique_ptr<T> make_unique(A &&...);
} // namespace std

struct R { int x; };

struct Owner {
  std::unique_ptr<R> _Nonnull res;
  std::unique_ptr<R> plain;
  int *_Nonnull raw;

  void reset_only() {
    res.reset();
  } // expected-warning {{nonnull member 'res' is null when the function returns}} expected-note {{assign it a value before returning}}

  void release_only() {
    R *r = res.release();
    (void)r;
  } // expected-warning {{nonnull member 'res' is null when the function returns}} expected-note {{assign it a value before returning}}

  std::unique_ptr<R> take() {
    std::unique_ptr<R> out = std::move(res);
    return out;
  } // expected-warning {{nonnull member 'res' is null when the function returns}} expected-note {{assign it a value before returning}}

  void refilled() {
    res.reset();
    res = std::make_unique<R>();
  }

  void reset_with_value() {
    res.reset(new R);
  }

  void one_path(bool c) {
    if (c)
      res.reset();
  }

  void plain_member() {
    plain.reset();
  }

  void raw_member() {
    raw = nullptr; // expected-warning {{assigning nullable pointer to nonnull member 'raw'}} expected-note {{add a null check}}
  }

  std::unique_ptr<R> consume() && {
    return std::move(res);
  }

  ~Owner() {
    res.reset();
  }

  void other(Owner &o) {
    o.res.reset();
  }
};
