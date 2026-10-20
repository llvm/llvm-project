// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify=nonnull
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -verify=nullable

namespace std {
template <class T> struct unique_ptr {
  unique_ptr();
  unique_ptr(decltype(nullptr));
  explicit unique_ptr(T *);
  unique_ptr(unique_ptr &&);
  unique_ptr &operator=(unique_ptr &&);
  unique_ptr &operator=(decltype(nullptr));
  T *operator->() const;
  T &operator*() const;
  T *get() const;
  T *release();
  void reset(T * = nullptr);
  void swap(unique_ptr &);
  explicit operator bool() const;
};
template <class T> bool operator!=(const unique_ptr<T> &, decltype(nullptr));
template <class T> bool operator==(const unique_ptr<T> &, decltype(nullptr));
template <class T> T &&move(T &);
template <class T> void swap(unique_ptr<T> &, unique_ptr<T> &);
} // namespace std

struct V { int x; };
V *makeRaw();
std::unique_ptr<V> makeUnique();
std::unique_ptr<V> _Nullable makeMaybe();

int raw_factory() { V *v = makeRaw(); return v->x; } // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}

int unique_factory_arrow() {
  std::unique_ptr<V> u = makeUnique();
  return u->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int unique_factory_get_local() {
  std::unique_ptr<V> u = makeUnique();
  V *v = u.get();
  return v->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int unique_factory_get_direct() {
  std::unique_ptr<V> u = makeUnique();
  return u.get()->x;
}

int comparison_does_not_taint() {
  std::unique_ptr<V> u = makeUnique();
  bool had = u != nullptr;
  (void)had;
  return u->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int nullable_return_keeps_warning() {
  std::unique_ptr<V> u = makeMaybe();
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int default_constructed() {
  std::unique_ptr<V> u;
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int constructed_from_nullptr() {
  std::unique_ptr<V> u(nullptr);
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int assigned_null() {
  std::unique_ptr<V> u = makeUnique();
  u = nullptr;
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int after_reset() {
  std::unique_ptr<V> u = makeUnique();
  u.reset();
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int after_reset_nullptr() {
  std::unique_ptr<V> u = makeUnique();
  u.reset(nullptr);
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int after_release() {
  std::unique_ptr<V> u = makeUnique();
  V *r = u.release();
  (void)r;
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int after_move() {
  std::unique_ptr<V> u = makeUnique();
  std::unique_ptr<V> w = std::move(u);
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int after_member_swap_with_empty() {
  std::unique_ptr<V> u = makeUnique();
  std::unique_ptr<V> e;
  u.swap(e);
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int after_std_swap_with_empty() {
  std::unique_ptr<V> u = makeUnique();
  std::unique_ptr<V> e;
  std::swap(u, e);
  return u->x; // nonnull-warning {{dereference of nullable pointer}} nonnull-note {{add a null check}} nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int reset_to_factory_result() {
  std::unique_ptr<V> u;
  u.reset(makeRaw());
  return u->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

// A class deriving from a std smart pointer reaches operator->, operator*,
// get() and operator bool through a derived-to-base cast. They are checked,
// and narrow, like the smart pointer's own.
struct DerivedUnique : std::unique_ptr<V> {};

int derived_arrow(DerivedUnique d) {
  return d->x; // nullable-warning {{dereference of nullable pointer 'DerivedUnique'}} nullable-note {{add a null check}}
}

int derived_star(DerivedUnique d) {
  return (*d).x; // nullable-warning {{dereference of nullable pointer 'DerivedUnique'}} nullable-note {{add a null check}}
}

int derived_checked(DerivedUnique d) {
  if (!d)
    return 0;
  return d->x;
}

int derived_get_checked(DerivedUnique d) {
  if (d.get())
    return (*d).x;
  return 0;
}
