// libstdc++ declares shared_ptr's operator->, operator*, get(), operator bool
// and reset() on base classes, so the receiver reaches them through a
// derived-to-base cast. The stub below has the same shape; dereferences,
// narrowing and taint must behave as when the members are on shared_ptr.

// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -verify=expected,nullable
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify=expected

namespace std {
template <typename T> class __shared_ptr_access {
public:
  T &operator*() const;
  T *operator->() const;
};
template <typename T> class __shared_ptr : public __shared_ptr_access<T> {
public:
  T *get() const;
  explicit operator bool() const;
  void reset();
  void swap(__shared_ptr &);
};
template <typename T> class shared_ptr : public __shared_ptr<T> {
public:
  shared_ptr();
  shared_ptr(decltype(nullptr));
};
} // namespace std

struct S { int x; };

int arrow(std::shared_ptr<S> s) {
  return s->x; // nullable-warning {{dereference of nullable pointer 'std::shared_ptr<S>'}} nullable-note {{add a null check}}
}

int star(std::shared_ptr<S> s) {
  return (*s).x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int checked(std::shared_ptr<S> s) {
  if (s)
    return s->x;
  return 0;
}

int early_return(std::shared_ptr<S> s) {
  if (!s)
    return 0;
  return (*s).x;
}

int checked_get(std::shared_ptr<S> s) {
  if (s) {
    S *p = s.get();
    return p->x;
  }
  return 0;
}

int after_reset(std::shared_ptr<S> s) {
  if (!s)
    return 0;
  s.reset();
  return s->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int null_branch(std::shared_ptr<S> s) {
  if (s)
    return 0;
  return s->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}

int declared_nonnull(std::shared_ptr<S> _Nonnull s) {
  return s->x;
}

int declared_nullable(std::shared_ptr<S> _Nullable s) {
  return s->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int get_checked(std::shared_ptr<S> s) {
  if (s.get())
    return s->x;
  return 0;
}

int get_compared(const std::shared_ptr<S> &s) {
  if (s.get() != nullptr)
    return (*s).x;
  return 0;
}

int get_negated(std::shared_ptr<S> s) {
  if (!s.get())
    return 0;
  return s->x;
}

int get_null_branch(std::shared_ptr<S> s) {
  if (s.get() == nullptr)
    return s->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
  return 0;
}

// A user class deriving from shared_ptr reaches the same members through a
// longer derived-to-base path: the object is cast straight to
// __shared_ptr_access, and shared_ptr appears only on the cast's path.
struct DerivedShared : std::shared_ptr<S> {};

int derived_arrow(DerivedShared d) {
  return d->x; // nullable-warning {{dereference of nullable pointer 'DerivedShared'}} nullable-note {{add a null check}}
}

int derived_star(const DerivedShared &d) {
  return (*d).x; // nullable-warning {{dereference of nullable pointer 'const DerivedShared &'}} nullable-note {{add a null check}}
}

int derived_checked(DerivedShared d) {
  if (d)
    return d->x;
  return 0;
}

int derived_get_checked(DerivedShared d) {
  if (!d.get())
    return 0;
  return (*d).x;
}
