// Nullability written on a smart pointer that is reached through a
// reference: the annotation sits on the referenced type
// (const std::shared_ptr<T> _Nonnull &), not on the reference, both for the
// dereference check and for opting a function in without
// -fnullability-default. Also: a local initialized from a call declared to
// return a _Nonnull smart pointer is non-null.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++20 %s -verify=expected,nullable
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++20 %s -verify=expected
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c++20 %s -verify=expected

namespace std {
using nullptr_t = decltype(nullptr);
template <class T> class shared_ptr {
public:
  shared_ptr() noexcept;
  shared_ptr(nullptr_t) noexcept;
  shared_ptr(const shared_ptr &) noexcept;
  shared_ptr(shared_ptr &&) noexcept;
  ~shared_ptr();
  shared_ptr &operator=(const shared_ptr &) noexcept;
  shared_ptr &operator=(shared_ptr &&) noexcept;
  T &operator*() const noexcept;
  T *operator->() const noexcept;
  explicit operator bool() const noexcept;
  void reset() noexcept;
};
} // namespace std

struct S {
  int x;
};

int nonnull_const_ref(const std::shared_ptr<S> _Nonnull &p) {
  return p->x;
}

int nonnull_ref_star(std::shared_ptr<S> _Nonnull &p) {
  return (*p).x;
}

int nonnull_rvalue_ref(std::shared_ptr<S> _Nonnull &&p) {
  return p->x;
}

int nonnull_local_ref(std::shared_ptr<S> _Nonnull q) {
  const std::shared_ptr<S> _Nonnull &r = q;
  return r->x;
}

// A local reference without its own annotation reads its referent's.
int local_ref_of_nonnull(std::shared_ptr<S> _Nonnull q) {
  const std::shared_ptr<S> &r = q;
  return r->x;
}

// The annotation on the reference parameter alone opts the function in, so
// this warns under every default, including the unspecified one.
int nullable_const_ref(const std::shared_ptr<S> _Nullable &p) {
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int nullable_const_ref_checked(const std::shared_ptr<S> _Nullable &p) {
  if (!p)
    return 0;
  return p->x;
}

// Flow facts still override the declared contract.
int nonnull_ref_after_reset(std::shared_ptr<S> _Nonnull &p) {
  p.reset();
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

struct Holder {
  const std::shared_ptr<S> _Nonnull &r;
  const std::shared_ptr<S> _Nullable &n;
};

int nonnull_ref_member(Holder h, S *_Nonnull unused) {
  return h.r->x;
}

int nullable_ref_member(Holder h, S *_Nonnull unused) {
  return h.n->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

// A function whose only annotation is its reference return type is analyzed.
const std::shared_ptr<S> _Nonnull &nonnull_ref_return(
    const std::shared_ptr<S> _Nullable &p, const std::shared_ptr<S> &q) {
  (void)p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  return q;
}

std::shared_ptr<S> _Nonnull make_nonnull();
const std::shared_ptr<S> _Nonnull &nonnull_ref_getter();

int explicit_local_from_nonnull_call(S *_Nonnull unused) {
  std::shared_ptr<S> q = make_nonnull();
  return q->x;
}

int auto_local_from_nonnull_call(S *_Nonnull unused) {
  auto q = make_nonnull();
  return q->x;
}

int ref_local_from_nonnull_getter(S *_Nonnull unused) {
  const std::shared_ptr<S> &r = nonnull_ref_getter();
  return r->x;
}

int assigned_from_nonnull_call(S *_Nonnull unused) {
  std::shared_ptr<S> q = nullptr;
  q = make_nonnull();
  return q->x;
}

// Unannotated: judged by the default only (unspecified mode does not analyze
// these functions at all).
int plain_const_ref(const std::shared_ptr<S> &p) {
  return p->x; // nullable-warning {{dereference of nullable pointer}} nullable-note {{add a null check}}
}
