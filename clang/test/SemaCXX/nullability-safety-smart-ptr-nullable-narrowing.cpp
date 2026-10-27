// A smart pointer declared _Nullable is narrowed by a null check like a raw
// pointer: the flow state after `if (p)` wins over the declared type.

// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -std=c++17 %s -verify
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c++17 %s -verify

namespace std {
template <typename T> struct unique_ptr {
  T *ptr;
  unique_ptr() : ptr(nullptr) {}
  unique_ptr(T *p) : ptr(p) {}
  T *operator->() const { return ptr; }
  T &operator*() const { return *ptr; }
  explicit operator bool() const { return ptr != nullptr; }
};
} // namespace std

struct W {
  int x;
};

std::unique_ptr<W> _Nullable make();

int checked_arrow() {
  std::unique_ptr<W> _Nullable p = make();
  if (p)
    return p->x;
  return 0;
}

int checked_star() {
  std::unique_ptr<W> _Nullable p = make();
  if (!p)
    return 0;
  return (*p).x;
}

int checked_param(std::unique_ptr<W> _Nullable p) {
  return p ? p->x : 0;
}

int unchecked(std::unique_ptr<W> _Nullable p) {
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

int wrong_branch(std::unique_ptr<W> _Nullable p) {
  if (p)
    return 0;
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
