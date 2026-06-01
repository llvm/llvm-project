// Known smart-pointer nullability modeling gaps.
//
// These are intentional expected failures so missing smart-pointer transfer
// behavior stays visible without breaking the main passing suites.
//
// XFAIL: *
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 %s -verify

#pragma clang assume_nonnull begin

namespace std {

template <typename T>
struct unique_ptr {
  T *ptr;
  unique_ptr(T *p) : ptr(p) {}
  T *operator->() { return ptr; }
};

} // namespace std

struct Node {
  int value;
};

Node * _Nonnull getSafeNode();
Node * _Nullable getNullableNode();

// Converting a nullable raw pointer into a _Nonnull unique_ptr should be
// rejected. nullable-clang currently does not model this constructor path.
void xfail_unique_ptr_ctor_from_nullable_raw() {
  _Nonnull std::unique_ptr<Node> p(getNullableNode()); // expected-warning{{implicit conversion from nullable pointer}}
}

namespace absl {

template <typename T>
std::unique_ptr<T> WrapUnique(T *p) {
  return std::unique_ptr<T>(p);
}

} // namespace absl

// Helper wrappers like WrapUnique should preserve the underlying raw pointer's
// nullability. nullable-clang currently treats the result as safe here.
void xfail_wrapunique_from_nullable_raw() {
  absl::WrapUnique(getNullableNode())->value = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// Safe input should remain safe once this modeling exists.
void xfail_wrapunique_from_nonnull_raw() {
  absl::WrapUnique(getSafeNode())->value = 1; // no warning
}

#pragma clang assume_nonnull end
