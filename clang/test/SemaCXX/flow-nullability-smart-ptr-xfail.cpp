// Known smart-pointer nullability modeling gaps.
//
// Converted from an unconditional expected-failure to assertions of CURRENT (incomplete) behavior, so
// the file PASSES today, the already-correct case stays protected from
// regression, and each missing transfer is documented with a FIXME. See
// flow-nullability-checker-gaps.cpp for the same pattern.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++20 -Wno-unused-value %s -verify

// All cases below currently emit nothing: the gap cases are MISSED warnings
// (marked FIXME) and the one already-correct case correctly stays silent. If
// any starts firing, this directive fails and flags the case for conversion.
// expected-no-diagnostics

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
// rejected. nullable-clang currently does not model this constructor path, so
// no diagnostic is produced.
void xfail_unique_ptr_ctor_from_nullable_raw() {
  // FIXME: should warn{{implicit conversion from nullable pointer}} once the
  // unique_ptr(T*) constructor propagates the raw pointer's nullability into
  // the _Nonnull smart-pointer target. Currently no warning fires.
  _Nonnull std::unique_ptr<Node> p(getNullableNode()); // no warning (gap)
}

namespace absl {

template <typename T>
std::unique_ptr<T> WrapUnique(T *p) {
  return std::unique_ptr<T>(p);
}

} // namespace absl

// Helper wrappers like WrapUnique should preserve the underlying raw pointer's
// nullability. nullable-clang currently treats the result as safe, so the
// nullable dereference below is NOT flagged.
void xfail_wrapunique_from_nullable_raw() {
  // FIXME: should warn{{dereference of nullable pointer}} once WrapUnique
  // forwards the argument's nullability through to the returned smart pointer.
  // Currently the result is treated as nonnull, so no warning fires.
  absl::WrapUnique(getNullableNode())->value = 1; // no warning (gap)
}

// Safe input must remain safe. This case already behaves correctly today;
// asserting it guards against a future over-eager false positive.
void xfail_wrapunique_from_nonnull_raw() {
  absl::WrapUnique(getSafeNode())->value = 1; // no warning
}

#pragma clang assume_nonnull end
