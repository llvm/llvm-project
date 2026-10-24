// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify -DFRIEND_FIRST %s
// expected-no-diagnostics

// This test shows that it's ok to cache concept-id satisfaction, because
// constraint satisfaction is independent of the context it's checked in.

template <class T> concept HasPrivate = requires(T t) { t.private_field_; };
template <class T> concept Outer = HasPrivate<T> && sizeof(T) > 0;

struct Private {
  friend void friend_fn();
private:
  int private_field_;
};
struct Public { int private_field_; };

#ifdef FRIEND_FIRST
void friend_fn() {
  static_assert(!HasPrivate<Private>);
  static_assert(!Outer<Private>);
  static_assert(Outer<Public>);
}
void non_friend() {
  static_assert(!HasPrivate<Private>);
  static_assert(!Outer<Private>);
  static_assert(Outer<Public>);
}
#else
void non_friend() {
  static_assert(!HasPrivate<Private>);
  static_assert(!Outer<Private>);
  static_assert(Outer<Public>);
}
void friend_fn() {
  static_assert(!HasPrivate<Private>);
  static_assert(!Outer<Private>);
  static_assert(Outer<Public>);
}
#endif
