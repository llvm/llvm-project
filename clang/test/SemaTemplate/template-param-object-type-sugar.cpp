// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// A template parameter object is shared by every use of the same value, so the
// type it was created with carries the spelling of whichever use came first.
// Diagnostics print its canonical type rather than that unrelated spelling.

namespace lib {
struct metre {};
inline constexpr metre m;
template <typename U> struct kilo_ {};
template <typename T> struct remove_ref { using type = T; };
template <auto U> constexpr kilo_<typename remove_ref<decltype(U)>::type> kilo;
template <auto V> struct quantity {}; // #quantity
template <auto V> constexpr quantity<V> make() { return {}; }
} // namespace lib

void take_int(int); // #take_int
void take_metre(lib::quantity<lib::m>); // #take_metre

void test() {
  take_int(lib::make<lib::kilo<lib::m>>());
  // expected-error@-1 {{no matching function for call to 'take_int'}}
  // expected-note@#take_int {{candidate function not viable: no known conversion from 'quantity<lib::kilo_<lib::metre>{}>' to 'int' for 1st argument}}

  take_metre(lib::make<lib::kilo<lib::m>>());
  // expected-error@-1 {{no matching function for call to 'take_metre'}}
  // expected-note@#take_metre {{candidate function not viable: no known conversion from 'quantity<lib::kilo_<lib::metre>{}>' to 'quantity<lib::metre{}>' for 1st argument}}
}
