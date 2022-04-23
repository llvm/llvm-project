// RUN: %clang_cc1 -fsyntax-only -verify -Wall -std=c++11 %s -Wno-unused-value

namespace std {

template <typename T>
void dummy(T &&) {}
template <typename T>
T &&move(T &&x) { return x; }
template <typename T, typename U>
void move(T &&, U &&) {}

inline namespace __1 {
template <typename T>
T &forward(T &x) { return x; }
} // namespace __1

struct foo {};

} // namespace std

namespace global {

using namespace std;

void f() {
  int i = 0;
  std::move(i);
  move(i);   // expected-warning{{unqualified call to 'std::move'}}
  (move)(i); // expected-warning{{unqualified call to 'std::move'}}
  std::dummy(1);
  dummy(1);
  std::move(1, 2);
  move(1, 2);
  forward<int>(i); // expected-warning{{unqualified call to 'std::forward'}}
  std::forward<int>(i);
}

template <typename T>
void g(T &&foo) {
  std::move(foo);
  move(foo); // expected-warning{{unqualified call to 'std::move}}

  std::forward<decltype(foo)>(foo);
  forward<decltype(foo)>(foo); // expected-warning{{unqualified call to 'std::forward}}
  move(1, 2);
  dummy(foo);
}

void call() {
  g(0); //expected-note {{here}}
}

} // namespace global

namespace named {

using std::forward;
using std::move;

void f() {
  int i = 0;
  move(i); // expected-warning{{unqualified call to 'std::move}}
  move(1, 2);
  forward<int>(i); // expected-warning{{unqualified call to 'std::forward}}
}

template <typename T>
void g(T &&foo) {
  move(foo);                     // expected-warning{{unqualified call to 'std::move}}
  forward<decltype(foo)>(foo);   // expected-warning{{unqualified call to 'std::forward}}
  (forward<decltype(foo)>)(foo); // expected-warning{{unqualified call to 'std::forward}}
  move(1, 2);
}

void call() {
  g(0); //expected-note {{here}}
}

} // namespace named

namespace overload {
using namespace std;
template <typename T>
int move(T &&);
void f() {
  int i = 0;
  move(i);
}
} // namespace overload

namespace adl {
void f() {
  move(std::foo{}); // expected-warning{{unqualified call to 'std::move}}
}

} // namespace adl

namespace std {

void f() {
  int i = 0;
  move(i);         // expected-warning{{unqualified call to 'std::move}}
  forward<int>(i); // expected-warning{{unqualified call to 'std::forward}}
}

} // namespace std

namespace test_alias {
namespace alias = std;
using namespace alias;
void f() {
  int i = 0;
  move(i); // expected-warning{{unqualified call to 'std::move}}
  move(1, 2);
  forward<int>(i); // expected-warning{{unqualified call to 'std::forward}}
}

} // namespace test_alias
