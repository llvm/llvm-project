// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fobjc-arc -verify -fblocks -Wpessimizing-move -Wredundant-move %s

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type &&move(T &&t);
}
}

class MoveOnly {
public:
  MoveOnly() { }
  MoveOnly(MoveOnly &&) = default; // expected-note 2 {{copy constructor is implicitly deleted}}
  MoveOnly &operator=(MoveOnly &&) = default;
  ~MoveOnly();
};

void copyInit() {
  __block MoveOnly temp;
  MoveOnly temp2 = temp; // expected-error {{call to implicitly-deleted copy constructor of 'MoveOnly'}}
  MoveOnly temp3 = std::move(temp); // ok
}

MoveOnly errorOnCopy() {
  __block MoveOnly temp;
  return temp; // expected-error {{call to implicitly-deleted copy constructor of 'MoveOnly'}}
}

MoveOnly dontWarnOnMove() {
  __block MoveOnly temp;
  return std::move(temp); // ok
}

class MoveOnlySub : public MoveOnly {};

MoveOnly dontWarnOnMoveSubclass() {
  __block MoveOnlySub temp;
  return std::move(temp); // ok
}
