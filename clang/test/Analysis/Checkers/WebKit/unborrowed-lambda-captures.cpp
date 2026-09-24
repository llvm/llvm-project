// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnborrowedLambdaCapturesChecker -verify %s

#include "mock-canborrow.h"

void someFunction();
void use(char *);

namespace loan_shapes {

void init_capture_computing_a_loan() {
  Vector<char> vec;
  callEscaping([q = vec.data()] { someFunction(); });
  // expected-warning@-1{{Captured variable 'q' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void reference_to_an_element() {
  Vector<char> vec;
  callEscaping([&c = vec[0]] { someFunction(); });
  // expected-warning@-1{{Captured variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void loan_on_a_parameter(Vector<char> &parameter) {
  callEscaping([q = parameter.data()] { someFunction(); });
  // expected-warning@-1{{Captured variable 'q' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void loan_on_a_nested_container() {
  Vector<Vector<char>> outer;
  callEscaping([&inner = outer[0]] { someFunction(); });
  // expected-warning@-1{{Captured variable 'inner' is a loan on CanBorrow type 'Vector<Vector<char>>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

} // namespace loan_shapes

namespace borrow_does_not_travel {

void loan_through_a_borrow() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  callEscaping([q = b.get().data()] { someFunction(); });
  // expected-warning@-1{{Captured variable 'q' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void element_through_a_borrow() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  callEscaping([&c = b.get()[0]] { someFunction(); });
  // expected-warning@-1{{Captured variable 'c' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void loan_through_a_borrow_temporary() {
  Vector<char> vec;
  callEscaping([q = borrow(vec).get().data()] { someFunction(); });
  // expected-warning@-1{{Captured variable 'q' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void capture_of_the_borrow_by_reference() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  callEscaping([&b] { someFunction(); });
  // expected-warning@-1{{Captured variable 'b' is a Borrow that does not travel with the lambda [alpha.webkit.UnborrowedLambdaCapturesChecker]}}
}

void borrow_inside_the_body() {
  Vector<char> vec;
  callEscaping([&vec] {
    Borrow<Vector<char>> b(vec);
    use(b.get().data());
  });
}

} // namespace borrow_does_not_travel

namespace noescape_borrows_work {

void loan_guarded_for_the_whole_call() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  callNoEscape([q = b.get().data()] { use(q); });
}

void borrow_captured_by_reference() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  callNoEscape([&b] { use(b.get().data()); });
}

void borrow_inside_the_body() {
  Vector<char> vec;
  callNoEscape([&vec] {
    Borrow<Vector<char>> b(vec);
    use(b.get().data());
  });
}

void nested_container_guarded_at_the_inner() {
  Vector<Vector<char>> outer;
  Borrow<Vector<char>> b(outer[0]);
  callNoEscape([q = b.get().data()] { use(q); });
}

} // namespace noescape_borrows_work

namespace not_a_loan {

void reference_to_the_container() {
  Vector<char> vec;
  callEscaping([&vec] { vec.append('x'); });
}

void copy_of_an_element() {
  Vector<char> vec;
  callEscaping([c = vec[0]] { someFunction(); });
}

void copy_through_a_reference_variable() {
  Vector<char> vec;
  Vector<char> &r = vec;
  callEscaping([r] { someFunction(); });
}

void view_on_a_non_container() {
  NotBorrowable buffer;
  callEscaping([&c = buffer.at(0)] { someFunction(); });
}

void unrelated_pointer_parameter(char *unrelated) {
  callEscaping([unrelated] { someFunction(); });
}

class Holder {
public:
  void capturesThis() { callEscaping([this] { someFunction(); }); }

private:
  Vector<char> m_vec;
};

// A pointer to the CanBorrow object itself is not a loan on its interior;
// -Wlifetime-safety covers it.
class Container : public CanBorrow {
public:
  void capturesThis() { callEscaping([this] { someFunction(); }); }
  void capturesThisImplicitly() {
    callEscaping([&] { someFunction(); m_size = 0; });
  }

private:
  unsigned m_size { 0 };
};

extern const Vector<char> globalConstBuffer;

void loan_on_const_global() {
  callEscaping([p = globalConstBuffer.data()] { someFunction(); });
}

} // namespace not_a_loan

namespace vla_type_capture {

// Naming a variable-length array type inside a lambda adds an implicit capture
// of the array bound, which has no variable and no capture initializer.
void alongside_a_loan(unsigned n) {
  Vector<char> vec;
  typedef char VLA[n];
  callEscaping([q = vec.data()] { someFunction(); use(q); (void)sizeof(VLA); });
  // expected-warning@-1{{Captured variable 'q' is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void alone(unsigned n) {
  typedef char VLA[n];
  callEscaping([] { someFunction(); (void)sizeof(VLA); });
}

} // namespace vla_type_capture

namespace known_gaps {

void noescape_parameter() {
  Vector<char> vec;
  callNoEscape([q = vec.data()] { someFunction(); });
}

void trivial_body() {
  Vector<char> vec;
  callEscaping([q = vec.data()] {});
}

void loan_through_a_named_local() {
  Vector<char> vec;
  char *p = vec.data();
  callNoEscape([p] { someFunction(); });
  callEscaping([p] { someFunction(); });
}

} // namespace known_gaps
