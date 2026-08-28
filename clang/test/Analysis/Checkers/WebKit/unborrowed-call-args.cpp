// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UnborrowedCallArgsChecker -verify %s

#include "mock-canborrow.h"

void process(char &);
void processConst(const char &);
void takePtr(char *);
void takeView(StringView);
void takeSpan(CharSpan);

void takeContainer(Vector<char> &);
void takeContainerPtr(Vector<char> *);

Vector<char> makeVector();
Vector<char> &getVectorRef();

Vector<char> globalVector;

namespace arg_origins {

void from_local() {
  Vector<char> vec;
  process(vec[0]);
  // expected-warning@-1{{Function argument 'vec[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void from_parameter(Vector<char> &vec) {
  process(vec[0]);
  // expected-warning@-1{{Function argument 'vec[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void from_global() {
  process(globalVector[0]);
  // expected-warning@-1{{Function argument 'globalVector[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void from_static_local() {
  static Vector<char> cache;
  process(cache[0]);
  // expected-warning@-1{{Function argument 'cache[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

class Holder {
public:
  void from_member() {
    process(m_vector[0]);
    // expected-warning@-1{{Function argument 'this->m_vector[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
  }

private:
  Vector<char> m_vector;
};

void from_call_result() {
  process(makeVector()[0]);
  // expected-warning@-1{{Function argument 'makeVector()[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void from_returned_reference() {
  process(getVectorRef()[0]);
  // expected-warning@-1{{Function argument 'getVectorRef()[0]' (to 'process') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void buffer_pointer() {
  Vector<char> vec;
  takePtr(vec.data());
  // expected-warning@-1{{Function argument 'vec.data()' (to 'takePtr') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void const_parameter() {
  Vector<char> vec;
  processConst(vec[0]);
  // expected-warning@-1{{Function argument 'vec[0]' (to 'processConst') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

} // namespace arg_origins

namespace implicit_object_arg {

void mutating_method() {
  Vector<Element> vec;
  vec[0].mutate();
  // expected-warning@-1{{Function argument 'vec[0]' (parameter 'this' to 'Element::mutate') is a loan on CanBorrow type 'Vector<Element>' that is not guarded by a Borrow}}
}

void const_method() {
  Vector<Element> vec;
  vec[0].inspect();
  // expected-warning@-1{{Function argument 'vec[0]' (parameter 'this' to 'Element::inspect') is a loan on CanBorrow type 'Vector<Element>' that is not guarded by a Borrow}}
}

class Holder {
public:
  void from_member() {
    m_elements[0].mutate();
    // expected-warning@-1{{Function argument 'this->m_elements[0]' (parameter 'this' to 'Element::mutate') is a loan on CanBorrow type 'Vector<Element>' that is not guarded by a Borrow}}
  }

private:
  Vector<Element> m_elements;
};

void nested_container() {
  Vector<Vector<char>> outer;
  outer[0].append('x');
  // expected-warning@-1{{Function argument 'outer[0]' (parameter 'this' to 'Vector<char>::append') is a loan on CanBorrow type 'Vector<Vector<char>>' that is not guarded by a Borrow}}
}

} // namespace implicit_object_arg

namespace not_a_loan {

void passing_the_container() {
  Vector<char> vec;
  takeContainer(vec);
}

void passing_the_containers_address() {
  Vector<char> vec;
  takeContainerPtr(&vec);
}

void passing_an_alias() {
  Vector<char> vec;
  Vector<char> &alias = vec;
  takeContainer(alias);
}

void method_on_the_container() {
  Vector<char> vec;
  vec.append('x');
}

void not_borrowable_container() {
  NotBorrowable nb;
  process(nb.at(0));
}

void smart_pointer_accessor(Owner<NotBorrowable> &owner) {
  process(owner->at(0));
}

} // namespace not_a_loan

namespace guarded {

void borrowed_element() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  process(b.get()[0]);
}

void borrowed_buffer() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
  takePtr(b.get().data());
}

void borrowed_implicit_object_arg() {
  Vector<Element> vec;
  Borrow<Vector<Element>> b(vec);
  b.get()[0].mutate();
}

void constructing_the_borrow_is_not_a_loan() {
  Vector<char> vec;
  Borrow<Vector<char>> b(vec);
}

} // namespace guarded

namespace nested_calls {

void annotated_intermediate() {
  Vector<char> vec;
  takeView(makeView(vec.data()));
  // expected-warning@-1{{Function argument 'makeView(vec.data())' (to 'takeView') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
  // expected-warning@-2{{Function argument 'vec.data()' (parameter 'data' to 'makeView') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

void two_loans_one_call() {
  Vector<char> vec;
  pick(vec.data(), vec.data());
  // expected-warning@-1{{Function argument 'vec.data()' (parameter 'a' to 'pick') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
  // expected-warning@-2{{Function argument 'vec.data()' (parameter 'b' to 'pick') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow}}
}

} // namespace nested_calls

namespace known_gaps {

void unannotated_intermediate() {
  Vector<char> vec;
  takeSpan(makeSpanUnannotated(vec));
  // expected-warning@-1{{Function argument 'makeSpanUnannotated(vec)' (to 'takeSpan') is a loan on CanBorrow type 'Vector<char>' that is not guarded by a Borrow [alpha.webkit.UnborrowedCallArgsChecker]}}
}

inline void trivialSink(char &c) {}

void trivial_callee() {
  Vector<char> vec;
  trivialSink(vec[0]);
}

} // namespace known_gaps
