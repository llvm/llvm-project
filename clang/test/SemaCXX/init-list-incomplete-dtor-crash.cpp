// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

// Regression test for https://github.com/llvm/llvm-project/issues/140685
//
// List-initialization of an array whose element type is an incomplete
// (forward-declared) class triggered destructor lookup on the incomplete
// type, hitting an assertion in Sema::LookupSpecialMember.

namespace gh140685 {
struct MoveOnly; // expected-note {{forward declaration of 'gh140685::MoveOnly'}}

void test() {
  MoveOnly(&&list)[1] = {};
  // expected-error@-1 {{initialization of incomplete type 'MoveOnly'}}
  // expected-note@-2 {{in implicit initialization of array element 0 with omitted initializer}}
  // expected-note@-3 {{in initialization of temporary of type 'MoveOnly[1]' created to list-initialize this reference}}
}
} // namespace gh140685
