// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir -verify %s

struct Empty {};

// Empty for layout, because its own member is an empty record, but not empty
// for the ABI, because a C++ record member is data without the attribute.
struct EmptyForLayoutOnly { Empty e; };

struct Wrapper {
  // expected-error@+1 {{ClangIR code gen Not Yet Implemented: [[no_unique_address]] field that is empty for layout but holds data for the ABI}}
  [[no_unique_address]] EmptyForLayoutOnly e;
};

Wrapper w;
