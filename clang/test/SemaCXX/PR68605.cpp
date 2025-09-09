// RUN: %clang_cc1 -verify -fsyntax-only -std=c++20 -Wshadow %s

// Test for issue #68605: False positive warning with `-Wshadow` when using 
// structured binding and lambda capture.
// 
// The issue is that structured bindings should behave consistently with 
// regular variables when used in lambda captures - no shadow warning should
// be emitted when a lambda capture variable has the same name as the captured
// structured binding, just like with regular parameters.

namespace std {
  template<typename T> T&& move(T&& t) { return static_cast<T&&>(t); }
}

namespace issue_68605 {

// Simple pair-like struct for testing
struct Pair {
  int first;
  int second;
  Pair(int f, int s) : first(f), second(s) {}
};

// Test case 1: Regular parameter - should NOT produce warning (baseline)
void foo1(Pair val) {
  [val = std::move(val)](){}(); // No warning expected
}

// Test case 2: Structured binding - should NOT produce warning
void foo2(Pair val) {
  auto [a,b] = val;
  [a = std::move(a)](){}(); // No warning - consistent with regular parameter behavior
}

// Test case 3: More complex example with multiple captures
void foo3() {
  Pair data{42, 100};
  auto [id, value] = data;
  
  // Both of these should NOT produce warnings
  auto lambda1 = [id = id](){ return id; }; // No warning
  auto lambda2 = [value = value](){ return value; }; // No warning
}

// Test case 4: Mixed scenario with regular var and structured binding
void foo4() {
  int regular_var = 10;
  Pair pair_data{1, 2};
  auto [x, y] = pair_data;
  
  // Regular variable capture - no warning expected (current behavior)
  auto lambda1 = [regular_var = regular_var](){};
  
  // Structured binding captures - should be consistent
  auto lambda2 = [x = x](){}; // No warning - consistent behavior
  auto lambda3 = [y = y](){}; // No warning - consistent behavior
}

// Test case 5: Ensure we don't break existing shadow detection for actual shadowing
void foo5() {
  int outer = 5; // expected-note {{previous declaration is here}}
  auto [a, b] = Pair{1, 2}; // expected-note {{previous declaration is here}}
  
  // This SHOULD still warn - it's actual shadowing within the lambda body
  auto lambda = [outer, a](){ // expected-note {{variable 'outer' is explicitly captured here}}
    int outer = 10; // expected-warning {{declaration shadows a local variable}}
    int a = 20;     // expected-warning {{declaration shadows a structured binding}}
  };
}

} // namespace issue_68605