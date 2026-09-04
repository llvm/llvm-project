// RUN: %clang_cc1 -verify -fsyntax-only -std=c++20 -Wshadow %s
// RUN: %clang_cc1 -verify=all -fsyntax-only -std=c++20 -Wshadow-all %s

// Test for issue #68605: Inconsistent shadow warnings for lambda capture of structured bindings.
// 
// The issue was that structured binding lambda captures were incorrectly classified
// as regular shadow warnings (shown with -Wshadow) while regular parameter captures 
// were classified as uncaptured-local warnings (shown only with -Wshadow-all).
//
// This test validates that both VarDecl and BindingDecl lambda captures now 
// behave consistently: no warnings with -Wshadow, but uncaptured-local warnings 
// with -Wshadow-all.

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

// Test case 1: Regular parameter - consistent behavior
void foo1(Pair val) { // all-note {{previous declaration is here}}
  [val = std::move(val)](){}(); // all-warning {{declaration shadows a local variable}}
}

// Test case 2: Structured binding - now consistent with regular parameter
void foo2(Pair val) {
  auto [a,b] = val; // all-note {{previous declaration is here}}
  [a = std::move(a)](){}(); // all-warning {{declaration shadows a structured binding}}
}

// Test case 3: Multiple captures showing consistent behavior
void foo3() {
  Pair data{42, 100};
  auto [id, value] = data; // all-note 2{{previous declaration is here}}
  
  // Both show consistent uncaptured-local warnings with -Wshadow-all
  auto lambda1 = [id = id](){ return id; }; // all-warning {{declaration shadows a structured binding}}
  auto lambda2 = [value = value](){ return value; }; // all-warning {{declaration shadows a structured binding}}
}

// Test case 4: Mixed scenario showing consistent behavior
void foo4() {
  int regular_var = 10; // all-note {{previous declaration is here}}
  Pair pair_data{1, 2};
  auto [x, y] = pair_data; // all-note 2{{previous declaration is here}}
  
  // All captures now show consistent uncaptured-local warnings with -Wshadow-all
  auto lambda1 = [regular_var = regular_var](){}; // all-warning {{declaration shadows a local variable}}
  auto lambda2 = [x = x](){}; // all-warning {{declaration shadows a structured binding}}
  auto lambda3 = [y = y](){}; // all-warning {{declaration shadows a structured binding}}
}

// Test case 5: Ensure we don't break existing shadow detection for actual shadowing
void foo5() {
  int outer = 5; // expected-note {{previous declaration is here}} all-note {{previous declaration is here}}
  auto [a, b] = Pair{1, 2}; // expected-note {{previous declaration is here}} all-note {{previous declaration is here}}
  
  // This SHOULD still warn - it's actual shadowing within the lambda body
  auto lambda = [outer, a](){ // expected-note {{variable 'outer' is explicitly captured here}} all-note {{variable 'outer' is explicitly captured here}} expected-note {{variable 'a' is explicitly captured here}} all-note {{variable 'a' is explicitly captured here}}
    int outer = 10; // expected-warning {{declaration shadows a local variable}} all-warning {{declaration shadows a local variable}}
    int a = 20;     // expected-warning {{declaration shadows a structured binding}} all-warning {{declaration shadows a structured binding}}
  };
}

} // namespace issue_68605
