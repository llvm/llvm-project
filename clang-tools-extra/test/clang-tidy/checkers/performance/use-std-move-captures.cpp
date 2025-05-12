// RUN: %check_clang_tidy %s performance-use-std-move %t -std=c++14,c++17,c++20,c++23
// RUN: %clang -std=c++14 -pedantic-errors -fsyntax-only -nostdinc++ -isystem %clang_tidy_headers/std %t.cpp

struct Value {
  Value();
  Value(const Value &);
  Value(Value &&) noexcept;
};
void initCapture(Value source) {
  auto closure = [copy = source] {};
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: auto closure = [copy = source] {};
}
void renamedCaptureWithUse(Value source) {
  auto closure = [copy = source] {};
  Value target(source);
  // CHECK-MESSAGES: :[[@LINE-1]]:{{[0-9]+}}: warning: 'source' could be moved here
  // CHECK-FIXES: Value target(std::move(source));
}
void movedCapture(Value source) {
  auto closure = [copy = static_cast<Value &&>(source)] {};
}

void genericLambda() {
  auto copy = [](auto source) {
    Value target(source);
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'source' could be moved here [performance-use-std-move]
    // CHECK-FIXES: Value target(source);
  };
  copy(Value{});
}
