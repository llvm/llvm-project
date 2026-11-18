// RUN: %check_clang_tidy -std=c++20-or-later %s misc-const-correctness %t -- \
// RUN:   -config="{CheckOptions: {\
// RUN:     misc-const-correctness.AnalyzeParameters: true \
// RUN:   }}" -- -fno-delayed-template-parsing

struct Bar {
  void const_method() const;
};

void abbreviated_template_ref_param(auto& b) {
  b.const_method();
}

void abbreviated_template_ptr_param(auto* b) {
  b->const_method();
}

void abbreviated_template_decl(const auto& b);

void plain_ref_param(Bar& b) {
  // CHECK-MESSAGES: [[@LINE-1]]:22: warning: variable 'b' of type 'Bar &' can be declared 'const'
  // CHECK-FIXES: void plain_ref_param(Bar const& b) {
  b.const_method();
}

void generic_lambda_param() {
  auto l = [](auto& b) { b.const_method(); };
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: variable 'l' of type '(lambda at {{.*}})' can be declared 'const'
  // CHECK-FIXES: auto const l = [](auto& b) { b.const_method(); };
  Bar bar;
  l(bar);
}
