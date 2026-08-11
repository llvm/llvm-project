// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++17 -verify=expected %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fsyntax-only -std=c++17 -verify=expected %s -fexperimental-new-constant-interpreter
constexpr void undefined();  // expected-note 4 {{declared here}}
static constexpr int r = [] { // expected-error {{constexpr variable 'r' must be initialized by a constant expression}} \
                                 expected-note {{in call to '[] {}.operator()()'}}}
  undefined();  // expected-note {{undefined function 'undefined' cannot be used in a constant expression}}
  return 0;
 }();

static constexpr auto valid_lambda = [] {};

static constexpr int nested = [] { // expected-error {{constexpr variable 'nested' must be initialized by a constant expression}} \
                                 expected-note {{in call to '[] {}.operator()()'}}}
  valid_lambda();
  auto inner = [] { // expected-note {{in call to '[] {}.operator()()'}}}
	undefined(); // expected-note {{undefined function 'undefined' cannot be used in a constant expression}}
	return 1;
  }();
  return 0;
}();

static constexpr float type_mismatch = [] { // expected-error {{cannot initialize a variable of type 'const float' with an rvalue of type 'void'}}
}();

constexpr auto named_lambda = [] {
  undefined(); // expected-note {{undefined function 'undefined' cannot be used in a constant expression}}
  return 0;
};

static constexpr int named_lambda_result = named_lambda(); // expected-error {{constexpr variable 'named_lambda_result' must be initialized by a constant expression}} \
expected-note {{in call to 'named_lambda.operator()()'}}

static constexpr int undeclared = []{ // expected-error {{constexpr variable 'undeclared' must be initialized by a constant expression}}
  foo();  // expected-error {{use of undeclared identifier 'foo'}}
  return 0;
}();


static constexpr int with_param = [](int x) { // expected-error {{constexpr variable 'with_param' must be initialized by a constant expression}} \
expected-note {{in call to '[](int x) {}.operator()(2)'}}
  undefined(); // expected-note {{undefined function 'undefined' cannot be used in a constant expression}}
  return x;
}(2);
