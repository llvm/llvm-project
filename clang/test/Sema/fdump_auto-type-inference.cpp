// RUN: %clang_cc1 -std=c++14 -mllvm -fdump-auto-type-inference %s

void testAuto() {
  // Test auto variables
  auto x = 5;
  auto y = 3.14;

  // Test auto return type of a lambda function
  auto add = [](int a, double b) -> double {
    return a + b;
  };

  // Expected remarks based on the compiler output
  // expected-remark@-5 {{type of 'x' deduced as 'int'}}
  // expected-remark@-4 {{type of 'y' deduced as 'double'}}
  // expected-remark@-3 {{type of 'add' deduced as '(lambda at %s'}}
}

int main() {
    testAuto();
    // Testing auto variables
    auto x = 5;            // int
    auto y = 3.14;         // double
    auto z = 'c';          // char

    // expected-remark@+1{{type of 'x' deduced as 'int'}}
    // expected-remark@+1{{type of 'y' deduced as 'double'}}
    // expected-remark@+1{{type of 'z' deduced as 'char'}}

    // Testing auto return type of a function
    auto add = [](auto a, auto b) {
        return a + b;
    };

    auto divide = [](auto a, auto b) -> decltype(a / b) {
        return a / b;
    };

    struct Foo {
        auto getVal() const {
            return val;
        }
        int val = 42;
    };

    // expected-remark@+2{{type of 'add' deduced as '(lambda}}
    // expected-remark@+1{{type of 'divide' deduced as '(lambda}}
    // expected-remark@+1{{function return type of 'getVal' deduced as 'int'}}

    return 0;
}