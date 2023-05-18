// RUN: %clang_cc1 -std=c++11 %s -verify

void test_reaching_scope() {
  int local; // expected-note{{declared here}}
  static int local_static;
  (void)[=]() {
    struct InnerLocal {
      void member() {
        (void)[=]() {
          return local + // expected-error{{reference to local variable 'local' declared in enclosing function 'test_reaching_scope'}}
            local_static;
        };
      }
    };
  };
}
