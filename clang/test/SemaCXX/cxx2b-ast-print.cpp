// RUN: %clang_cc1 -std=c++2b -fsyntax-only -ast-print %s | FileCheck %s

void test_auto_expr(long long y, auto &&z) {
  int x[] = {3, 4};

  // CHECK{LITERAL}: (int)(x[1])
  void(auto(x[1]));
  // CHECK{LITERAL}: (int){x[1]}
  void(auto{x[1]});

  // CHECK{LITERAL}: (int *)(x)
  void(auto(x));
  // CHECK{LITERAL}: (int *){x}
  void(auto{x});

  // CHECK{LITERAL}: auto(z)
  void(auto(z));
  // CHECK{LITERAL}: auto({z})
  void(auto{z}); // T({z}) is legal unless T = auto

  // CHECK{LITERAL}: new int *(x)
  void(new auto(x));
  // CHECK{LITERAL}: new int *{x}
  void(new auto{x});

  // CHECK{LITERAL}: new long long(y)
  void(new decltype(auto)(y));
  // CHECK{LITERAL}: new long long{y}
  void(new decltype(auto){y});
}
