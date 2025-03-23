// RUN: %clang_analyze_cc1 -std=c++23 %s -verify -analyzer-checker=alpha.cplusplus.InvalidatedIterator -analyzer-config aggressive-binary-operation-simplification=true

// expected-no-diagnostics

class ExplicitThis {
  int f = 0;
public:
  ExplicitThis();
  ExplicitThis(ExplicitThis& other);

  ExplicitThis& operator=(this ExplicitThis& self, ExplicitThis const& other) { // no crash
    self.f = other.f;
    return self;
  }

  ~ExplicitThis();
};

void func(ExplicitThis& obj1) {
    obj1 = obj1;
}
