// RUN: %clang_cc1 %s -triple x86_64-windows -fdiagnostics-format msvc -fno-rtti-data -fsyntax-only -verify

namespace std {
struct type_info {};
} // namespace std
class B {
public:
  virtual ~B() = default;
};

class D1 : public B {
public:
  ~D1() = default;
};

void f() {
  B* b = new D1();
  auto d = dynamic_cast<D1 *>(b); // expected-warning{{dynamic_cast will not work since RTTI data is disabled by /GR-}}
  void* v = dynamic_cast<void *>(b); // expected-warning{{dynamic_cast will not work since RTTI data is disabled by /GR-}}
  (void)typeid(int);              // expected-warning{{typeid will not work since RTTI data is disabled by /GR-}}
}
