// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std
#include <memory>

namespace std {
template< class T >
class enable_shared_from_this {
public:
  std::shared_ptr<T> shared_from_this() { return std::shared_ptr<T>(); }
  std::shared_ptr<const T> shared_from_this() const { return std::shared_ptr<const T>(); }
};

template< class T, class U >
std::shared_ptr<T>
    dynamic_pointer_cast( const std::shared_ptr<U>& ) noexcept {
      return std::shared_ptr<T>();
    }
}

// All cases were taken from https://cmu-sei.github.io/secure-coding-standards/sei-cert-cpp-coding-standard/rules/memory-management-mem/mem56-cpp/

// ╔══════════════════════════════════════════════════════════════╗
// ║                  Noncompliant Code Example                   ║
// ╚══════════════════════════════════════════════════════════════╝
void f0() {
  int *i = new int;
  std::shared_ptr<int> p1(i);
  std::shared_ptr<int> p2(i);
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: passing a raw pointer 'int*' to 'std::shared_ptr<int>' constructor may cause double deletion
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                    Compliant Solution                        ║
// ╚══════════════════════════════════════════════════════════════╝
void f1() {
  std::shared_ptr<int> p1 = std::make_shared<int>();
  std::shared_ptr<int> p2(p1);
}


struct B {
  virtual ~B() = default; // Polymorphic object
  // ...
};
struct D : B {};

void g(std::shared_ptr<D> derived);

// ╔══════════════════════════════════════════════════════════════╗
// ║                  Noncompliant Code Example                   ║
// ╚══════════════════════════════════════════════════════════════╝
void f2() {
  std::shared_ptr<B> poly(new D);
  // ...
  g(std::shared_ptr<D>(dynamic_cast<D *>(poly.get())));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'D*' to 'std::shared_ptr<D>' constructor may cause double deletion
  // Any use of poly will now result in accessing freed memory.
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                    Compliant Solution                        ║
// ╚══════════════════════════════════════════════════════════════╝
void f3() {
  std::shared_ptr<B> poly(new D);
  // ...
  g(std::dynamic_pointer_cast<D, B>(poly));
  // poly is still referring to a valid pointer value.
}


// ╔══════════════════════════════════════════════════════════════╗
// ║                  Noncompliant Code Example                   ║
// ╚══════════════════════════════════════════════════════════════╝
struct S1 {
  std::shared_ptr<S1> g() { return std::shared_ptr<S1>(this); }
  // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: passing a raw pointer 'S1*' to 'std::shared_ptr<S1>' constructor may cause double deletion
};

void f4() {
  std::shared_ptr<S1> s1 = std::make_shared<S1>();
  // ...
  std::shared_ptr<S1> s2 = s1->g();
}

// ╔══════════════════════════════════════════════════════════════╗
// ║                    Compliant Solution                        ║
// ╚══════════════════════════════════════════════════════════════╝
struct S2 : std::enable_shared_from_this<S2> {
  std::shared_ptr<S2> g() { return shared_from_this(); }    
};

void f5() {
  std::shared_ptr<S2> s1 = std::make_shared<S2>();
  std::shared_ptr<S2> s2 = s1->g();
}
