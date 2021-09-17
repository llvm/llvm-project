// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -emit-llvm-only %s
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fdelayed-template-parsing %s -DDELAYED_TEMPLATE_PARSING
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fms-extensions %s -DMS_EXTENSIONS
// RUN: %clang_cc1 -std=c++1z -verify -fsyntax-only -fblocks -fdelayed-template-parsing -fms-extensions %s -DMS_EXTENSIONS -DDELAYED_TEMPLATE_PARSING

template <class, class>
constexpr bool is_same = false;
template <class T>
constexpr bool is_same<T, T> = true;

namespace test_star_this {
namespace ns1 {
class A {
  int x = 345;
  auto foo() {
    (void)[ *this, this ]{}; //expected-error{{'this' can appear only once}}
    (void)[this] { ++x; };
    (void)[*this] { ++x; }; //expected-error{{read-only variable}}
    (void)[*this]() mutable { ++x; };
    (void)[=] { return x; };
    (void)[&, this ] { return x; };
    (void)[ =, *this ] { return x; };
    (void)[&, *this ] { return x; };
  }
};
} // namespace ns1

namespace ns2 {
class B {
  B(const B &) = delete; //expected-note{{deleted here}}
  int *x = (int *)456;
  void foo() {
    (void)[this] { return x; };
    (void)[*this] { return x; }; //expected-error{{call to deleted}}
  }
};
} // namespace ns2

namespace ns3 {
class B {
  B(const B &) = delete; //expected-note2{{deleted here}}

  int *x = (int *)456;

public:
  template <class T = int>
  void foo() {
    (void)[this] { return x; };
    (void)[*this] { return x; }; //expected-error2{{call to deleted}}
  }

  B() = default;
} b;
B *c = (b.foo(), nullptr); //expected-note{{in instantiation}}
} // namespace ns3

namespace ns4 {
template <class U>
class B {
  B(const B &) = delete; //expected-note{{deleted here}}
  double d = 3.14;

public:
  template <class T = int>
  auto foo() {
    const auto &L = [*this](auto a) mutable { //expected-error{{call to deleted}}
      d += a;
      return [this](auto b) { return d += b; };
    };
  }

  B() = default;
};
void main() {
  B<int *> b;
  b.foo(); //expected-note{{in instantiation}}
} // end main
} // namespace ns4

namespace ns5 {

struct X {
  double d = 3.14;
  X(const volatile X &);
  void foo() {
  }

  void foo() const { //expected-note{{const}}

    auto L = [*this]() mutable {
      static_assert(is_same<decltype(this), X *>);
      ++d;
      auto M = [this] {
        static_assert(is_same<decltype(this), X *>);
        ++d;
        auto N = [] {
          static_assert(is_same<decltype(this), X *>);
        };
      };
    };

    auto L1 = [*this] {
      static_assert(is_same<decltype(this), const X *>);
      auto M = [this]() mutable {
        static_assert(is_same<decltype(this), const X *>);
        auto N = [] {
          static_assert(is_same<decltype(this), const X *>);
        };
      };
      auto M2 = [*this]() mutable {
        static_assert(is_same<decltype(this), X *>);
        auto N = [] {
          static_assert(is_same<decltype(this), X *>);
        };
      };
    };

    auto GL1 = [*this](auto a) {
      static_assert(is_same<decltype(this), const X *>);
      auto M = [this](auto b) mutable {
        static_assert(is_same<decltype(this), const X *>);
        auto N = [](auto c) {
          static_assert(is_same<decltype(this), const X *>);
        };
        return N;
      };

      auto M2 = [*this](auto a) mutable {
        static_assert(is_same<decltype(this), X *>);
        auto N = [](auto b) {
          static_assert(is_same<decltype(this), X *>);
        };
        return N;
      };
      return [=](auto a) mutable { M(a)(a); M2(a)(a); };
    };

    GL1("abc")
    ("abc");

    auto L2 = [this]() mutable {
      static_assert(is_same<decltype(this), const X *>);
      ++d; //expected-error{{cannot assign}}
    };
    auto GL = [*this](auto a) mutable {
      static_assert(is_same<decltype(this), X *>);
      ++d;
      auto M = [this](auto b) {
        static_assert(is_same<decltype(this), X *>);
        ++d;
        auto N = [](auto c) {
          static_assert(is_same<decltype(this), X *>);
        };
        N(3.14);
      };
      M("abc");
    };
    GL(3.14);
  }
  void foo() volatile const {
    auto L = [this]() {
      static_assert(is_same<decltype(this), const volatile X *>);
      auto M = [*this]() mutable {
        static_assert(is_same<decltype(this), X *>);
        auto N = [this] {
          static_assert(is_same<decltype(this), X *>);
          auto M = [] {
            static_assert(is_same<decltype(this), X *>);
          };
        };
        auto N2 = [*this] {
          static_assert(is_same<decltype(this), const X *>);
        };
      };
      auto M2 = [*this]() {
        static_assert(is_same<decltype(this), const X *>);
        auto N = [this] {
          static_assert(is_same<decltype(this), const X *>);
        };
      };
    };
  }
};

} // namespace ns5
namespace ns6 {
struct X {
  double d;
  auto foo() const {
    auto L = [*this]() mutable {
      auto M = [=](auto a) {
        auto N = [this] {
          ++d;
          static_assert(is_same<decltype(this), X *>);
          auto O = [*this] {
            static_assert(is_same<decltype(this), const X *>);
          };
        };
        N();
        static_assert(is_same<decltype(this), X *>);
      };
      return M;
    };
    return L;
  }
};

int main() {
  auto L = X{}.foo();
  auto M = L();
  M(3.14);
}
} // namespace ns6
namespace ns7 {

struct X {
  double d;
  X();
  X(const X &);
  X(X &) = delete;
  auto foo() const {
    //OK - the object used to initialize our capture is a const object and so prefers the non-deleted ctor.
    const auto &&L = [*this]{};
  }
};
int main() {
  X x;
  x.foo();
}
} // namespace ns7

} // namespace test_star_this

namespace PR32831 {
// https://bugs.llvm.org/show_bug.cgi?id=32831
namespace ns1 {
template <typename Func>
void fun_template(Func func) {
  (void)[&]() {
    func(0);
  };
}

class A {
  void member_foo() {
    (void)[this] {
      (void)[this] {
        fun_template(
            [this](auto X) {
              auto L = [this](auto Y) { member_foo(); };
              L(5);
            });
        fun_template(
            [this](auto) { member_foo(); });
      };
    };
  }
};
} // namespace ns1

namespace ns2 {

struct B {
  int data = 0;
  template <class F>
  void mem2(F f) {
    (void)[&](auto f) {
      (void)[&] { f(this->data); };
    }
    (f);
  }
};

class A {
  void member_foo() {
    (void)[this] {
      (void)[this] {
        B{}.mem2(
            [this](auto X) {
              auto L = [this](auto Y) { member_foo(); };
              L(5);
            });
        B{}.mem2(
            [this](auto) { member_foo(); });
      };
    };
  }
  int data = 0;
  auto m2() {
    return [this] { return [] () -> decltype(data){ return 0; }; };
  }
  auto m3() {
    return [] { return [] () -> decltype(data){ return 0; }; };
  }
};

} // namespace ns2

} // namespace PR32831

namespace PR45881 {
struct A {
    void f();
};
int id(A*);
void A::f() {
    auto z = [*this](auto z2, decltype(z2(this)) z3){};
    z(id,3);
}
} // namespace PR45881
