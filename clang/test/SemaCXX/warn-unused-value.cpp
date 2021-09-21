// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value -std=c++11 %s

// PR4806
namespace test0 {
  class Box {
  public:
    int i;
    volatile int j;
  };

  void doit() {
    // pointer to volatile has side effect (thus no warning)
    Box* box = new Box;
    box->i; // expected-warning {{expression result unused}}
    box->j;
#if __cplusplus <= 199711L
    // expected-warning@-2 {{expression result unused}}
#endif
  }
}

namespace test1 {
struct Foo {
  int i;
  bool operator==(const Foo& rhs) {
    return i == rhs.i;
  }
};

#define NOP(x) (x)
void b(Foo f1, Foo f2) {
  NOP(f1 == f2);  // expected-warning {{expression result unused}}
}
#undef NOP
}

namespace test2 {
  extern "C++" {
    namespace std {
      template<typename T> struct basic_string {
        struct X {};
        void method() const {
         X* x;
         &x[0];  // expected-warning {{expression result unused}}
        }
      };
      typedef basic_string<char> string;
      void func(const std::string& str) {
        str.method();  // expected-note {{in instantiation of member function}}
      }
    }
  }
}

namespace test3 {
struct Used {
  Used();
  Used(int);
  Used(int, int);
  ~Used() {}
};
struct __attribute__((warn_unused)) Unused {
  Unused();
  Unused(int);
  Unused(int, int);
  ~Unused() {}
};
void f() {
  Used();
  Used(1);
  Used(1, 1);
  Unused();     // expected-warning {{expression result unused}}
  Unused(1);    // expected-warning {{expression result unused}}
  Unused(1, 1); // expected-warning {{expression result unused}}
#if __cplusplus >= 201103L // C++11 or later
  Used({});
  Unused({}); // expected-warning {{expression result unused}}
#endif
}
}

namespace std {
  struct type_info {};
}

namespace test4 {
struct Good { Good &f(); };
struct Bad { virtual Bad& f(); };

void f() {
  int i = 0;
  (void)typeid(++i); // expected-warning {{expression with side effects has no effect in an unevaluated context}}

  Good g;
  (void)typeid(g.f()); // Ok; not a polymorphic use of a glvalue.

  // This is a polymorphic use of a glvalue, which results in the typeid being
  // evaluated instead of unevaluated.
  Bad b;
  (void)typeid(b.f()); // expected-warning {{expression with side effects will be evaluated despite being used as an operand to 'typeid'}}

  // A dereference of a volatile pointer is a side effecting operation, however
  // since it is idiomatic code, and the alternatives induce higher maintenance
  // costs, it is allowed.
  int * volatile x;
  (void)sizeof(*x); // Ok
}
}

static volatile char var1 = 'a';
volatile char var2 = 'a';
static volatile char arr1[] = "hello";
volatile char arr2[] = "hello";
void volatile_array() {
  static volatile char var3 = 'a';
  volatile char var4 = 'a';
  static volatile char arr3[] = "hello";
  volatile char arr4[] = "hello";

  // These all result in volatile loads in C and C++11. In C++98, they don't,
  // but we suppress the warning in the case where '(void)var;' might be
  // idiomatically suppressing an 'unused variable' warning.
  (void)var1;
  (void)var2;
#if __cplusplus < 201103L
  // expected-warning@-2 {{expression result unused; assign into a variable to force a volatile load}}
#endif
  (void)var3;
  (void)var4;

  // None of these result in volatile loads in any language mode, and it's not
  // really reasonable to assume that they would, since volatile array loads
  // don't really exist anywhere.
  (void)arr1;
  (void)arr2;
  (void)arr3;
  (void)arr4;
}
