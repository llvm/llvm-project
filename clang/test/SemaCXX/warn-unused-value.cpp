// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value %s

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
    box->j; // expected-warning {{expression result unused}}
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
  extern "C" {
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

