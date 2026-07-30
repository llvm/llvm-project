// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

void check(int&) = delete;
void check(int const&) { }

template <typename>
struct A {
    union {
        int b;
    };
    struct {
      int c;
    };
    union {
      struct {
        union {
          struct {
            struct {
              int d;
            };
          };
        };
      };
    };
    int e;
    void foo() const {
      check(b);
      check(c);
      check(d);
      check(d);
      check(e);
    }
};

int main(){
    A<int> a;
    a.foo();
}
