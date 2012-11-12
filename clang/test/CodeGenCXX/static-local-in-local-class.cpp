// RUN: %clang_cc1 -emit-llvm -o %t %s
// PR6769

struct X {
  static void f();
};

void X::f() {
  static int *i;
  {
    struct Y {
      static void g() {
        i = new int();
	*i = 100;
	(*i) = (*i) +1;
      }
    };
    (void)Y::g();
  }
  (void)i;
}

// pr7101
void foo() {
    static int n = 0;
    struct Helper {
        static void Execute() {
            n++;
        }
    };
    Helper::Execute();
}

