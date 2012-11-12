// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -emit-llvm -O3 -o - | FileCheck %s

struct Empty { };

struct A { 
  explicit A(unsigned a = 0xffffffff) : a(a) { }
  
  unsigned a;
};

struct B : A, Empty { 
  B() : A(), Empty() { }
};

struct C : A, Empty {
  C() : A(), Empty() { }
  C(const C& other) : A(0x12345678), Empty(other) { }
};

struct D : A, Empty {
  D& operator=(const D& other) {
    a = 0x87654321;
    Empty::operator=(other);
    
    return *this;
  }
};

#define CHECK(x) if (!(x)) return __LINE__

// PR7012
// CHECK: define i32 @_Z1fv()
int f() {
  B b1;

  // Check that A::a is not overwritten by the Empty default constructor.
  CHECK(b1.a == 0xffffffff);
  
  C c1;
  C c2(c1);
  
  // Check that A::a has the value set in the C::C copy constructor.
  CHECK(c2.a == 0x12345678);
  
  D d1, d2;
  d2 = d1;

  // Check that A::as has the value set in the D copy assignment operator.
  CHECK(d2.a == 0x87654321);
  
  // Success!
  // CHECK: ret i32 0
  return 0;
}

namespace PR8796 {
  struct FreeCell {
  };
  union ThingOrCell {
    FreeCell t;
    FreeCell cell;
  };
  struct Things {
    ThingOrCell things;
  };
  Things x;
}

#ifdef HARNESS
extern "C" void printf(const char *, ...);

int main() {
  int result = f();
  
  if (result == 0)
    printf("success!\n");
  else
    printf("test on line %d failed!\n", result);

  return result;
}
#endif
