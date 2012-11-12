// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks -std=c++11
// expected-no-diagnostics

extern "C" int exit(int);

typedef struct {
    unsigned long ps[30];
    int qs[30];
} BobTheStruct;

int main (int argc, const char * argv[]) {
    BobTheStruct inny;
    BobTheStruct outty;
    BobTheStruct (^copyStruct)(BobTheStruct);
    int i;
    
    for(i=0; i<30; i++) {
        inny.ps[i] = i * i * i;
        inny.qs[i] = -i * i * i;
    }
    
    copyStruct = ^(BobTheStruct aBigStruct){ return aBigStruct; };  // pass-by-value intrinsically copies the argument
    
    outty = copyStruct(inny);

    if ( &inny == &outty ) {
        exit(1);
    }
    for(i=0; i<30; i++) {
        if ( (inny.ps[i] != outty.ps[i]) || (inny.qs[i] != outty.qs[i]) ) {
            exit(1);
        }
    }
    
    return 0;
}

namespace rdar8134521 {
  void foo() {
    int (^P)(int) = reinterpret_cast<int(^)(int)>(1);
    P = (int(^)(int))(1);
    
    P = reinterpret_cast<int(^)(int)>((void*)1);
    P = (int(^)(int))((void*)1);
  }
}

namespace rdar11055105 {
  struct A {
    void foo();
  };

  template <class T> void foo(T &x) noexcept(noexcept(x.foo()));

  void (^block)() = ^{
    A a;
    foo(a);
  };
}
