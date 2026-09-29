// RUN: %clang_cc1 %s -std=c++26 -freflection -fblocks -fsyntax-only

struct A {
  int operator^(int (^block)(int x)) const {
    return block(0);
  }
};


consteval void test()
{
    (void)(A{}^^(int y){ return y + 1; });
    (void)(1^^(){ return 1; }());
    (void)(1^^{ return 1; }());

    {
      (void)(^^int);
      (void)(A{}^^(int y){ return y + 1; });
    }
}
