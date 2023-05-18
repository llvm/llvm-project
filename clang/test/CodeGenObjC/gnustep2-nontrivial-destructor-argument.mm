// RUN: %clang_cc1 -triple x86_64-unknow-windows-msvc -S -emit-llvm -fobjc-runtime=gnustep-2.0 -o - %s 

// Regression test.  Ensure that C++ arguments with non-trivial destructors
// don't crash the compiler.

struct X
{
  int a;
  ~X();
};

@protocol Y
- (void)foo: (X)bar;
@end


void test(id<Y> obj)
{
  X a{12};
  [obj foo: a];
}

