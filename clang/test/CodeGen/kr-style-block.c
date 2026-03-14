// RUN: %clang_cc1 -emit-llvm %s -o %t -fblocks

void foo (void(^)());

int main()
{
foo(
  ^() { }
);
}
