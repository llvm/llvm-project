// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

   struct fu;
   void foo(struct fu);
   void bar() {
      foo;
   }
