// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fapple-kext -fno-rtti -emit-llvm -o - %s | FileCheck %s
// rdar://8825235
/**
1) Normally, global object construction code ends up in __StaticInit segment of text section
   .section __TEXT,__StaticInit,regular,pure_instructions
   In kext mode, they end up in the __text segment.
*/

class foo {
public:
  foo();
  virtual ~foo();
};

foo a;
foo b;
foo c;
foo::~foo() {}

// CHECK-NOT: __TEXT,__StaticInit,regular,pure_instructions
