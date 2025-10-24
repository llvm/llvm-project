// RUN: %clang_cc1  -triple x86_64-apple-macosx -emit-llvm -debug-info-kind=limited -stdlib=libc++ -fblocks -fexceptions -x objective-c++ -o - %s | FileCheck %s

// Verify that all invoke instructions have a debug location.
// Literally: There are no unwind lines that don't end with ", (!dbg 123)".
// CHECK-NOT: {{to label %.* unwind label [^,]+$}}

void block(void (^)(void));
extern void foo();
struct A {
  ~A(void) { foo(); }
  void bar() const {}
};
void baz(void const *const) {}
struct B : A {};
void test() {
  A a;
  B b;
  block(^(void) {
    baz(&b);
    block(^() {
      a.bar();
    });
  });
}
