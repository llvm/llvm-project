// RUN: %clang_cc1 -S -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

bool bar();
void f(bool, bool);

void foo(int i) {
  [[clang::nomerge]] bar();
  [[clang::nomerge]] (i = 4, bar());
  [[clang::nomerge]] (void)(bar());
  [[clang::nomerge]] f(bar(), bar());
  [[clang::nomerge]] [] { bar(); bar(); }(); // nomerge only applies to the anonymous function call
  [[clang::nomerge]] for (bar(); bar(); bar()) {}
  [[clang::nomerge]] { asm("nop"); }
  bar();
}
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR:[0-9]+]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call void @_Z1fbb({{.*}}) #[[NOMERGEATTR]]
// CHECK: call void @"_ZZ3fooiENK3$_0clEv"(%class.anon* {{[^,]*}} %ref.tmp) #[[NOMERGEATTR]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call zeroext i1 @_Z3barv() #[[NOMERGEATTR]]
// CHECK: call void asm {{.*}} #[[NOMERGEATTR2:[0-9]+]]
// CHECK: call zeroext i1 @_Z3barv()
// CHECK: attributes #[[NOMERGEATTR]] = { nomerge }
// CHECK: attributes #[[NOMERGEATTR2]] = { nomerge nounwind }
