// RUN: %clang_cc1 -triple=i686-pc-unknown -std=c++11 %s  -emit-llvm -o - | FileCheck %s

// This was a problem in Sema, but only shows up as noinline missing
// in CodeGen.

// CHECK: define{{.*}} i32 @_Z15noduplicatedfuni(i32 %a) [[NI:#[0-9]+]]

int noduplicatedfun [[clang::noduplicate]] (int a) {

  return a+1;

}

int main() {

  return noduplicatedfun(5);

}

// CHECK: attributes [[NI]] = { noduplicate {{.*}}nounwind{{.*}} }
