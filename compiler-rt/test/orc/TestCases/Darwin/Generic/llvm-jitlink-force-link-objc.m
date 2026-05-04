// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang -c -o %t/EmptyClassFoo.o  %S/Inputs/EmptyClassFoo.m
// RUN: ar r %t/libFooClass.a %t/EmptyClassFoo.o
// RUN: %clang -c -o %t/force-objc.o %s
// RUN: %llvm_jitlink -ObjC %t/force-objc.o -L%t -lFooClass
//
// REQUIRES: system-darwin && host-arch-compatible

id objc_getClass(const char *name);

int main(int argc, char *argv[]) {
  // Return succeess if we find Foo, error otherwise.
  return objc_getClass("Foo") ? 0 : 1;
}
