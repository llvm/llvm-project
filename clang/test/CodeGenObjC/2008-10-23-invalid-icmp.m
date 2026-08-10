// RUN: %clang_cc1 -emit-llvm -o %t %s

@protocol P @end

int f0(id<P> d) {
  return (d != ((void*) 0));
}
