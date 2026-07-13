// STL allocators should not have unrelated-cast tests applied
// RUN: %clang_cc1 -flto -triple x86_64-unknown-linux -fvisibility=hidden -fsanitize=cfi-unrelated-cast -emit-llvm -o - %s | FileCheck %s

#include <stddef.h>

template<class T>
class myalloc {
 public:
  // CHECK: define{{.*}}allocateE{{.}}
  // CHECK-NOT: llvm.type.test
  T *allocate(size_t sz) {
    return (T*)::operator new(sz);
  }

  // CHECK: define{{.*}}allocateE{{.}}PKv
  // CHECK-NOT: llvm.type.test
  T *allocate(size_t sz, const void *ptr) {
    return (T*)::operator new(sz);
  }

  // CHECK: define{{.*}}differentName
  // CHECK: llvm.type.test
  T *differentName(size_t sz, const void *ptr) {
    return (T*)::operator new(sz);
  }
};

class C1 {
  virtual void f() {}
};

void f1() {
  myalloc<C1> allocator;
  (void)allocator.allocate(16);
  (void)allocator.allocate(16, 0);
  (void)allocator.differentName(16, 0);
}
