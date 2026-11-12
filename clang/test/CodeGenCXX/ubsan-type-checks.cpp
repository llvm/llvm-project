// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=alignment | FileCheck %s -check-prefixes=ALIGN,COMMON
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=null | FileCheck %s -check-prefixes=NULL,COMMON
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=object-size | FileCheck %s -check-prefixes=OBJSIZE,COMMON
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=null,vptr | FileCheck %s -check-prefixes=VPTR
// RUN: %clang_cc1 -std=c++11 -triple x86_64-apple-darwin10 -emit-llvm -o - %s -fsanitize=vptr | FileCheck %s -check-prefixes=VPTR_NO_NULL

struct A {
  // COMMON-LABEL: define linkonce_odr void @_ZN1A10do_nothingEv
  void do_nothing() {
    // ALIGN-NOT: ptrtoint ptr %{{.*}} to i64, !nosanitize
 
    // NULL: icmp ne ptr %{{.*}}, null, !nosanitize
 
    // OBJSIZE-NOT: call i64 @llvm.objectsize
  }
};

struct B {
  int x;

  // COMMON-LABEL: define linkonce_odr void @_ZN1B10do_nothingEv
  void do_nothing() {
    // ALIGN: ptrtoint ptr %{{.*}} to i64, !nosanitize
    // ALIGN: and i64 %{{.*}}, 3, !nosanitize

    // NULL: icmp ne ptr %{{.*}}, null, !nosanitize

    // OBJSIZE-NOT: call i64 @llvm.objectsize
    // OBJSIZE: ret void
  }
};

struct Animal {
  virtual const char *speak() = 0;
};

struct Cat : Animal {
  const char *speak() override { return "meow"; }
};

struct Dog : Animal {
  const char *speak() override { return "woof"; }
};

// VPTR-LABEL: define{{.*}} void @_Z12invalid_castP3Cat
void invalid_cast(Cat *cat = nullptr) {
  // If -fsanitize=null is available, we'll reuse its check:
  //
  // VPTR: [[ICMP:%.*]] = icmp ne ptr {{.*}}, null
  // VPTR-NEXT: br i1 [[ICMP]]
  // VPTR: call void @__ubsan_handle_type_mismatch
  // VPTR-NOT: icmp ne ptr {{.*}}, null
  // VPTR: br i1 [[ICMP]]
  // VPTR: call void @__ubsan_handle_dynamic_type_cache_miss
  //
  // Fall back to the vptr sanitizer's null check when -fsanitize=null isn't
  // available.
  //
  // VPTR_NO_NULL-NOT: call void @__ubsan_handle_type_mismatch
  // VPTR_NO_NULL: [[ICMP:%.*]] = icmp ne ptr {{.*}}, null
  // VPTR_NO_NULL-NEXT: br i1 [[ICMP]]
  // VPTR_NO_NULL: call void @__ubsan_handle_dynamic_type_cache_miss
  auto *badDog = reinterpret_cast<Dog *>(cat);
  badDog->speak();
}

// VPTR_NO_NULL-LABEL: define{{.*}} void @_Z13invalid_cast2v
void invalid_cast2() {
  // We've got a pointer to an alloca, so there's no run-time null check needed.
  // VPTR_NO_NULL-NOT: call void @__ubsan_handle_type_mismatch
  // VPTR_NO_NULL: call void @__ubsan_handle_dynamic_type_cache_miss
  Cat cat;
  cat.speak();
}

int main() {
  A a;
  a.do_nothing();

  B b;
  b.do_nothing();

  invalid_cast();
  return 0;
}
