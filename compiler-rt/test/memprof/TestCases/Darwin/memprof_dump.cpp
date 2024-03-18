// UNSUPPORTED: ios

// RUN: %clangxx_memprof -O0 %s -o %t
// RUN: %env_memprof_opts=print_binary_refs=true:print_text=true:log_path=stdout:verbosity=2 %run %t &> %t.log
// RUN: llvm-nm %t &> %t2.log
// RUN: cat %t2.log %t.log | FileCheck %s

#include <sanitizer/memprof_interface.h>
struct __attribute__((visibility("default"))) A {
  virtual void foo() {}
};

void test_1(A *p) {
  // A has default visibility, so no need for type.checked.load.
  p->foo();
}

struct __attribute__((visibility("hidden")))
[[clang::lto_visibility_public]] B {
  virtual void foo() {}
};

void test_2(B *p) {
  // B has public LTO visibility, so no need for type.checked.load.
  p->foo();
}

struct __attribute__((visibility("hidden"))) C {
  virtual void foo() {}
  virtual void bar() {}
};

void test_3(C *p) {
  // C has hidden visibility, so we generate type.checked.load to allow VFE.
  p->foo();
}

void test_4(C *p) {
  // When using type.checked.load, we pass the vtable offset to the intrinsic,
  // rather than adding it to the pointer with a GEP.
  p->bar();
}

void test_5(C *p, void (C::*q)(void)) {
  // We also use type.checked.load for the virtual side of member function
  // pointer calls. We use a GEP to calculate the address to load from and pass
  // 0 as the offset to the intrinsic, because we know that the load must be
  // from exactly the point marked by one of the function-type metadatas (in
  // this case "_ZTSM1CFvvE.virtual"). If we passed the offset from the member
  // function pointer to the intrinsic, this information would be lost. No
  // codegen changes on the non-virtual side.
  (p->*q)();
}
int main() {
  C *p = new C;
  test_3(p);
  test_4(p);
  //__memprof_profile_dump(); // dump accesses to p, no dumping to access of the vtable
  A *a = new A;
  test_1(a);
  B *b = new B;
  test_2(b);
  __sanitizer_set_report_path("stdout");
  //__memprof_profile_dump();
  //__sanitizer_set_report_path(nullptr);
  return 0; // at exit, dump accesses to a, b
}
// CHECK: __ZTV1A
// CHECK: __ZTV1B
// 5 sections corresponding to xxx xxx __got __mod_init_func __const
// vtables are in __const
// CHECK: BuildIdName:{{.*}}memprof-vtable
// CHECK: Offset:
// CHECK: BuildIdName:{{.*}}memprof-vtable
// CHECK: Offset:
// CHECK: BuildIdName:{{.*}}memprof-vtable
// CHECK: Offset:
// CHECK: BuildIdName:{{.*}}memprof-vtable
// CHECK: Offset:
// CHECK: BuildIdName:{{.*}}memprof-vtable
// CHECK-NEXT: Start:
// symbol address for __ZTV1B + 0x10 Offset is the address in "Shadow:" (+0x10 to get the vfunc pointer)
// CHECK-NEXT: Shadow: {{.*}}
