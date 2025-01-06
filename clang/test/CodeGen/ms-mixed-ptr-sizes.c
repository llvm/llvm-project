// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm -O2 < %s | FileCheck %s --check-prefixes=X64,ALL
// RUN: %clang_cc1 -triple i386-pc-win32 -fms-extensions -emit-llvm -O2 < %s | FileCheck %s --check-prefixes=X86,ALL
// RUN: %clang_cc1 -triple aarch64-windows-msvc -fms-extensions -emit-llvm -O2 < %s | FileCheck %s --check-prefixes=AARCH64,ALL

struct Foo {
  int * __ptr32 p32;
  int * __ptr64 p64;
};
void use_foo(struct Foo *f);
void test_sign_ext(struct Foo *f, int * __ptr32 __sptr i) {
// X64-LABEL: define dso_local void @test_sign_ext({{.*}}ptr addrspace(270) noundef %i)
// X86-LABEL: define dso_local void @test_sign_ext(ptr noundef initializes((8, 16)) %f, ptr noundef %i)
// AARCH64-LABEL: define dso_local void @test_sign_ext({{.*}}ptr addrspace(270) noundef %i) local_unnamed_addr #0
// X64: %{{.+}} = addrspacecast ptr addrspace(270) %i to ptr
// X86: %{{.+}} = addrspacecast ptr %i to ptr addrspace(272)
// AARCH64: %{{.+}} = addrspacecast ptr addrspace(270) %i to ptr
  f->p64 = i;
  use_foo(f);
}
void test_zero_ext(struct Foo *f, int * __ptr32 __uptr i) {
// X64-LABEL: define dso_local void @test_zero_ext(ptr noundef initializes((8, 16)) %f, ptr addrspace(271) noundef %i)
// X86-LABEL: define dso_local void @test_zero_ext({{.*}}ptr addrspace(271) noundef %i)
// AARCH64-LABEL: define dso_local void @test_zero_ext(ptr noundef initializes((8, 16)) %f, ptr addrspace(271) noundef %i) local_unnamed_addr #0
// X64: %{{.+}} = addrspacecast ptr addrspace(271) %i to ptr
// X86: %{{.+}} = addrspacecast ptr addrspace(271) %i to ptr addrspace(272)
// AARCH64: %{{.+}} = addrspacecast ptr addrspace(271) %i to ptr
  f->p64 = i;
  use_foo(f);
}
void test_trunc(struct Foo *f, int * __ptr64 i) {
// X64-LABEL: define dso_local void @test_trunc(ptr noundef initializes((0, 4)) %f, ptr noundef %i)
// X86-LABEL: define dso_local void @test_trunc({{.*}}ptr addrspace(272) noundef %i)
// AARCH64-LABEL: define dso_local void @test_trunc(ptr noundef initializes((0, 4)) %f, ptr noundef %i) local_unnamed_addr #0
// X64: %{{.+}} = addrspacecast ptr %i to ptr addrspace(270)
// X86: %{{.+}} = addrspacecast ptr addrspace(272) %i to ptr
// AARCH64: %{{.+}} = addrspacecast ptr %i to ptr addrspace(270)
  f->p32 = i;
  use_foo(f);
}
void test_noop(struct Foo *f, int * __ptr32 i) {
// X64-LABEL: define dso_local void @test_noop({{.*}}ptr addrspace(270) noundef %i)
// X86-LABEL: define dso_local void @test_noop({{.*}}ptr noundef %i)
// AARCH64-LABEL: define dso_local void @test_noop({{.*}}ptr addrspace(270) noundef %i) local_unnamed_addr #0
// X64-NOT: addrspacecast
// X86-NOT: addrspacecast
// AARCH64-NOT: addrspacecast
  f->p32 = i;
  use_foo(f);
}

void test_other(struct Foo *f, __attribute__((address_space(10))) int *i) {
// X64-LABEL: define dso_local void @test_other({{.*}}ptr addrspace(10) noundef %i)
// X86-LABEL: define dso_local void @test_other({{.*}}ptr addrspace(10) noundef %i)
// AARCH64-LABEL: define dso_local void @test_other({{.*}}ptr addrspace(10) noundef %i) local_unnamed_addr #0
// X64: %{{.+}} = addrspacecast ptr addrspace(10) %i to ptr addrspace(270)
// X86: %{{.+}} = addrspacecast ptr addrspace(10) %i to ptr
// AARCH64: %{{.+}} = addrspacecast ptr addrspace(10) %i to ptr addrspace(270)
  f->p32 = (int * __ptr32)i;
  use_foo(f);
}

int test_compare1(int *__ptr32 __uptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local range(i32 0, 2) i32 @test_compare1
  // X64: %{{.+}} = addrspacecast ptr %j to ptr addrspace(271)
  // X64: %cmp = icmp eq ptr addrspace(271) %i, %{{.+}}
  // X86: %{{.+}} = addrspacecast ptr addrspace(272) %j to ptr addrspace(271)
  // X86: %cmp = icmp eq ptr addrspace(271) %i, %{{.+}}
  // AARCH64: %{{.+}} = addrspacecast ptr %j to ptr addrspace(271)
  // AARCH64: %cmp = icmp eq ptr addrspace(271) %i, %{{.+}}
  return (i == j);
}

int test_compare2(int *__ptr32 __sptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local range(i32 0, 2) i32 @test_compare2
  // X64: %{{.+}} = addrspacecast ptr %j to ptr addrspace(270)
  // X64: %cmp = icmp eq ptr addrspace(270) %i, %{{.+}}
  // X86: %{{.+}} = addrspacecast ptr addrspace(272) %j to ptr
  // X86: %cmp = icmp eq ptr %i, %{{.+}}
  // AARCH64: %{{.+}} = addrspacecast ptr %j to ptr addrspace(270)
  // AARCH64: %cmp = icmp eq ptr addrspace(270) %i, %{{.+}}
  return (i == j);
}

int test_compare3(int *__ptr32 __uptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local range(i32 0, 2) i32 @test_compare3
  // X64: %{{.+}} = addrspacecast ptr addrspace(271) %i to ptr
  // X64: %cmp = icmp eq ptr %j, %{{.+}}
  // X86: %{{.+}} = addrspacecast ptr addrspace(271) %i to ptr addrspace(272)
  // X86: %cmp = icmp eq ptr addrspace(272) %j, %{{.+}}
  // AARCH64: %{{.+}} = addrspacecast ptr addrspace(271) %i to ptr
  // AARCH64: %cmp = icmp eq ptr %j, %{{.+}}
  return (j == i);
}

int test_compare4(int *__ptr32 __sptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local range(i32 0, 2) i32 @test_compare4
  // X64: %{{.+}} = addrspacecast ptr addrspace(270) %i to ptr
  // X64: %cmp = icmp eq ptr %j, %{{.+}}
  // X86: %{{.+}} = addrspacecast ptr %i to ptr addrspace(272)
  // X86: %cmp = icmp eq ptr addrspace(272) %j, %{{.+}}
  // AARCH64: %{{.+}} = addrspacecast ptr addrspace(270) %i to ptr
  // AARCH64: %cmp = icmp eq ptr %j, %{{.+}}
  return (j == i);
}
