// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm -O2 < %s | FileCheck %s --check-prefixes=X64,ALL
// RUN: %clang_cc1 -triple i386-pc-win32 -fms-extensions -emit-llvm -O2 < %s | FileCheck %s --check-prefixes=X86,ALL

struct Foo {
  int * __ptr32 p32;
  int * __ptr64 p64;
};
void use_foo(struct Foo *f);
void test_sign_ext(struct Foo *f, int * __ptr32 __sptr i) {
// X64-LABEL: define dso_local void @test_sign_ext({{.*}}i32 addrspace(270)* noundef %i)
// X86-LABEL: define dso_local void @test_sign_ext(%struct.Foo* noundef %f, i32* noundef %i)
// X64: %{{.+}} = addrspacecast i32 addrspace(270)* %i to i32*
// X86: %{{.+}} = addrspacecast i32* %i to i32 addrspace(272)*
  f->p64 = i;
  use_foo(f);
}
void test_zero_ext(struct Foo *f, int * __ptr32 __uptr i) {
// X64-LABEL: define dso_local void @test_zero_ext({{.*}}i32 addrspace(271)* noundef %i)
// X86-LABEL: define dso_local void @test_zero_ext({{.*}}i32 addrspace(271)* noundef %i)
// X64: %{{.+}} = addrspacecast i32 addrspace(271)* %i to i32*
// X86: %{{.+}} = addrspacecast i32 addrspace(271)* %i to i32 addrspace(272)*
  f->p64 = i;
  use_foo(f);
}
void test_trunc(struct Foo *f, int * __ptr64 i) {
// X64-LABEL: define dso_local void @test_trunc(%struct.Foo* noundef %f, i32* noundef %i)
// X86-LABEL: define dso_local void @test_trunc({{.*}}i32 addrspace(272)* noundef %i)
// X64: %{{.+}} = addrspacecast i32* %i to i32 addrspace(270)*
// X86: %{{.+}} = addrspacecast i32 addrspace(272)* %i to i32*
  f->p32 = i;
  use_foo(f);
}
void test_noop(struct Foo *f, int * __ptr32 i) {
// X64-LABEL: define dso_local void @test_noop({{.*}}i32 addrspace(270)* noundef %i)
// X86-LABEL: define dso_local void @test_noop({{.*}}i32* noundef %i)
// X64-NOT: addrspacecast
// X86-NOT: addrspacecast
  f->p32 = i;
  use_foo(f);
}

void test_other(struct Foo *f, __attribute__((address_space(10))) int *i) {
// X64-LABEL: define dso_local void @test_other({{.*}}i32 addrspace(10)* noundef %i)
// X86-LABEL: define dso_local void @test_other({{.*}}i32 addrspace(10)* noundef %i)
// X64: %{{.+}} = addrspacecast i32 addrspace(10)* %i to i32 addrspace(270)*
// X86: %{{.+}} = addrspacecast i32 addrspace(10)* %i to i32*
  f->p32 = (int * __ptr32)i;
  use_foo(f);
}

int test_compare1(int *__ptr32 __uptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local i32 @test_compare1
  // X64: %{{.+}} = addrspacecast i32* %j to i32 addrspace(271)*
  // X64: %cmp = icmp eq i32 addrspace(271)* %{{.+}}, %i
  // X86: %{{.+}} = addrspacecast i32 addrspace(272)* %j to i32 addrspace(271)*
  // X86: %cmp = icmp eq i32 addrspace(271)* %{{.+}}, %i
  return (i == j);
}

int test_compare2(int *__ptr32 __sptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local i32 @test_compare2
  // X64: %{{.+}} = addrspacecast i32* %j to i32 addrspace(270)*
  // X64: %cmp = icmp eq i32 addrspace(270)* %{{.+}}, %i
  // X86: %{{.+}} = addrspacecast i32 addrspace(272)* %j to i32*
  // X86: %cmp = icmp eq i32* %{{.+}}, %i
  return (i == j);
}

int test_compare3(int *__ptr32 __uptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local i32 @test_compare3
  // X64: %{{.+}} = addrspacecast i32 addrspace(271)* %i to i32*
  // X64: %cmp = icmp eq i32* %{{.+}}, %j
  // X86: %{{.+}} = addrspacecast i32 addrspace(271)* %i to i32 addrspace(272)*
  // X86: %cmp = icmp eq i32 addrspace(272)* %{{.+}}, %j
  return (j == i);
}

int test_compare4(int *__ptr32 __sptr i, int *__ptr64 j) {
  // ALL-LABEL: define dso_local i32 @test_compare4
  // X64: %{{.+}} = addrspacecast i32 addrspace(270)* %i to i32*
  // X64: %cmp = icmp eq i32* %{{.+}}, %j
  // X86: %{{.+}} = addrspacecast i32* %i to i32 addrspace(272)*
  // X86: %cmp = icmp eq i32 addrspace(272)* %{{.+}}, %j
  return (j == i);
}
