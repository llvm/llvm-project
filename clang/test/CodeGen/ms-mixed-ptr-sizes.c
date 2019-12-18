// RUN: %clang_cc1 -triple x86_64-windows-msvc -fms-extensions -emit-llvm -O2 \
// RUN:   < %s | FileCheck %s --check-prefixes=X64,CHECK
// RUN: %clang_cc1 -triple i386-pc-win32 -fms-extensions -emit-llvm -O2 \
// RUN:   < %s | FileCheck %s --check-prefixes=X86,CHECK

struct Foo {
  int * __ptr32 p32;
  int * __ptr64 p64;
};
void use_foo(struct Foo *f);
void test_sign_ext(struct Foo *f, int * __ptr32 __sptr i) {
// X64-LABEL: define dso_local void @test_sign_ext({{.*}}i32 addrspace(270)* %i)
// X86-LABEL: define dso_local void @test_sign_ext(%struct.Foo* %f, i32* %i)
// X64: %{{.+}} = addrspacecast i32 addrspace(270)* %i to i32*
// X86: %{{.+}} = addrspacecast i32* %i to i32 addrspace(272)*
  f->p64 = i;
  use_foo(f);
}
void test_zero_ext(struct Foo *f, int * __ptr32 __uptr i) {
// X64-LABEL: define dso_local void @test_zero_ext({{.*}}i32 addrspace(271)* %i)
// X86-LABEL: define dso_local void @test_zero_ext({{.*}}i32 addrspace(271)* %i)
// X64: %{{.+}} = addrspacecast i32 addrspace(271)* %i to i32*
// X86: %{{.+}} = addrspacecast i32 addrspace(271)* %i to i32 addrspace(272)*
  f->p64 = i;
  use_foo(f);
}
void test_trunc(struct Foo *f, int * __ptr64 i) {
// X64-LABEL: define dso_local void @test_trunc(%struct.Foo* %f, i32* %i)
// X86-LABEL: define dso_local void @test_trunc({{.*}}i32 addrspace(272)* %i)
// X64: %{{.+}} = addrspacecast i32* %i to i32 addrspace(270)*
// X86: %{{.+}} = addrspacecast i32 addrspace(272)* %i to i32*
  f->p32 = i;
  use_foo(f);
}
void test_noop(struct Foo *f, int * __ptr32 i) {
// X64-LABEL: define dso_local void @test_noop({{.*}}i32 addrspace(270)* %i)
// X86-LABEL: define dso_local void @test_noop({{.*}}i32* %i)
// X64-NOT: addrspacecast
// X86-NOT: addrspacecast
  f->p32 = i;
  use_foo(f);
}

void test_other(struct Foo *f, __attribute__((address_space(10))) int *i) {
// X64-LABEL: define dso_local void @test_other({{.*}}i32 addrspace(10)* %i)
// X86-LABEL: define dso_local void @test_other({{.*}}i32 addrspace(10)* %i)
// X64: %{{.+}} = addrspacecast i32 addrspace(10)* %i to i32 addrspace(270)*
// X86: %{{.+}} = addrspacecast i32 addrspace(10)* %i to i32*
  f->p32 = (int * __ptr32)i;
  use_foo(f);
}
