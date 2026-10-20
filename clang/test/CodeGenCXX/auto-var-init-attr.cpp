// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown -fblocks -fdeclspec -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s

struct S { char c; };
class C { char c; };
enum class E { ZERO };
union U { char c; int i; };

struct __declspec(no_init_all) NoInitS { char c; };
class __declspec(no_init_all) NoInitC { char c; };
enum class __declspec(no_init_all) NoInitE { ZERO };
union __declspec(no_init_all) NoInitU { char c; int i; };

extern "C" {
  void test_no_attr() {
    // CHECK-LABEL: @test_no_attr()
    // CHECK-NEXT:  entry:
    // CHECK-NEXT:  %s = alloca %struct.S, align 1
    // CHECK-NEXT:  %c = alloca %class.C, align 1
    // CHECK-NEXT:  %e = alloca i32, align 4
    // CHECK-NEXT:  %u = alloca %union.U, align 4
    // CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 1 %s, i8 0, i64 1, i1 false)
    // CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 1 %c, i8 0, i64 1, i1 false)
    // CHECK-NEXT:  store i32 0, ptr %e, align 4
    // CHECK-NEXT:  call void @llvm.memset.p0.i64(ptr align 4 %u, i8 0, i64 4, i1 false)
    // CHECK-NEXT   ret void
    S s;
    C c;
    E e;
    U u;
  }

  void __declspec(no_init_all) test_attr_on_function() {
    // CHECK-LABEL: @test_attr_on_function()
    // CHECK-NEXT:  entry:
    // CHECK-NEXT:  %s = alloca %struct.S, align 1
    // CHECK-NEXT:  %c = alloca %class.C, align 1
    // CHECK-NEXT:  %e = alloca i32, align 4
    // CHECK-NEXT:  %u = alloca %union.U, align 4
    // CHECK-NEXT:  ret void
    S s;
    C c;
    E e;
    U u;
  }

  void test_attr_on_decl() {
    // CHECK-LABEL: @test_attr_on_decl()
    // CHECK-NEXT:  entry:
    // CHECK-NEXT:  %s = alloca %struct.NoInitS, align 1
    // CHECK-NEXT:  %c = alloca %class.NoInitC, align 1
    // CHECK-NEXT:  %e = alloca i32, align 4
    // CHECK-NEXT:  %u = alloca %union.NoInitU, align 4
    // CHECK-NEXT:  ret void
    NoInitS s;
    NoInitC c;
    NoInitE e;
    NoInitU u;
  }
}