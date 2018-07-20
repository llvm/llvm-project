// RUN: %clang_cc1 -triple i686-windows %s -fms-extensions -emit-llvm -o - | FileCheck %s
//
// Test that LLVM optimizations leave these intrinsics alone, for the most part.
// RUN: %clang_cc1 -O2 -triple i686-windows %s -fms-extensions -emit-llvm -o - | FileCheck %s

void test1(void) {
  __annotation(L"a1");
  __annotation(L"a1", L"a2");
  __annotation(L"a1", L"a2", L"a3");
  __annotation(L"multi " L"part " L"string");
  __annotation(L"unicode: \u0ca0_\u0ca0");
}

// CHECK-LABEL: define dso_local void @test1()
// CHECK: call void @llvm.codeview.annotation(metadata ![[A1:[0-9]+]])
// CHECK: call void @llvm.codeview.annotation(metadata ![[A2:[0-9]+]])
// CHECK: call void @llvm.codeview.annotation(metadata ![[A3:[0-9]+]])
// CHECK: call void @llvm.codeview.annotation(metadata ![[A4:[0-9]+]])
// CHECK: call void @llvm.codeview.annotation(metadata ![[A5:[0-9]+]])
// CHECK: ret void

// CHECK: ![[A1]] = !{!"a1"}
// CHECK: ![[A2]] = !{!"a1", !"a2"}
// CHECK: ![[A3]] = !{!"a1", !"a2", !"a3"}
// CHECK: ![[A4]] = !{!"multi part string"}
// CHECK: ![[A5]] = !{!"unicode: \E0\B2\A0_\E0\B2\A0"}
