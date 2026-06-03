// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Globals whose initializer contains the global's own address with a non-zero
// byte offset. The correctness property under test is that the resulting GEP
// (or CIR global_view index path) lands on the named byte, no matter how the
// initializer's record-type representation is shaped.

// Case 1: simple self-reference, no empty fields, no alignment padding.
struct S1 { int x; int y; int *p; };
struct S1 s1 = { 1, 2, &s1.y };

// CIR-LABEL: cir.global external @s1
// CIR-SAME: #cir.global_view<@s1, [1 : i32]> : !cir.ptr<!s32i>
// LLVM: @s1 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @s1, i64 4)
// OGCG: @s1 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @s1, i64 4)

// Case 2: self-reference through a nested struct member.
struct Inner2 { int a; int b; };
struct S2 { int x; struct Inner2 inner; int *p; };
struct S2 s2 = { 10, {20, 30}, &s2.inner.b };

// CIR-LABEL: cir.global external @s2
// CIR-SAME: #cir.global_view<@s2, [1 : i32, 1 : i32]> : !cir.ptr<!s32i>
// LLVM: @s2 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @s2, i64 8)
// OGCG: @s2 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @s2, i64 8)

// Case 3: self-reference into an array element.
struct S3 { int header; int arr[4]; int *p; };
struct S3 s3 = { 100, {11, 22, 33, 44}, &s3.arr[2] };

// CIR-LABEL: cir.global external @s3
// CIR-SAME: #cir.global_view<@s3, [1 : i32, 2 : i32]> : !cir.ptr<!s32i>
// LLVM: @s3 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @s3, i64 12)
// OGCG: @s3 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @s3, i64 12)

// Case 4: empty struct field precedes the referenced field. The referenced
// byte is at offset 0 (because the empty struct contributes no bytes).
struct E4 {};
struct S4 { struct E4 e; int x; int *p; };
struct S4 s4 = { {}, 42, &s4.x };

// CIR-LABEL: cir.global external @s4
// CIR-SAME: #cir.global_view<@s4> : !cir.ptr<!s32i>
// LLVM: @s4 ={{.*}}ptr @s4
// OGCG: @s4 ={{.*}}ptr @s4

// Case 5: empty struct field BEFORE the referenced field. Currently silently
// miscompiles in CIR — the GEP lands on the wrong offset because the empty
// struct filler shifts the structural indices used by global_view.
struct E5 {};
struct Inner5 { int x; };
struct S5 { struct E5 e; int n; struct Inner5 inner; int *p; };
struct S5 s5 = { .p = &s5.inner.x };

// CIR-LABEL: cir.global external @s5
// CIR-SAME: #cir.global_view<@s5, [1 : i32]> : !cir.ptr<!s32i>
// LLVM: @s5 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @s5, i64 4)
// OGCG: @s5 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @s5, i64 4)

// Case 6: empty struct fields PLUS alignment-induced padding. This is the
// 714 reproducer pattern — currently crashes the CIR-to-LLVM lowering with
// "type 'i32' cannot be indexed (index #2)".
struct E6a {};
struct E6b {};
struct Inner6 { int a, b; };
struct S6 {
  struct E6a ea;
  int x;
  struct E6b eb;
  struct Inner6 inner;
  int *p;          // forces 4 bytes alignment padding before p
};
struct S6 s6 = { .x = 7, .p = &s6.inner.b };

// CIR-LABEL: cir.global external @s6
// CIR-SAME: #cir.global_view<@s6, [1 : i32, 1 : i32]> : !cir.ptr<!s32i>
// LLVM: @s6 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @s6, i64 8)
// OGCG: @s6 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @s6, i64 8)
