// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR %s --input-file=%t.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM %s --input-file=%t.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --check-prefix=OGCG %s --input-file=%t-og.ll

// Test for crash in replaceGlobal with ConstantOp containing GlobalViewAttr.
// The crash occurred because getNewInitValue() can return GlobalViewAttr,
// ConstRecordAttr, or ConstArrayAttr, but the code incorrectly assumed
// it always returns ConstArrayAttr.

char typedef e;
typedef struct a *b;
typedef struct { b a; } f;
struct a { e a; f b; };
static struct a d = {};
const b a = &d;
b c() { return a; }

// CIR: cir.global "private" internal dso_local @d
// CIR: cir.global constant external @a = #cir.global_view<@d>

// LLVM: %struct.f = type { ptr }
// LLVM: @d = internal global { i8, [7 x i8], %struct.f } zeroinitializer, align 8
// LLVM: @a = constant ptr @d, align 8

// OGCG: %struct.f = type { ptr }
// OGCG: @a = constant ptr @d, align 8
// OGCG: @d = internal global { i8, [7 x i8], %struct.f } zeroinitializer, align 8
