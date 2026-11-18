// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef struct { char c[64]; } Big;
typedef struct { int a, b; } Pair;
typedef struct { long a, b; } Two;
typedef struct { } E0;
typedef void (*LogFn)(const char *, ...);
typedef void (*PairFn)(Pair, ...);
typedef int (*PairIntFn)(Pair, ...);
typedef Big (*SretFn)(Pair, ...);

// A struct too large for registers goes byval through the ellipsis and stays
// out of the callee pointer's function type.
void call_it(LogFn p, Big b) { p("%s", b); }

// CIR-LABEL: cir.func {{.*}}@call_it(%arg0: !cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...)>> {llvm.noundef} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef} loc({{[^)]+}}))
// CIR-NOT: cir.cast bitcast {{.*}}!cir.func
// CIR:   cir.call %{{.+}}(%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...)>>, !cir.ptr<!s8i> {llvm.noundef}, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @call_it(ptr noundef %{{[^,)]+}}, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})
// LLVM: call void (ptr, ...) %{{[^,)]+}}(ptr noundef @.str, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})

// A one-eightbyte record is coerced to an integer register and a
// two-eightbyte record flattened into two.  A _BitInt(8), which the default
// argument promotions leave alone, is extended.  None of them retypes the
// callee pointer.
void pass_pair(LogFn p, const char *f, Pair x) { p(f, x); }
void pass_two(LogFn p, const char *f, Two x) { p(f, x); }
void pass_bitint(LogFn p, const char *f, _BitInt(8) x) { p(f, x); }

// CIR-LABEL: cir.func {{.*}}@pass_pair(
// CIR-NOT: cir.cast bitcast {{.*}}!cir.func
// CIR: cir.call %{{.+}}(%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...)>>, !cir.ptr<!s8i> {llvm.noundef}, !u64i) -> ()
// CIR-LABEL: cir.func {{.*}}@pass_two(
// CIR-NOT: cir.cast bitcast {{.*}}!cir.func
// CIR: cir.call %{{.+}}(%{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...)>>, !cir.ptr<!s8i> {llvm.noundef}, !s64i, !s64i) -> ()
// CIR-LABEL: cir.func {{.*}}@pass_bitint(
// CIR-NOT: cir.cast bitcast {{.*}}!cir.func
// CIR: cir.call %{{.+}}(%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!s8i>, ...)>>, !cir.ptr<!s8i> {llvm.noundef}, !s8i_bitint {llvm.noundef, llvm.signext}) -> ()

// LLVM-LABEL: define dso_local void @pass_pair(
// LLVM: call void (ptr, ...) %{{[^,)]+}}(ptr noundef %{{[^,)]+}}, i64 %{{[^,)]+}})
// LLVM-LABEL: define dso_local void @pass_two(
// LLVM: call void (ptr, ...) %{{[^,)]+}}(ptr noundef %{{[^,)]+}}, i64 %{{[^,)]+}}, i64 %{{[^,)]+}})
// LLVM-LABEL: define dso_local void @pass_bitint(
// LLVM: call void (ptr, ...) %{{[^,)]+}}(ptr noundef %{{[^,)]+}}, i8 noundef signext %{{[^,)]+}})

// The callee pointer is retyped to the coerced declared parameter, behind the
// sret slot when there is one.  Arguments passed through the ellipsis stay
// out of that type, and an empty record there is dropped.
int pass_declared_coerced(PairIntFn p, Pair x, Big b) { return p(x, b); }
Big pass_sret(SretFn p, Pair x, Big b, Pair y) { return p(x, b, y); }
void pass_trailing_empty(PairFn p, Pair x, E0 e) { p(x, e); }

// CIR-LABEL: cir.func {{.*}}@pass_declared_coerced(
// CIR: %[[CAST:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Pair, ...) -> !s32i>> -> !cir.ptr<!cir.func<(!u64i, ...) -> !s32i>>
// CIR: cir.call %[[CAST]](%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!u64i, ...) -> !s32i>>, !u64i, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}) -> !s32i
// CIR-LABEL: cir.func {{.*}}@pass_sret(
// CIR: %[[CAST:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Pair, ...) -> !rec_Big>> -> !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !u64i, ...)>>
// CIR: cir.call %[[CAST]](%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Big>, !u64i, ...)>>, !cir.ptr<!rec_Big> {llvm.align = 1 : i64, llvm.dead_on_unwind, llvm.sret = !rec_Big, llvm.writable}, !u64i, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}, !u64i) -> ()
// CIR-LABEL: cir.func {{.*}}@pass_trailing_empty(
// CIR: %[[CAST:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Pair, ...)>> -> !cir.ptr<!cir.func<(!u64i, ...)>>
// CIR: cir.call %[[CAST]](%{{.+}}) : (!cir.ptr<!cir.func<(!u64i, ...)>>, !u64i) -> ()

// LLVM-LABEL: define dso_local i32 @pass_declared_coerced(
// LLVM: call i32 (i64, ...) %{{[^,)]+}}(i64 %{{[^,)]+}}, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})
// LLVM-LABEL: define dso_local void @pass_sret(
// LLVM: call void (ptr, i64, ...) %{{[^,)]+}}(ptr dead_on_unwind writable sret(%struct.Big) align 1 %{{[^,)]+}}, i64 %{{[^,)]+}}, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}}, i64 %{{[^,)]+}})
// LLVM-LABEL: define dso_local void @pass_trailing_empty(
// LLVM: call void (i64, ...) %{{[^,)]+}}(i64 %{{[^,)]+}})

// A declared parameter that flattens into two registers puts both in the
// retyped callee pointer's function type, and the argument passed through the
// ellipsis stays out of it.
typedef int (*TwoFn)(Two, ...);
int pass_declared_flat(TwoFn p, Two t, Pair q) { return p(t, q); }

// CIR-LABEL: cir.func {{.*}}@pass_declared_flat(
// CIR: %[[CAST:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Two, ...) -> !s32i>> -> !cir.ptr<!cir.func<(!s64i, !s64i, ...) -> !s32i>>
// CIR: cir.call %[[CAST]](%{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!s64i, !s64i, ...) -> !s32i>>, !s64i, !s64i, !u64i) -> !s32i

// LLVM-LABEL: define dso_local i32 @pass_declared_flat(
// LLVM: call i32 (i64, i64, ...) %{{[^,)]+}}(i64 %{{[^,)]+}}, i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// The ellipsis arguments are classified with the rest of the call, so the
// same Two that pass_two passes in registers goes byval once the integer
// registers run out.
typedef int (*IntFn)(int, ...);
int call_exhausted(IntFn p, long a, long b, long c, long d, long e, Two q) {
  return p(1, a, b, c, d, e, q);
}

// CIR-LABEL: cir.func {{.*}}@call_exhausted(
// CIR-NOT: cir.cast bitcast {{.*}}!cir.func
// CIR: cir.call %{{.+}}(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!s32i, ...) -> !s32i>>, !s32i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !cir.ptr<!rec_Two> {llvm.align = 8 : i64, llvm.byval = !rec_Two, llvm.noundef}) -> !s32i

// LLVM-LABEL: define dso_local i32 @call_exhausted(
// LLVM: call i32 (i32, ...) %{{[^,)]+}}(i32 noundef 1, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, i64 noundef %{{[^,)]+}}, ptr noundef byval(%struct.Two) align 8 %{{[^,)]+}})

// The address of a rewritten variadic definition is cast back to its written
// type, and a call through it coerces to the rewritten signature again.
int vdef(Pair p, ...) { return p.a; }
int call_addr(Pair x, Big b) {
  int (*fp)(Pair, ...) = vdef;
  return fp(x, b);
}

// CIR-LABEL: cir.func {{.*}}@call_addr(
// CIR: %[[ADDR:.+]] = cir.get_global @vdef : !cir.ptr<!cir.func<(!u64i, ...) -> !s32i>>
// CIR: cir.cast bitcast %[[ADDR]] : !cir.ptr<!cir.func<(!u64i, ...) -> !s32i>> -> !cir.ptr<!cir.func<(!rec_Pair, ...) -> !s32i>>
// CIR: %[[CAST:.+]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Pair, ...) -> !s32i>> -> !cir.ptr<!cir.func<(!u64i, ...) -> !s32i>>
// CIR: cir.call %[[CAST]](%{{.+}}, %{{.+}}) : (!cir.ptr<!cir.func<(!u64i, ...) -> !s32i>>, !u64i, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}) -> !s32i

// LLVM-LABEL: define dso_local i32 @call_addr(
// LLVM: call i32 (i64, ...) %{{[^,)]+}}(i64 %{{[^,)]+}}, ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})
