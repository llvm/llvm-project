// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

typedef struct { long a, b, c, d; } Big;
typedef struct { int a, b; } Pair;

// Scalar indirect call: no ABI reshaping, the callee pointer is called as-is.
int call_scalar(int (*fp)(int), int x) { return fp(x); }

// CIR: cir.func {{.*}}@call_scalar(%arg0: !cir.ptr<!cir.func<(!s32i) -> !s32i>> {{.*}}, %arg1: !s32i {{.*}}) -> !s32i
// CIR:   %{{.+}} = cir.call %{{.+}}(%{{.+}}) : (!cir.ptr<!cir.func<(!s32i) -> !s32i>>, !s32i {llvm.noundef}) -> !s32i
// LLVM: define dso_local i32 @call_scalar(ptr noundef %{{.+}}, i32 noundef %{{.+}})
// LLVM:   call i32 %{{.+}}(i32 noundef %{{.+}})

// Indirect call whose small-struct argument and return are coerced to a
// register: the callee pointer is bitcast so its pointee tracks both.
Pair call_coerce(Pair (*fp)(Pair), Pair p) { return fp(p); }

// CIR: cir.func {{.*}}@call_coerce(%arg0: !cir.ptr<!cir.func<(!rec_Pair) -> !rec_Pair>> {{.*}}, %arg1: !u64i{{.*}}) -> !u64i
// CIR:   %[[PCAST:.*]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Pair) -> !rec_Pair>> -> !cir.ptr<!cir.func<(!u64i) -> !u64i>>
// CIR:   %{{.+}} = cir.call %[[PCAST]](%{{.+}}) : (!cir.ptr<!cir.func<(!u64i) -> !u64i>>, !u64i) -> !u64i
// LLVM: define dso_local i64 @call_coerce(ptr noundef %{{.+}}, i64 %{{.+}})
// LLVM:   call i64 %{{.+}}(i64 %{{.+}})

// Indirect call with a byval struct argument: the argument is spilled to a
// stack slot and the callee pointer is bitcast to the coerced signature.
long call_byval(long (*fp)(Big), Big b) { return fp(b); }

// CIR: cir.func {{.*}}@call_byval(%arg0: !cir.ptr<!cir.func<(!rec_Big) -> !s64i>> {{.*}}, %arg1: !cir.ptr<!rec_Big> {{.*}}llvm.byval = !rec_Big{{.*}}) -> !s64i
// CIR:   %[[CAST:.*]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<(!rec_Big) -> !s64i>> -> !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>) -> !s64i>>
// CIR:   %{{.+}} = cir.call %[[CAST]](%{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Big>) -> !s64i>>, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}) -> !s64i
// LLVM: define dso_local i64 @call_byval(ptr noundef %{{.+}}, ptr noundef byval(%struct.Big) align 8 %{{.+}})
// LLVM: call i64 %{{.+}}(ptr noundef byval(%struct.Big) align 8 %{{.+}})

// Indirect call returning a large struct: an sret pointer slot is prepended
// and the callee is bitcast to the void-returning sret signature.
Big call_sret(Big (*fp)(void)) { return fp(); }

// CIR: cir.func {{.*}}@call_sret(%arg0: !cir.ptr<!rec_Big> {{.*}}llvm.sret = !rec_Big{{.*}}, %arg1: !cir.ptr<!cir.func<() -> !rec_Big>> {{.*}})
// CIR:   %[[SCAST:.*]] = cir.cast bitcast %{{.+}} : !cir.ptr<!cir.func<() -> !rec_Big>> -> !cir.ptr<!cir.func<(!cir.ptr<!rec_Big>)>>
// CIR:   cir.call %[[SCAST]](%{{.+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Big>)>>, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.dead_on_unwind, llvm.sret = !rec_Big, llvm.writable}) -> ()
// LLVM: define dso_local void @call_sret(ptr dead_on_unwind noalias writable sret(%struct.Big) align 8 %{{.+}}, ptr noundef %{{.+}})
// LLVM:   call void %{{.+}}(ptr dead_on_unwind writable sret(%struct.Big) align 8 %{{.+}})
