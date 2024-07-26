// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

class a {
public:
  static char *b(int);
};
int h=0;
class f {
public:
  const char *b();
  a g;
};
const char *f::b() { return g.b(h); }
void fn1() { f f1; }

// CIR: ty_22a22 = !cir.struct<class "a" {!cir.int<u, 8>} #cir.record.decl.ast>
// CIR: ty_22f22 = !cir.struct<class "f" {!cir.struct<class "a" {!cir.int<u, 8>} #cir.record.decl.ast>}>

// CIR: cir.global external @h = #cir.int<0>
// CIR: cir.func private @_ZN1a1bEi(!s32i) -> !cir.ptr<!s8i>

// CIR: cir.func @_ZN1f1bEv(%arg0: !cir.ptr<!ty_22f22> loc{{.*}}) -> !cir.ptr<!s8i>
// CIR: [[H_PTR:%.*]] = cir.get_global @h : !cir.ptr<!s32i> loc(#loc18)
// CIR: [[H_VAL:%.*]] = cir.load [[H_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR: [[RET1_VAL:%.*]] = cir.call @_ZN1a1bEi([[H_VAL]]) : (!s32i) -> !cir.ptr<!s8i>
// CIR: cir.store [[RET1_VAL]], [[RET1_P:%.*]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[RET1_VAL2:%.*]] = cir.load [[RET1_P]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
//    %7 = cir.load %1 : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!s8i>
// CIR: cir.return [[RET1_VAL2]] : !cir.ptr<!s8i>

// CIR: cir.func @_Z3fn1v()
// CIR: [[CLS_F:%.*]] = cir.alloca !ty_22f22, !cir.ptr<!ty_22f22>, ["f1"] {alignment = 1 : i64}
// CIR: cir.return

// LLVM: %class.f = type { %class.a }
// LLVM:  %class.a = type { i8 }
// LLVM: @h = global i32 0
// LLVM: declare {{.*}} ptr @_ZN1a1bEi(i32)

// LLVM: define dso_local ptr @_ZN1f1bEv(ptr [[ARG0:%.*]])
// LLVM: [[ARG0_SAVE:%.*]] = alloca ptr, i64 1, align 8
// LLVM: [[RET_SAVE:%.*]] = alloca ptr, i64 1, align 8
// LLVM: store ptr [[ARG0]], ptr [[ARG0_SAVE]], align 8,
// LLVM: [[ARG0_LOAD:%.*]] = load ptr, ptr [[ARG0_SAVE]], align 8
// LLVM: [[FUNC_PTR:%.*]] = getelementptr %class.f, ptr [[ARG0_LOAD]], i32 0, i32 0,
// LLVM: [[VAR_H:%.*]] = load i32, ptr @h, align 4
// LLVM: [[RET_VAL:%.*]] = call ptr @_ZN1a1bEi(i32 [[VAR_H]]),
// LLVM: store ptr [[RET_VAL]], ptr [[RET_SAVE]], align 8,
// LLVM: [[RET_VAL2:%.*]] = load ptr, ptr [[RET_SAVE]], align 8
// LLVM: ret ptr [[RET_VAL2]]

// LLVM: define dso_local void @_Z3fn1v()
// LLVM: [[FUNC_PTR:%.*]] = alloca %class.f, i64 1, align 1
// LLVM: ret void
