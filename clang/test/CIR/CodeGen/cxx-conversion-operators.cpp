// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++11 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct inline_operator {
  operator int() const { return 987; }

  int operator+(inline_operator) { return 666; }
};

struct out_of_line_operator {
  operator int();
};

out_of_line_operator::operator int() { return 123; }

void test() {
  int x = 42;

  inline_operator i;
  x = i;

  out_of_line_operator o;
  x = o;
}

// CIR: cir.func dso_local @_ZN20out_of_line_operatorcviEv(%[[THIS_ARG:.+]]: !cir.ptr<!rec_out_of_line_operator>{{.*}}) -> !s32i
// CIR:   %[[THIS_ALLOCA:.+]] = cir.alloca !cir.ptr<!rec_out_of_line_operator>, !cir.ptr<!cir.ptr<!rec_out_of_line_operator>>, ["this", init]
// CIR:   %[[RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ALLOCA]] : !cir.ptr<!rec_out_of_line_operator>, !cir.ptr<!cir.ptr<!rec_out_of_line_operator>>
// CIR:   %[[THIS_LOAD:.+]] = cir.load %[[THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_out_of_line_operator>>, !cir.ptr<!rec_out_of_line_operator>
// CIR:   %[[CONST_123:.+]] = cir.const #cir.int<123> : !s32i
// CIR:   cir.store %[[CONST_123]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[RET_LOAD:.+]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[RET_LOAD]] : !s32i
// CIR: }

// CIR: cir.func comdat linkonce_odr @_ZNK15inline_operatorcviEv(%[[INLINE_THIS_ARG:.+]]: !cir.ptr<!rec_inline_operator>{{.*}}) -> !s32i
// CIR:   %[[INLINE_THIS_ALLOCA:.+]] = cir.alloca !cir.ptr<!rec_inline_operator>, !cir.ptr<!cir.ptr<!rec_inline_operator>>, ["this", init]
// CIR:   %[[INLINE_RETVAL:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR:   cir.store %[[INLINE_THIS_ARG]], %[[INLINE_THIS_ALLOCA]] : !cir.ptr<!rec_inline_operator>, !cir.ptr<!cir.ptr<!rec_inline_operator>>
// CIR:   %[[INLINE_THIS_LOAD:.+]] = cir.load %[[INLINE_THIS_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_inline_operator>>, !cir.ptr<!rec_inline_operator>
// CIR:   %[[CONST_987:.+]] = cir.const #cir.int<987> : !s32i
// CIR:   cir.store %[[CONST_987]], %[[INLINE_RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[INLINE_RET_LOAD:.+]] = cir.load %[[INLINE_RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return %[[INLINE_RET_LOAD]] : !s32i
// CIR: }

// CIR: cir.func dso_local @_Z4testv()
// CIR:   %[[X_ALLOCA:.+]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR:   %[[I_ALLOCA:.+]] = cir.alloca {{.*}}, {{.*}}, ["i"]
// CIR:   %[[O_ALLOCA:.+]] = cir.alloca {{.*}}, {{.*}}, ["o"]
// CIR:   %[[CONST_42:.+]] = cir.const #cir.int<42> : !s32i
// CIR:   cir.store align(4) %[[CONST_42]], %[[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[INLINE_CALL:.+]] = cir.call @_ZNK15inline_operatorcviEv(%[[I_ALLOCA]]) : ({{.*}}) -> !s32i
// CIR:   cir.store align(4) %[[INLINE_CALL]], %[[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[OUTLINE_CALL:.+]] = cir.call @_ZN20out_of_line_operatorcviEv(%[[O_ALLOCA]]) : ({{.*}}) -> !s32i
// CIR:   cir.store align(4) %[[OUTLINE_CALL]], %[[X_ALLOCA]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.return
// CIR: }

// LLVM: define dso_local i32 @_ZN20out_of_line_operatorcviEv(ptr %[[PARAM0:.+]])
// LLVM:   %[[THIS_ALLOCA:.+]] = alloca ptr, i64 1
// LLVM:   %[[RETVAL:.+]] = alloca i32, i64 1
// LLVM:   store ptr %[[PARAM0]], ptr %[[THIS_ALLOCA]]
// LLVM:   %[[THIS_LOAD:.+]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   store i32 123, ptr %[[RETVAL]]
// LLVM:   %[[RET_LOAD:.+]] = load i32, ptr %[[RETVAL]]
// LLVM:   ret i32 %[[RET_LOAD]]
// LLVM: }

// LLVM: define linkonce_odr i32 @_ZNK15inline_operatorcviEv(ptr %[[INLINE_PARAM0:.+]])
// LLVM:   %[[INLINE_THIS_ALLOCA:.+]] = alloca ptr, i64 1
// LLVM:   %[[INLINE_RETVAL:.+]] = alloca i32, i64 1
// LLVM:   store ptr %[[INLINE_PARAM0]], ptr %[[INLINE_THIS_ALLOCA]]
// LLVM:   %[[INLINE_THIS_LOAD:.+]] = load ptr, ptr %[[INLINE_THIS_ALLOCA]]
// LLVM:   store i32 987, ptr %[[INLINE_RETVAL]]
// LLVM:   %[[INLINE_RET_LOAD:.+]] = load i32, ptr %[[INLINE_RETVAL]]
// LLVM:   ret i32 %[[INLINE_RET_LOAD]]
// LLVM: }

// LLVM: define dso_local void @_Z4testv()
// LLVM:   %[[X_ALLOCA:.+]] = alloca i32, i64 1
// LLVM:   %[[I_ALLOCA:.+]] = alloca {{.*}}, i64 1
// LLVM:   %[[O_ALLOCA:.+]] = alloca {{.*}}, i64 1
// LLVM:   store i32 42, ptr %[[X_ALLOCA]]
// LLVM:   %[[INLINE_CALL:.+]] = call i32 @_ZNK15inline_operatorcviEv(ptr %[[I_ALLOCA]])
// LLVM:   store i32 %[[INLINE_CALL]], ptr %[[X_ALLOCA]]
// LLVM:   %[[OUTLINE_CALL:.+]] = call i32 @_ZN20out_of_line_operatorcviEv(ptr %[[O_ALLOCA]])
// LLVM:   store i32 %[[OUTLINE_CALL]], ptr %[[X_ALLOCA]]
// LLVM:   ret void
// LLVM: }

// OGCG: define dso_local noundef i32 @_ZN20out_of_line_operatorcviEv(ptr {{.*}} %[[THIS_PARAM:.+]])
// OGCG: entry:
// OGCG:   %[[THIS_ADDR:.+]] = alloca ptr
// OGCG:   store ptr %[[THIS_PARAM]], ptr %[[THIS_ADDR]]
// OGCG:   %[[THIS_LOAD:.+]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   ret i32 123
// OGCG: }

// OGCG: define dso_local void @_Z4testv()
// OGCG: entry:
// OGCG:   %[[X_VAR:.+]] = alloca i32
// OGCG:   %[[I_VAR:.+]] = alloca {{.*}}
// OGCG:   %[[O_VAR:.+]] = alloca {{.*}}
// OGCG:   store i32 42, ptr %[[X_VAR]]
// OGCG:   %[[INLINE_CALL:.+]] = call noundef i32 @_ZNK15inline_operatorcviEv(ptr {{.*}} %[[I_VAR]])
// OGCG:   store i32 %[[INLINE_CALL]], ptr %[[X_VAR]]
// OGCG:   %[[OUTLINE_CALL:.+]] = call noundef i32 @_ZN20out_of_line_operatorcviEv(ptr {{.*}} %[[O_VAR]])
// OGCG:   store i32 %[[OUTLINE_CALL]], ptr %[[X_VAR]]
// OGCG:   ret void
// OGCG: }

// OGCG: define linkonce_odr noundef i32 @_ZNK15inline_operatorcviEv(ptr {{.*}} %[[INLINE_THIS_PARAM:.+]])
// OGCG: entry:
// OGCG:   %[[INLINE_THIS_ADDR:.+]] = alloca ptr
// OGCG:   store ptr %[[INLINE_THIS_PARAM]], ptr %[[INLINE_THIS_ADDR]]
// OGCG:   %[[INLINE_THIS_LOAD:.+]] = load ptr, ptr %[[INLINE_THIS_ADDR]]
// OGCG:   ret i32 987
// OGCG: }
