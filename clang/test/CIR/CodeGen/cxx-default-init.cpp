// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -mconstructor-aliases -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct Pair {
  int a;
  int b;
};

struct ZeroInit {
  int i{};
  Pair p{};
  int  arr[4]{};
  float _Complex c{};
  unsigned bf : 8 {};
  ZeroInit() = default;
};

// CIR: cir.func{{.*}} @_ZN8ZeroInitC2Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_ZeroInit> {{.*}})
// CIR:   %[[ITER:.*]] = cir.alloca {{.*}} ["arrayinit.temp"]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ALLOCA:.*]]
// CIR:   %[[I:.*]] = cir.get_member %[[THIS]][0] {name = "i"}
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[I]]
// CIR:   %[[P:.*]] = cir.get_member %[[THIS]][1] {name = "p"}
// CIR:   %[[P_A:.*]] = cir.get_member %[[P]][0] {name = "a"}
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[P_A]]
// CIR:   %[[P_B:.*]] = cir.get_member %[[P]][1] {name = "b"}
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store{{.*}} %[[ZERO]], %[[P_B]]
// CIR:   %[[ARR:.*]] = cir.get_member %[[THIS]][2] {name = "arr"}
// CIR:   %[[ARR_BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>
// CIR:   cir.store{{.*}} %[[ARR_BEGIN]], %[[ITER]]
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4> : !s64i
// CIR:   %[[END:.*]] = cir.ptr_stride %[[ARR_BEGIN]], %[[FOUR]] : (!cir.ptr<!s32i>, !s64i)
// CIR:   cir.do {
// CIR:     %[[CUR:.*]] = cir.load{{.*}} %[[ITER]]
// CIR:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:     cir.store{{.*}} %[[ZERO]], %[[CUR]]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i)
// CIR:     cir.store{{.*}} %[[NEXT]], %[[ITER]]
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CUR:.*]] = cir.load{{.*}} %[[ITER]]
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[CUR]], %[[END]])
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   %[[C:.*]] = cir.get_member %[[THIS]][3] {name = "c"}
// CIR:   %[[ZERO:.*]] = cir.const #cir.zero : !cir.complex<!cir.float>
// CIR:   cir.store{{.*}} %[[ZERO]], %[[C]]
// CIR:   %[[BF:.*]] = cir.get_member %[[THIS]][4] {name = "bf"}
// CIR:   %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CIR:   %[[BF_VAL:.*]] = cir.set_bitfield{{.*}} (#bfi_bf, %[[BF]] : !cir.ptr<!u8i>, %[[ZERO]] : !u32i)

// LLVM: define{{.*}} void @_ZN8ZeroInitC2Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[ITER:.*]] = alloca ptr
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[I:.*]] = getelementptr %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 0
// LLVM:   store i32 0, ptr %[[I]]
// LLVM:   %[[P:.*]] = getelementptr %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 1
// LLVM:   %[[P_A:.*]] = getelementptr %struct.Pair, ptr %[[P]], i32 0, i32 0
// LLVM:   store i32 0, ptr %[[P_A]]
// LLVM:   %[[P_B:.*]] = getelementptr %struct.Pair, ptr %[[P]], i32 0, i32 1
// LLVM:   store i32 0, ptr %[[P_B]]
// LLVM:   %[[ARR:.*]] = getelementptr %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 2
// LLVM:   %[[ARR_BEGIN:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:   store ptr %[[ARR_BEGIN]], ptr %[[ITER]]
// LLVM:   %[[ARR_END:.*]] = getelementptr i32, ptr %[[ARR_BEGIN]], i64 4
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_COND:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[ARR_END]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   store i32 0, ptr %[[CUR]]
// LLVM:   %[[NEXT:.*]] = getelementptr i32, ptr %[[CUR]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[LOOP_END]]:
// LLVM:   %[[C:.*]] = getelementptr %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 3
// LLVM:   store { float, float } zeroinitializer, ptr %[[C]]
// LLVM:   %[[BF:.*]] = getelementptr %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 4
// LLVM:   store i8 0, ptr %[[BF]]

// OGCG: define{{.*}} void @_ZN8ZeroInitC2Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA:.*]]
// OGCG:   %[[I:.*]] = getelementptr inbounds nuw %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 0
// OGCG:   store i32 0, ptr %[[I]]
// OGCG:   %[[P:.*]] = getelementptr inbounds nuw %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 1
// OGCG:   %[[P_A:.*]] = getelementptr inbounds nuw %struct.Pair, ptr %[[P]], i32 0, i32 0
// OGCG:   store i32 0, ptr %[[P_A]]
// OGCG:   %[[P_B:.*]] = getelementptr inbounds nuw %struct.Pair, ptr %[[P]], i32 0, i32 1
// OGCG:   store i32 0, ptr %[[P_B]]
// OGCG:   %[[ARR:.*]] = getelementptr inbounds nuw %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 2
// OGCG:   %[[ARR_END:.*]] = getelementptr inbounds i32, ptr %[[ARR]], i64 4
// OGCG:   br label %[[LOOP_BODY:.*]]
// OGCG: [[LOOP_BODY]]:
// OGCG:   %[[CUR:.*]] = phi ptr [ %[[ARR]], %entry ], [ %[[NEXT:.*]], %[[LOOP_BODY]] ]
// OGCG:   store i32 0, ptr %[[CUR]]
// OGCG:   %[[NEXT]] = getelementptr inbounds i32, ptr %[[CUR]], i64 1
// OGCG:   %[[CMP:.*]] = icmp eq ptr %[[NEXT]], %[[ARR_END]]
// OGCG:   br i1 %[[CMP]], label %[[LOOP_END:.*]], label %[[LOOP_BODY]]
// OGCG: [[LOOP_END]]:
// OGCG:   %[[C:.*]] = getelementptr inbounds nuw %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 3
// OGCG:   %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C]], i32 0, i32 0
// OGCG:   %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C]], i32 0, i32 1
// OGCG:   store float 0.000000e+00, ptr %[[C_REAL_PTR]]
// OGCG:   store float 0.000000e+00, ptr %[[C_IMAG_PTR]]
// OGCG:   %[[BF:.*]] = getelementptr inbounds nuw %struct.ZeroInit, ptr %[[THIS]], i32 0, i32 4
// OGCG:   store i8 0, ptr %[[BF]]

struct ValueInit {
  int i{1};
  Pair p{2, 3};
  int  arr[4]{4, 5};
  float _Complex c{6.0f, 7.0f};
  unsigned bf : 8 {0xFF};
  ValueInit() = default;
};

// CIR: cir.func{{.*}} @_ZN9ValueInitC2Ev(%[[THIS_ARG:.*]]: !cir.ptr<!rec_ValueInit> {{.*}})
// CIR:   %[[ITER:.*]] = cir.alloca {{.*}} ["arrayinit.temp"]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ALLOCA:.*]]
// CIR:   %[[I:.*]] = cir.get_member %[[THIS]][0] {name = "i"}
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[ONE]], %[[I]]
// CIR:   %[[P:.*]] = cir.get_member %[[THIS]][1] {name = "p"}
// CIR:   %[[P_A:.*]] = cir.get_member %[[P]][0] {name = "a"}
// CIR:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:   cir.store{{.*}} %[[TWO]], %[[P_A]]
// CIR:   %[[P_B:.*]] = cir.get_member %[[P]][1] {name = "b"}
// CIR:   %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:   cir.store{{.*}} %[[THREE]], %[[P_B]]
// CIR:   %[[ARR:.*]] = cir.get_member %[[THIS]][2] {name = "arr"}
// CIR:   %[[ARR_BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR]] : !cir.ptr<!cir.array<!s32i x 4>> -> !cir.ptr<!s32i>
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:   cir.store{{.*}} %[[FOUR]], %[[ARR_BEGIN]]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[SECOND:.*]] = cir.ptr_stride %[[ARR_BEGIN]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i)
// CIR:   %[[FIVE:.*]] = cir.const #cir.int<5> : !s32i
// CIR:   cir.store{{.*}} %[[FIVE]], %[[SECOND]]
// CIR:   %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:   %[[NEXT:.*]] = cir.ptr_stride %[[SECOND]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i)
// CIR:   cir.store{{.*}} %[[NEXT]], %[[ITER]]
// CIR:   %[[FOUR:.*]] = cir.const #cir.int<4> : !s64i
// CIR:   %[[END:.*]] = cir.ptr_stride %[[ARR_BEGIN]], %[[FOUR]] : (!cir.ptr<!s32i>, !s64i)
// CIR:   cir.do {
// CIR:     %[[CUR:.*]] = cir.load{{.*}} %[[ITER]]
// CIR:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR:     cir.store{{.*}} %[[ZERO]], %[[CUR]]
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[ONE]] : (!cir.ptr<!s32i>, !s64i)
// CIR:     cir.store{{.*}} %[[NEXT]], %[[ITER]]
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CUR:.*]] = cir.load{{.*}} %[[ITER]]
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[CUR]], %[[END]])
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   %[[C:.*]] = cir.get_member %[[THIS]][3] {name = "c"}
// CIR:   %[[FOUR_FIVEI:.*]] = cir.const #cir.const_complex<#cir.fp<6.000000e+00> : !cir.float, #cir.fp<7.000000e+00>
// CIR:   cir.store{{.*}} %[[FOUR_FIVEI]], %[[C]]
// CIR:   %[[BF:.*]] = cir.get_member %[[THIS]][4] {name = "bf"}
// CIR:   %[[FF:.*]] = cir.const #cir.int<255> : !s32i
// CIR:   %[[FF_CAST:.*]] = cir.cast integral %[[FF]] : !s32i -> !u32i
// CIR:   %[[BF_VAL:.*]] = cir.set_bitfield{{.*}} (#bfi_bf, %[[BF]] : !cir.ptr<!u8i>, %[[FF_CAST]] : !u32i)

// LLVM: define{{.*}} void @_ZN9ValueInitC2Ev(ptr %[[THIS_ARG:.*]])
// LLVM:   %[[THIS_ALLOCA:.*]] = alloca ptr
// LLVM:   %[[ITER:.*]] = alloca ptr
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA]]
// LLVM:   %[[I:.*]] = getelementptr %struct.ValueInit, ptr %[[THIS]], i32 0, i32 0
// LLVM:   store i32 1, ptr %[[I]]
// LLVM:   %[[P:.*]] = getelementptr %struct.ValueInit, ptr %[[THIS]], i32 0, i32 1
// LLVM:   %[[P_A:.*]] = getelementptr %struct.Pair, ptr %[[P]], i32 0, i32 0
// LLVM:   store i32 2, ptr %[[P_A]]
// LLVM:   %[[P_B:.*]] = getelementptr %struct.Pair, ptr %[[P]], i32 0, i32 1
// LLVM:   store i32 3, ptr %[[P_B]]
// LLVM:   %[[ARR:.*]] = getelementptr %struct.ValueInit, ptr %[[THIS]], i32 0, i32 2
// LLVM:   %[[ARR_BEGIN:.*]] = getelementptr i32, ptr %[[ARR]], i32 0
// LLVM:   store i32 4, ptr %[[ARR_BEGIN]]
// LLVM:   %[[ARR_1:.*]] = getelementptr i32, ptr %[[ARR_BEGIN]], i64 1
// LLVM:   store i32 5, ptr %[[ARR_1]]
// LLVM:   %[[ARR_2:.*]] = getelementptr i32, ptr %[[ARR_1]], i64 1
// LLVM:   store ptr %[[ARR_2]], ptr %[[ITER]]
// LLVM:   %[[ARR_END:.*]] = getelementptr i32, ptr %[[ARR_BEGIN]], i64 4
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_COND:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[ARR_END]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[ITER]]
// LLVM:   store i32 0, ptr %[[CUR]]
// LLVM:   %[[NEXT:.*]] = getelementptr i32, ptr %[[CUR]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[ITER]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[LOOP_END]]:
// LLVM:   %[[C:.*]] = getelementptr %struct.ValueInit, ptr %[[THIS]], i32 0, i32 3
// LLVM:   store { float, float } { float 6.000000e+00, float 7.000000e+00 }, ptr %[[C]]
// LLVM:   %[[BF:.*]] = getelementptr %struct.ValueInit, ptr %[[THIS]], i32 0, i32 4
// LLVM:   store i8 -1, ptr %[[BF]]

// OGCG: define{{.*}} void @_ZN9ValueInitC2Ev(ptr {{.*}} %[[THIS_ARG:.*]])
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ALLOCA:.*]]
// OGCG:   %[[I:.*]] = getelementptr inbounds nuw %struct.ValueInit, ptr %[[THIS]], i32 0, i32 0
// OGCG:   store i32 1, ptr %[[I]]
// OGCG:   %[[P:.*]] = getelementptr inbounds nuw %struct.ValueInit, ptr %[[THIS]], i32 0, i32 1
// OGCG:   %[[P_A:.*]] = getelementptr inbounds nuw %struct.Pair, ptr %[[P]], i32 0, i32 0
// OGCG:   store i32 2, ptr %[[P_A]]
// OGCG:   %[[P_B:.*]] = getelementptr inbounds nuw %struct.Pair, ptr %[[P]], i32 0, i32 1
// OGCG:   store i32 3, ptr %[[P_B]]
// OGCG:   %[[ARR:.*]] = getelementptr inbounds nuw %struct.ValueInit, ptr %[[THIS]], i32 0, i32 2
// OGCG:   store i32 4, ptr %[[ARR]]
// OGCG:   %[[ARR_1:.*]] = getelementptr inbounds i32, ptr %[[ARR]], i64 1
// OGCG:   store i32 5, ptr %[[ARR_1]]
// OGCG:   %[[ARR_BEGIN:.*]] = getelementptr inbounds i32, ptr %[[ARR]], i64 2
// OGCG:   %[[ARR_END:.*]] = getelementptr inbounds i32, ptr %[[ARR]], i64 4
// OGCG:   br label %[[LOOP_BODY:.*]]
// OGCG: [[LOOP_BODY]]:
// OGCG:   %[[CUR:.*]] = phi ptr [ %[[ARR_BEGIN]], %entry ], [ %[[NEXT:.*]], %[[LOOP_BODY]] ]
// OGCG:   store i32 0, ptr %[[CUR]]
// OGCG:   %[[NEXT]] = getelementptr inbounds i32, ptr %[[CUR]], i64 1
// OGCG:   %[[CMP:.*]] = icmp eq ptr %[[NEXT]], %[[ARR_END]]
// OGCG:   br i1 %[[CMP]], label %[[LOOP_END:.*]], label %[[LOOP_BODY]]
// OGCG: [[LOOP_END]]:
// OGCG:   %[[C:.*]] = getelementptr inbounds nuw %struct.ValueInit, ptr %[[THIS]], i32 0, i32 3
// OGCG:   %[[C_REAL_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C]], i32 0, i32 0
// OGCG:   %[[C_IMAG_PTR:.*]] = getelementptr inbounds nuw { float, float }, ptr %[[C]], i32 0, i32 1
// OGCG:   store float 6.000000e+00, ptr %[[C_REAL_PTR]]
// OGCG:   store float 7.000000e+00, ptr %[[C_IMAG_PTR]]
// OGCG:   %[[BF:.*]] = getelementptr inbounds nuw %struct.ValueInit, ptr %[[THIS]], i32 0, i32 4
// OGCG:   store i8 -1, ptr %[[BF]]

void use_structs() {
  ZeroInit zi;
  ValueInit vi;
}
