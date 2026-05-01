// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVM-BOTH --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG,LLVM-BOTH --input-file=%t.ll %s

struct HasMem { int x;};

// CIR-DAG: cir.global "private" constant linkonce_odr comdat @_ZTAXtl6HasMemLi3EEE = #cir.const_record<{#cir.int<3> : !s32i}> : !rec_HasMem
// CIR-DAG: cir.global external @ptr = #cir.global_view<@_ZTAXtl6HasMemLi3EEE> : !cir.ptr<!rec_HasMem>

// LLVM-BOTH-DAG: @_ZTAXtl6HasMemLi1EEE = linkonce_odr constant %struct.HasMem { i32 1 }, comdat
// LLVM-BOTH-DAG: @_ZTAXtl6HasMemLi2EEE = linkonce_odr constant %struct.HasMem { i32 2 }, comdat
// LLVM-BOTH-DAG: @_ZTAXtl6HasMemLi3EEE = linkonce_odr constant %struct.HasMem { i32 3 }, comdat
// LLVM-BOTH-DAG: @ptr = global ptr @_ZTAXtl6HasMemLi3EEE

template <HasMem m>
constexpr const HasMem *get_ptr() { return &m; }

const auto *ptr = get_ptr<HasMem{3}>();

// CIR-DAG: cir.global "private" constant linkonce_odr comdat @_ZTAXtl6HasMemLi1EEE = #cir.const_record<{#cir.int<1> : !s32i}> : !rec_HasMem
// CIR-DAG: cir.global "private" constant linkonce_odr comdat @_ZTAXtl6HasMemLi2EEE = #cir.const_record<{#cir.int<2> : !s32i}> : !rec_HasMem

template<HasMem m>
int get_x() { return m.x; }
// CIR-LABEL: cir.func {{.*}}@_Z5get_xIXtl6HasMemLi1EEEEiv()
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// Note: This is emitted as an 'ignored' expression, but we cannot omit it,
// since it might be used in an example like below.
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZTAXtl6HasMemLi1EEE : !cir.ptr<!rec_HasMem>
// CIR: %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store %[[ONE]], %[[RET_ALLOCA]] : !s32i, !cir.ptr<!s32i>
//
// LLVM-BOTH-LABEL: define {{.*}}@_Z5get_xIXtl6HasMemLi1EEEEiv()
// LLVM: %[[RET_ALLOCA:.*]] = alloca i32
// LLVM: store i32 1, ptr %[[RET_ALLOCA]]
// LLVM: %[[LOAD_RET:.*]] = load i32, ptr %[[RET_ALLOCA]]
// LLVM: ret i32 %[[LOAD_RET]]
// OGCG: ret i32 1

template<HasMem m>
HasMem get_m() { return m; }
// CIR-LABEL: cir.func {{.*}}@_Z5get_mIXtl6HasMemLi2EEEES0_v()
// CIR: %[[RET_ALLOCA:.*]] = cir.alloca !rec_HasMem, !cir.ptr<!rec_HasMem>, ["__retval"] {alignment = 4 : i64}
// CIR: %[[GET_GLOB:.*]] = cir.get_global @_ZTAXtl6HasMemLi2EEE : !cir.ptr<!rec_HasMem>
// CIR: cir.copy %[[GET_GLOB]] to %[[RET_ALLOCA]] : !cir.ptr<!rec_HasMem>
// 
// LLVM-BOTH-LABEL: define {{.*}}@_Z5get_mIXtl6HasMemLi2EEEES0_v()
// LLVM-BOTH: %[[RET_ALLOCA:.*]] = alloca %struct.HasMem
// LLVM-BOTH: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[RET_ALLOCA]], ptr {{.*}}@_ZTAXtl6HasMemLi2EEE, i64 4, i1 false)
// LLVM: %[[LOAD_RET:.*]] = load %struct.HasMem, ptr %[[RET_ALLOCA]]
// LLVM: ret %struct.HasMem %[[LOAD_RET]]

// OGCG: %[[COERCE:.*]] = getelementptr{{.*}} %struct.HasMem, ptr %[[RET_ALLOCA]], i32 0, i32 0
// OGCG: %[[TO_RET:.*]] = load i32, ptr %[[COERCE]]
// OGCG: ret i32 %[[TO_RET]]

void caller() {
  get_x<{1}>();
  get_m<{2}>();
}
