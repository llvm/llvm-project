// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

template<typename T>
auto getT() {
  return T{1, 2};
}

struct Type { int a, b, c; };

Type t{1,2,3};

// LLVM: @_ZDC2t12t22t3E = global %struct.Type zeroinitializer
// LLVM: @_ZDC3t113t123t13E = global ptr null
// LLVM: @_ZDC3dt13dt23dt3E = global %struct.DtorType zeroinitializer

auto [t1, t2, t3] = t;
// CIR: cir.global external @_ZDC2t12t22t3E = #cir.zero : !rec_Type
// CIR: cir.func internal private @__cxx_global_var_init{{.*}}()
// CIR:  %[[SB:.*]] = cir.get_global @_ZDC2t12t22t3E : !cir.ptr<!rec_Type>
// CIR:  %[[T:.*]] = cir.get_global @t : !cir.ptr<!rec_Type>
// CIR:  cir.copy %[[T]] to %[[SB]] : !cir.ptr<!rec_Type>

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}@_ZDC2t12t22t3E, ptr {{.*}}@t, i64 12, i1 false)

const auto & [t11, t12, t13] = getT<Type>();
// CIR: cir.global external @_ZDC3t113t123t13E = #cir.ptr<null> : !cir.ptr<!rec_Type>
// CIR: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR:   %[[SB:.*]] = cir.get_global @_ZDC3t113t123t13E : !cir.ptr<!cir.ptr<!rec_Type>>
// CIR:   %[[SB_REF:.*]] = cir.get_global @_ZGRDC3t113t123t13E_ : !cir.ptr<!rec_Type>
// CIR:   %[[GETTCALL:.*]] = cir.call @_Z4getTI4TypeEDav() : () -> !rec_Type
// CIR:   cir.store align(4) %[[GETTCALL]], %[[SB_REF]] : !rec_Type, !cir.ptr<!rec_Type>
// CIR:   cir.store align(8) %[[SB_REF]], %[[SB]] : !cir.ptr<!rec_Type>, !cir.ptr<!cir.ptr<!rec_Type>>

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVMCIR:   %[[GETTCALL:.*]] = call %struct.Type @_Z4getTI4TypeEDav()
// OGCG:      %[[GETTCALL:.*]] = call { i64, i32 } @_Z4getTI4TypeEDav()
// LLVMCIR:   store %struct.Type %[[GETTCALL]], ptr @_ZGRDC3t113t123t13E_, align 4
// OGCG:      store { i64, i32 } %call, ptr %[[COERCED_PTR:.*]],
// OGCG:      call void @llvm.memcpy.p0.p0.i64(ptr align 4 @_ZGRDC3t113t123t13E_, ptr align 8 %[[COERCED_PTR]], i64 12, i1 false)
// LLVM:   store ptr @_ZGRDC3t113t123t13E_, ptr @_ZDC3t113t123t13E, align 8

struct DtorType { int a, b, c; ~DtorType(); };

DtorType dt;
auto [dt1, dt2, dt3] = dt;

// CIR: cir.global external @_ZDC3dt13dt23dt3E = #cir.zero : !rec_DtorType
// CIR: cir.func internal private @__cxx_global_var_init{{.*}}() {
// CIR:   %[[SB:.*]] = cir.get_global @_ZDC3dt13dt23dt3E : !cir.ptr<!rec_DtorType>
// CIR:   %[[DT:.*]] = cir.get_global @dt : !cir.ptr<!rec_DtorType>
// CIR:   cir.copy %[[DT]] to %[[SB]] : !cir.ptr<!rec_DtorType>
// CIR:   %[[SB:.*]] = cir.get_global @_ZDC3dt13dt23dt3E : !cir.ptr<!rec_DtorType>
// CIR:   %[[DTOR_PTR:.*]] = cir.get_global @_ZN8DtorTypeD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_DtorType>)>>
// CIR:   %[[DTOR_PTR_CAST:.*]] = cir.cast bitcast %[[DTOR_PTR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_DtorType>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[SB_VOIDPTR:.*]] = cir.cast bitcast %[[SB]] : !cir.ptr<!rec_DtorType> -> !cir.ptr<!void>
// CIR:   %[[DSO_HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_PTR_CAST]], %[[SB_VOIDPTR]], %[[DSO_HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()

// LLVM: define internal void @__cxx_global_var_init{{.*}}()
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}@_ZDC3dt13dt23dt3E, ptr {{.*}}@dt, i64 12, i1 false)
// LLVM:   call {{.*}} @__cxa_atexit(ptr @_ZN8DtorTypeD1Ev, ptr @_ZDC3dt13dt23dt3E, ptr @__dso_handle)

extern "C" int use() {
  return t1 + t2 + t3 +
         t11 + t12 + t13 +
         dt1 + dt2 + dt3;
  // CIR-LABEL: use()
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC2t12t22t3E : !cir.ptr<!rec_Type>
  // CIR:  cir.get_member %[[GET_GLOB]][0] {name = "a"} : !cir.ptr<!rec_Type> -> !cir.ptr<!s32i>
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC2t12t22t3E : !cir.ptr<!rec_Type>
  // CIR:  cir.get_member %[[GET_GLOB]][1] {name = "b"} : !cir.ptr<!rec_Type> -> !cir.ptr<!s32i>
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC2t12t22t3E : !cir.ptr<!rec_Type>
  // CIR:  cir.get_member %[[GET_GLOB]][2] {name = "c"} : !cir.ptr<!rec_Type> -> !cir.ptr<!s32i>

  // LLVM: load i32, ptr @_ZDC2t12t22t3E, align 4
  // LLVM: load i32, ptr getelementptr inbounds nuw (i8, ptr @_ZDC2t12t22t3E, i64 4), align 4
  // LLVM: load i32, ptr getelementptr inbounds nuw (i8, ptr @_ZDC2t12t22t3E, i64 8), align 4

  // Extra load is because this is a reference.
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC3t113t123t13E : !cir.ptr<!cir.ptr<!rec_Type>>
  // CIR:  %[[LOAD_GLOB:.*]] = cir.load %[[GET_GLOB]] : !cir.ptr<!cir.ptr<!rec_Type>>, !cir.ptr<!rec_Type>
  // CIR:  cir.get_member %[[LOAD_GLOB]][0] {name = "a"} : !cir.ptr<!rec_Type> -> !cir.ptr<!s32i>
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC3t113t123t13E : !cir.ptr<!cir.ptr<!rec_Type>>
  // CIR:  %[[LOAD_GLOB:.*]] = cir.load %[[GET_GLOB]] : !cir.ptr<!cir.ptr<!rec_Type>>, !cir.ptr<!rec_Type>
  // CIR:  cir.get_member %[[LOAD_GLOB]][1] {name = "b"} : !cir.ptr<!rec_Type> -> !cir.ptr<!s32i>
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC3t113t123t13E : !cir.ptr<!cir.ptr<!rec_Type>>
  // CIR:  %[[LOAD_GLOB:.*]] = cir.load %[[GET_GLOB]] : !cir.ptr<!cir.ptr<!rec_Type>>, !cir.ptr<!rec_Type>
  // CIR:  cir.get_member %[[LOAD_GLOB]][2] {name = "c"} : !cir.ptr<!rec_Type> -> !cir.ptr<!s32i>

  // LLVM: %[[LOAD_REF:.*]] = load ptr, ptr @_ZDC3t113t123t13E, align 8
  // LLVM: getelementptr {{.*}}%struct.Type, ptr %[[LOAD_REF]], i32 0, i32 0
  // LLVM: %[[LOAD_REF:.*]] = load ptr, ptr @_ZDC3t113t123t13E, align 8
  // LLVM: getelementptr {{.*}}%struct.Type, ptr %[[LOAD_REF]], i32 0, i32 1
  // LLVM: %[[LOAD_REF:.*]] = load ptr, ptr @_ZDC3t113t123t13E, align 8
  // LLVM: etelementptr {{.*}}%struct.Type, ptr %[[LOAD_REF]], i32 0, i32 2

  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC3dt13dt23dt3E : !cir.ptr<!rec_DtorType>
  // CIR:  cir.get_member %[[GET_GLOB]][0] {name = "a"} : !cir.ptr<!rec_DtorType> -> !cir.ptr<!s32i>
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC3dt13dt23dt3E : !cir.ptr<!rec_DtorType>
  // CIR:  cir.get_member %[[GET_GLOB]][1] {name = "b"} : !cir.ptr<!rec_DtorType> -> !cir.ptr<!s32i>
  // CIR:  %[[GET_GLOB:.*]] = cir.get_global @_ZDC3dt13dt23dt3E : !cir.ptr<!rec_DtorType>
  // CIR:  cir.get_member %[[GET_GLOB]][2] {name = "c"} : !cir.ptr<!rec_DtorType> -> !cir.ptr<!s32i>
 
  // LLVM: load i32, ptr @_ZDC3dt13dt23dt3E, align 4
  // LLVM: load i32, ptr getelementptr inbounds nuw (i8, ptr @_ZDC3dt13dt23dt3E, i64 4), align 4
  // LLVM: load i32, ptr getelementptr inbounds nuw (i8, ptr @_ZDC3dt13dt23dt3E, i64 8), align 4
}

