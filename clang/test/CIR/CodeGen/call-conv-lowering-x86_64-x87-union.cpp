// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=NOMOVE --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=NOMOVE --input-file=%t.ll %s

struct View {
  unsigned long size;
  const char *data;
};

union Value {
  long double ld;
  View view;
  int i;
};

union WideInt {
  long double ld;
  __int128 i128;
};

struct Arg {
  Value value;
  unsigned char type;
};

// CIR-DAG: !rec_anon_struct = !cir.struct<{data !u64i, data !u64i}>
// CIR-DAG: !rec_Value = !cir.union<"Value" {data !cir.long_double<!cir.f80>, data !rec_View, data !s32i}>
// CIR-DAG: !rec_WideInt = !cir.union<"WideInt" {data !cir.long_double<!cir.f80>, data !s128i}>
// CIR-DAG: !rec_Arg = !cir.struct<"Arg" {data !rec_Value, data !u8i}>

// LLVM-DAG: %struct.Arg = type { %union.Value, i8 }
// LLVM-DAG: %union.Value = type { x86_fp80 }
// LLVM-DAG: %union.WideInt = type { x86_fp80 }

Arg ret_arg(const Arg *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @_Z7ret_argPK3Arg(%arg0: !cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Arg, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_Arg> {llvm.noundef} loc({{[^)]+}}))
// CIR:   cir.copy %{{[^ ]+}} align(16) to %arg0 align(16) : !cir.ptr<!rec_Arg>

// LLVM-LABEL: define dso_local void @_Z7ret_argPK3Arg(ptr dead_on_unwind noalias writable sret(%struct.Arg) align 16 %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 32, i1 false)

// NOMOVE-LABEL: define {{.*}}@_Z7ret_argPK3Arg(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

unsigned char take_arg(Arg a) { return a.type; }

// CIR-LABEL: cir.func {{.*}} @_Z8take_arg3Arg(%arg0: !cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.byval = !rec_Arg, llvm.noundef} loc({{[^)]+}})) -> (!u8i {llvm.noundef})
// CIR:   cir.copy %arg0 align(16) to %{{[^ ]+}} align(16) : !cir.ptr<!rec_Arg>

// LLVM-LABEL: define dso_local noundef i8 @_Z8take_arg3Arg(ptr noundef byval(%struct.Arg) align 16 %{{[^,)]+}})
// LLVMCIR:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 32, i1 false)
// OGCG-NOT:  call void @llvm.memcpy

// NOMOVE-LABEL: define {{.*}}@_Z8take_arg3Arg(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

unsigned char call_take_arg(const Arg *p) { return take_arg(*p); }

// CIR-LABEL: cir.func {{.*}} @_Z13call_take_argPK3Arg(
// CIR:   cir.call @_Z8take_arg3Arg(%{{[^,)]+}}) : (!cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.byval = !rec_Arg, llvm.noundef}) -> (!u8i {llvm.noundef})

// LLVM-LABEL: define dso_local noundef i8 @_Z13call_take_argPK3Arg(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[TMP:[^,)]+]], ptr align 16 %{{[^,)]+}}, i64 32, i1 false)
// LLVM:   call noundef i8 @_Z8take_arg3Arg(ptr noundef byval(%struct.Arg) align 16 %[[TMP]])

// NOMOVE-LABEL: define {{.*}}@_Z13call_take_argPK3Arg(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

unsigned char call_arg_ptr(unsigned char (*f)(Arg), const Arg *p) {
  return f(*p);
}

// CIR-LABEL: cir.func {{.*}} @_Z12call_arg_ptrPFh3ArgEPKS_(
// CIR:   cir.call %{{[^(]+}}(%{{[^,)]+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Arg>) -> !u8i>>, !cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.byval = !rec_Arg, llvm.noundef}) -> (!u8i {llvm.noundef})

// LLVM-LABEL: define dso_local noundef i8 @_Z12call_arg_ptrPFh3ArgEPKS_(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[PTMP:[^,)]+]], ptr align 16 %{{[^,)]+}}, i64 32, i1 false)
// LLVM:   call noundef i8 %{{[^(]+}}(ptr noundef byval(%struct.Arg) align 16 %[[PTMP]])

// NOMOVE-LABEL: define {{.*}}@_Z12call_arg_ptrPFh3ArgEPKS_(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void variadic(int n, ...);

void call_variadic(const Arg *a, const Value *v) { variadic(2, *a, *v); }

// CIR-LABEL: cir.func {{.*}} @_Z13call_variadicPK3ArgPK5Value(
// CIR:   cir.call @_Z8variadiciz(%{{[^,)]+}}, %{{[^,)]+}}, %{{[^,)]+}}, %{{[^,)]+}}) : (!s32i {llvm.noundef}, !cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.byval = !rec_Arg, llvm.noundef}, !u64i, !u64i) -> ()

// LLVM-LABEL: define dso_local void @_Z13call_variadicPK3ArgPK5Value(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void (i32, ...) @_Z8variadiciz(i32 noundef 2, ptr noundef byval(%struct.Arg) align 16 %{{[^,)]+}}, i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z13call_variadicPK3ArgPK5Value(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void store_args(Arg *a, int i, const Arg *p) { a[i] = ret_arg(p); }

// CIR-LABEL: cir.func {{.*}} @_Z10store_argsP3ArgiPKS_(
// CIR:   cir.call @_Z7ret_argPK3Arg(%{{[^,)]+}}, %{{[^,)]+}}) : (!cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.sret = !rec_Arg, llvm.writable}, !cir.ptr<!rec_Arg> {llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @_Z10store_argsP3ArgiPKS_(ptr noundef %{{[^,)]+}}, i32 noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @_Z7ret_argPK3Arg(ptr dead_on_unwind writable sret(%struct.Arg) align 16 %{{[^,)]+}}, ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z10store_argsP3ArgiPKS_(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

Value ret_value(const Value *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @_Z9ret_valuePK5Value(%arg0: !cir.ptr<!rec_Value> {llvm.noundef} loc({{[^)]+}})) -> !rec_anon_struct
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_Value>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[RETVIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_Value> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[RETVIEW]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR:   cir.return %[[RET]] : !rec_anon_struct

// LLVM-LABEL: define dso_local { i64, i64 } @_Z9ret_valuePK5Value(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z9ret_valuePK5Value(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

const char *take_value(Value v) { return v.view.data; }

// CIR-LABEL: cir.func {{.*}} @_Z10take_value5Value(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> (!cir.ptr<!s8i> {llvm.noundef})
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[SLOT]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   cir.store %arg0, %[[LOP]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[HIP:.+]] = cir.get_member %[[SLOT]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   cir.store %arg1, %[[HIP]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_Value>
// CIR:   %[[V:.+]] = cir.alloca "v" align(16) init : !cir.ptr<!rec_Value>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[V]] align(16) : !cir.ptr<!rec_Value>

// LLVM-LABEL: define dso_local noundef ptr @_Z10take_value5Value(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z10take_value5Value(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

Value pass_value(Value v) { return v; }

// CIR-LABEL: cir.func {{.*}} @_Z10pass_value5Value(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> !rec_anon_struct
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[SLOT]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   cir.store %arg0, %[[LOP]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[HIP:.+]] = cir.get_member %[[SLOT]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   cir.store %arg1, %[[HIP]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_Value>
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[V:.+]] = cir.alloca "v" align(16) init : !cir.ptr<!rec_Value>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[V]] align(16) : !cir.ptr<!rec_Value>
// CIR:   cir.copy %[[V]] align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[RETVIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_Value> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[RETVIEW]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR:   cir.return %[[RET]] : !rec_anon_struct

// LLVM-LABEL: define dso_local { i64, i64 } @_Z10pass_value5Value(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z10pass_value5Value(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

const char *call_take_value(const Value *p) { return take_value(*p); }

// CIR-LABEL: cir.func {{.*}} @_Z15call_take_valuePK5Value(
// CIR:   %[[TMP:.+]] = cir.alloca "agg.tmp0" align(16) : !cir.ptr<!rec_Value>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_Value> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[VIEW]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[LO:.+]] = cir.load align(16) %[[LOP]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[HIP:.+]] = cir.get_member %[[VIEW]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[HI:.+]] = cir.load align(8) %[[HIP]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.call @_Z10take_value5Value(%[[LO]], %[[HI]]) : (!u64i, !u64i) -> (!cir.ptr<!s8i> {llvm.noundef})

// LLVM-LABEL: define dso_local noundef ptr @_Z15call_take_valuePK5Value(ptr noundef %{{[^,)]+}})
// LLVM:   call noundef ptr @_Z10take_value5Value(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z15call_take_valuePK5Value(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

const char *call_value_ptr(const char *(*f)(Value), const Value *p) {
  return f(*p);
}

// CIR-LABEL: cir.func {{.*}} @_Z14call_value_ptrPFPKc5ValueEPKS1_(
// CIR:   %[[TMP:.+]] = cir.alloca "agg.tmp0" align(16) : !cir.ptr<!rec_Value>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_Value> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[VIEW]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[LO:.+]] = cir.load align(16) %[[LOP]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[HIP:.+]] = cir.get_member %[[VIEW]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[HI:.+]] = cir.load align(8) %[[HIP]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.call %{{[^(]+}}(%[[LO]], %[[HI]]) : (!cir.ptr<!cir.func<(!u64i, !u64i) -> !cir.ptr<!s8i>>>, !u64i, !u64i) -> (!cir.ptr<!s8i> {llvm.noundef})

// LLVM-LABEL: define dso_local noundef ptr @_Z14call_value_ptrPFPKc5ValueEPKS1_(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call noundef ptr %{{[^(]+}}(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z14call_value_ptrPFPKc5ValueEPKS1_(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

Value call_ret_ptr(Value (*f)(const Value *), const Value *p) { return f(p); }

// CIR-LABEL: cir.func {{.*}} @_Z12call_ret_ptrPF5ValuePKS_ES1_(
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[RES:.+]] = cir.call %{{[^(]+}}(%{{[^,)]+}}) : (!cir.ptr<!cir.func<(!cir.ptr<!rec_Value>) -> !rec_anon_struct>>, !cir.ptr<!rec_Value> {llvm.noundef}) -> !rec_anon_struct
// CIR:   cir.store %[[RES]], %[[SLOT]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_Value>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_Value>

// LLVM-LABEL: define dso_local { i64, i64 } @_Z12call_ret_ptrPF5ValuePKS_ES1_(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call { i64, i64 } %{{[^(]+}}(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z12call_ret_ptrPF5ValuePKS_ES1_(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void store_values(Value *a, int i, const Value *p) { a[i] = ret_value(p); }

// CIR-LABEL: cir.func {{.*}} @_Z12store_valuesP5ValueiPKS_(
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[TMP:.+]] = cir.alloca "ref.tmp0" align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[RES:.+]] = cir.call @_Z9ret_valuePK5Value(%{{[^,)]+}}) : (!cir.ptr<!rec_Value> {llvm.noundef}) -> !rec_anon_struct
// CIR:   cir.store %[[RES]], %[[SLOT]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_Value>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_Value>

// LLVM-LABEL: define dso_local void @_Z12store_valuesP5ValueiPKS_(ptr noundef %{{[^,)]+}}, i32 noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call { i64, i64 } @_Z9ret_valuePK5Value(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z12store_valuesP5ValueiPKS_(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

WideInt ret_wide(const WideInt *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @_Z8ret_widePK7WideInt(%arg0: !cir.ptr<!rec_WideInt> {llvm.noundef} loc({{[^)]+}})) -> !s128i
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_WideInt>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_WideInt>
// CIR:   %[[RETVIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_WideInt> -> !cir.ptr<!s128i>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[RETVIEW]] : !cir.ptr<!s128i>, !s128i
// CIR:   cir.return %[[RET]] : !s128i

// LLVMCIR-LABEL: define dso_local i128 @_Z8ret_widePK7WideInt(ptr noundef %{{[^,)]+}})
// OGCG-LABEL:    define dso_local { i64, i64 } @_Z8ret_widePK7WideInt(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z8ret_widePK7WideInt(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

__int128 take_wide(WideInt w) { return w.i128; }

// CIR-LABEL: cir.func {{.*}} @_Z9take_wide7WideInt(%arg0: !s128i loc({{[^)]+}})) -> (!s128i {llvm.noundef})
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!s128i>
// CIR:   cir.store %arg0, %[[SLOT]] : !s128i, !cir.ptr<!s128i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!s128i> -> !cir.ptr<!rec_WideInt>
// CIR:   %[[W:.+]] = cir.alloca "w" align(16) init : !cir.ptr<!rec_WideInt>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[W]] align(16) : !cir.ptr<!rec_WideInt>

// LLVMCIR-LABEL: define dso_local noundef i128 @_Z9take_wide7WideInt(i128 %{{[^,)]+}})
// OGCG-LABEL:    define dso_local noundef i128 @_Z9take_wide7WideInt(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z9take_wide7WideInt(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

__int128 call_take_wide(const WideInt *p) { return take_wide(*p); }

// CIR-LABEL: cir.func {{.*}} @_Z14call_take_widePK7WideInt(
// CIR:   %[[TMP:.+]] = cir.alloca "agg.tmp0" align(16) : !cir.ptr<!rec_WideInt>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_WideInt>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_WideInt> -> !cir.ptr<!s128i>
// CIR:   %[[ARG:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!s128i>, !s128i
// CIR:   cir.call @_Z9take_wide7WideInt(%[[ARG]]) : (!s128i) -> (!s128i {llvm.noundef})

// LLVM-LABEL: define dso_local noundef i128 @_Z14call_take_widePK7WideInt(ptr noundef %{{[^,)]+}})
// LLVMCIR:   call noundef i128 @_Z9take_wide7WideInt(i128 %{{[^,)]+}})
// OGCG:      call noundef i128 @_Z9take_wide7WideInt(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z14call_take_widePK7WideInt(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void store_wides(WideInt *a, int i, const WideInt *p) { a[i] = ret_wide(p); }

// CIR-LABEL: cir.func {{.*}} @_Z11store_widesP7WideIntiPKS_(
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!s128i>
// CIR:   %[[TMP:.+]] = cir.alloca "ref.tmp0" align(16) : !cir.ptr<!rec_WideInt>
// CIR:   %[[RES:.+]] = cir.call @_Z8ret_widePK7WideInt(%{{[^,)]+}}) : (!cir.ptr<!rec_WideInt> {llvm.noundef}) -> !s128i
// CIR:   cir.store %[[RES]], %[[SLOT]] : !s128i, !cir.ptr<!s128i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!s128i> -> !cir.ptr<!rec_WideInt>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_WideInt>

// LLVM-LABEL: define dso_local void @_Z11store_widesP7WideIntiPKS_(ptr noundef %{{[^,)]+}}, i32 noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVMCIR:   call i128 @_Z8ret_widePK7WideInt(ptr noundef %{{[^,)]+}})
// OGCG:      call { i64, i64 } @_Z8ret_widePK7WideInt(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@_Z11store_widesP7WideIntiPKS_(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

Value va_value(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  Value v = __builtin_va_arg(ap, Value);
  __builtin_va_end(ap);
  return v;
}

// CIR-LABEL: cir.func {{.*}} @_Z8va_valueiz(
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[REG:.+]] = cir.alloca "vaarg.reg" align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[ADDR:.+]] = cir.ternary(
// CIR:     %[[REGSLOT:.+]] = cir.cast bitcast %{{[^ ]+}} : !cir.ptr<!u8i> -> !cir.ptr<!rec_Value>
// CIR:     cir.copy %[[REGSLOT]] align(8) to %[[REG]] align(16) : !cir.ptr<!rec_Value>
// CIR:   %[[RES:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_Value>
// CIR:   cir.copy %[[RES]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_Value>

// LLVM-LABEL: define dso_local { i64, i64 } @_Z8va_valueiz(i32 noundef %{{[^,)]+}}, ...)
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 8 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@_Z8va_valueiz(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

WideInt va_wide(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  WideInt w = __builtin_va_arg(ap, WideInt);
  __builtin_va_end(ap);
  return w;
}

// CIR-LABEL: cir.func {{.*}} @_Z7va_wideiz(
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(16) : !cir.ptr<!rec_WideInt>
// CIR:   %[[REG:.+]] = cir.alloca "vaarg.reg" align(16) : !cir.ptr<!rec_WideInt>
// CIR:   %[[ADDR:.+]] = cir.ternary(
// CIR:     %[[REGSLOT:.+]] = cir.cast bitcast %{{[^ ]+}} : !cir.ptr<!u8i> -> !cir.ptr<!rec_WideInt>
// CIR:     cir.copy %[[REGSLOT]] align(8) to %[[REG]] align(16) : !cir.ptr<!rec_WideInt>
// CIR:   %[[RES:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_WideInt>
// CIR:   cir.copy %[[RES]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_WideInt>

// LLVMCIR-LABEL: define dso_local i128 @_Z7va_wideiz(i32 noundef %{{[^,)]+}}, ...)
// OGCG-LABEL:    define dso_local { i64, i64 } @_Z7va_wideiz(i32 noundef %{{[^,)]+}}, ...)
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 8 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@_Z7va_wideiz(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

Arg va_arg_arg(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  Arg a = __builtin_va_arg(ap, Arg);
  __builtin_va_end(ap);
  return a;
}

// CIR-LABEL: cir.func {{.*}} @_Z10va_arg_argiz(%arg0: !cir.ptr<!rec_Arg> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_Arg, llvm.writable} loc({{[^)]+}}), %arg1: !s32i {llvm.noundef} loc({{[^)]+}}), ...)
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(16) : !cir.ptr<!rec_Arg>
// CIR:   %[[RES:.+]] = cir.cast bitcast %{{[^ ]+}} : !cir.ptr<!u8i> -> !cir.ptr<!rec_Arg>
// CIR:   cir.copy %[[RES]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_Arg>
// CIR:   cir.copy %[[TMP]] align(16) to %arg0 align(16) : !cir.ptr<!rec_Arg>

// LLVM-LABEL: define dso_local void @_Z10va_arg_argiz(ptr dead_on_unwind noalias writable sret(%struct.Arg) align 16 %{{[^,)]+}}, i32 noundef %{{[^,)]+}}, ...)
// LLVMCIR:   %[[TMP:.+]] = alloca %struct.Arg, align 16
// LLVMCIR:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[TMP]], ptr align 16 %{{[^,)]+}}, i64 32, i1 false)
// LLVMCIR:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %[[TMP]], i64 32, i1 false)
// OGCG:      %[[AREA:.+]] = call ptr @llvm.ptrmask.p0.i64(
// OGCG:      call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %[[AREA]], i64 32, i1 false)

// NOMOVE-LABEL: define {{.*}}@_Z10va_arg_argiz(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

// A union of only long doubles classifies X87, so it is returned as an
// x86_fp80 read out of the record's memory and passed in memory.
union X87U { long double a, b; };
X87U make_x87u();

X87U ret_x87u(const X87U *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @_Z8ret_x87uPK4X87U(%arg0: !cir.ptr<!rec_X87U> {llvm.noundef} loc({{[^)]+}})) -> !cir.f80
// CIR:   %[[RV:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_X87U>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[RV]] align(16) : !cir.ptr<!rec_X87U>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RV]] : !cir.ptr<!rec_X87U> -> !cir.ptr<!cir.f80>
// CIR:   %[[LD:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!cir.f80>, !cir.f80
// CIR:   cir.return %[[LD]] : !cir.f80

// LLVM-LABEL: define dso_local x86_fp80 @_Z8ret_x87uPK4X87U(ptr noundef %{{[^,)]+}})
// LLVM:   %[[RV:.+]] = alloca %union.X87U, align 16
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[RV]], ptr align 16 %{{[^,)]+}}, i64 16, i1 false)
// LLVM:   %[[LD:.+]] = load x86_fp80, ptr %[[RV]], align 16
// LLVM:   ret x86_fp80 %[[LD]]

// NOMOVE-LABEL: define {{.*}}@_Z8ret_x87uPK4X87U(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* %union\.}}

long double take_x87u(X87U u) { return u.b; }

// CIR-LABEL: cir.func {{.*}} @_Z9take_x87u4X87U(%arg0: !cir.ptr<!rec_X87U> {llvm.align = 16 : i64, llvm.byval = !rec_X87U, llvm.noundef} loc({{[^)]+}})) -> (!cir.long_double<!cir.f80> {llvm.noundef})
// CIR:   %[[U:.+]] = cir.alloca "u" align(16) init : !cir.ptr<!rec_X87U>
// CIR:   cir.copy %arg0 align(16) to %[[U]] align(16) : !cir.ptr<!rec_X87U>

// LLVM-LABEL: define dso_local noundef x86_fp80 @_Z9take_x87u4X87U(ptr noundef byval(%union.X87U) align 16 %{{[^,)]+}})
// LLVMCIR:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)
// OGCG-NOT:  call void @llvm.memcpy
// LLVM:      load x86_fp80, ptr %{{[^,)]+}}, align 16

// NOMOVE-LABEL: define {{.*}}@_Z9take_x87u4X87U(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* %union\.}}

void store_x87u(X87U *q) { *q = make_x87u(); }

// CIR-LABEL: cir.func {{.*}} @_Z10store_x87uP4X87U(
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!cir.f80>
// CIR:   %[[TMP:.+]] = cir.alloca "ref.tmp0" align(16) : !cir.ptr<!rec_X87U>
// CIR:   %[[RES:.+]] = cir.call @_Z9make_x87uv() : () -> !cir.f80
// CIR:   cir.store %[[RES]], %[[SLOT]] : !cir.f80, !cir.ptr<!cir.f80>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!cir.f80> -> !cir.ptr<!rec_X87U>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_X87U>

// LLVM-LABEL: define dso_local void @_Z10store_x87uP4X87U(ptr noundef %{{[^,)]+}})
// LLVMCIR: %[[SLOT:.+]] = alloca x86_fp80, align 16
// LLVM:    %[[TMP:.+]] = alloca %union.X87U, align 16
// LLVM:    %[[RES:.+]] = call x86_fp80 @_Z9make_x87uv()
// LLVMCIR: store x86_fp80 %[[RES]], ptr %[[SLOT]], align 16
// LLVMCIR: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[TMP]], ptr align 16 %[[SLOT]], i64 16, i1 false)
// LLVMCIR: call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %[[TMP]], i64 16, i1 false)
// OGCG:    store x86_fp80 %[[RES]], ptr %[[TMP]], align 16
// OGCG:    call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %[[TMP]], i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@_Z10store_x87uP4X87U(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* %union\.}}
