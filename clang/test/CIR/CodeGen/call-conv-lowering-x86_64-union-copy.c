// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: FileCheck --check-prefix=NOMOVE --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s
// RUN: FileCheck --check-prefix=NOMOVE --input-file=%t.ll %s

typedef long double ld1 __attribute__((vector_size(16)));

union H {
  struct {
    char a;
    int b;
  } s;
  int c[2];
};

union U {
  long double ld;
  __int128 i;
};

union U2 {
  long double ld;
  struct {
    unsigned long n;
    const char *p;
  } v;
};

union V {
  ld1 v;
  long double ld;
};

union VI {
  ld1 v;
  long l;
};

union C {
  _Complex long double c;
  long double ld;
};

union M {
  long double ld;
  char buf[32];
};

struct S16 {
  int a, b, c, d;
};

// CIR-DAG: !rec_H = !cir.union<"H" {data !rec_anon2E0, data !cir.array<!s32i x 2>}>
// CIR-DAG: !rec_U = !cir.union<"U" {data !cir.long_double<!cir.f80>, data !s128i}>
// CIR-DAG: !rec_U2 = !cir.union<"U2" {data !cir.long_double<!cir.f80>, data !rec_anon2E1}>
// CIR-DAG: !rec_V = !cir.union<"V" {data !cir.vector<1 x !cir.long_double<!cir.f80>>, data !cir.long_double<!cir.f80>}>
// CIR-DAG: !rec_VI = !cir.union<"VI" {data !cir.vector<1 x !cir.long_double<!cir.f80>>, data !s64i}>
// CIR-DAG: !rec_C = !cir.union<"C" {data !cir.complex<!cir.long_double<!cir.f80>>, data !cir.long_double<!cir.f80>}>
// CIR-DAG: !rec_M = !cir.union<"M" {data !cir.long_double<!cir.f80>, data !cir.array<!s8i x 32>}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_anon_struct = !cir.struct<{data !u64i, data !u64i}>
// CIR-DAG: !rec_anon_struct1 = !cir.struct<{data !u64i, data !cir.double}>
// CIR-DAG: !rec_S16 = !cir.struct<"S16" {data !s32i, data !s32i, data !s32i, data !s32i}>

// LLVM-DAG: %union.H = type { %struct.anon{{(\.[0-9]+)?}} }
// LLVM-DAG: %union.U = type { x86_fp80 }
// LLVM-DAG: %union.U2 = type { x86_fp80 }
// LLVM-DAG: %union.V = type { <1 x x86_fp80> }
// LLVM-DAG: %union.VI = type { <1 x x86_fp80> }
// LLVM-DAG: %union.C = type { { x86_fp80, x86_fp80 } }
// LLVM-DAG: %union.M = type { x86_fp80, [16 x i8] }
// LLVM-DAG: %struct.S16 = type { i32, i32, i32, i32 }

// The storage type of H is a struct with padding after s.a.
union H retH(const union H *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @retH(%arg0: !cir.ptr<!rec_H> {llvm.noundef} loc({{[^)]+}})) -> !u64i
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(4) : !cir.ptr<!rec_H>
// CIR:   cir.copy %{{[^ ]+}} align(4) to %[[RETVAL]] align(4) : !cir.ptr<!rec_H>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_H> -> !cir.ptr<!u64i>
// CIR:   %[[RET:.+]] = cir.load align(4) %[[VIEW]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.return %[[RET]] : !u64i

// LLVM-LABEL: define dso_local i64 @retH(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{[^,)]+}}, ptr align 4 %{{[^,)]+}}, i64 8, i1 false)
// LLVM:   %[[RET:.+]] = load i64, ptr %{{[^,)]+}}, align 4
// LLVM:   ret i64 %[[RET]]

// NOMOVE-LABEL: define {{.*}}@retH(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

int takeH(union H h) { return h.c[0]; }

// CIR-LABEL: cir.func {{.*}} @takeH(%arg0: !u64i loc({{[^)]+}})) -> !s32i
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR:   cir.store %arg0, %[[SLOT]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_H>
// CIR:   %[[H:.+]] = cir.alloca "h" align(4) init : !cir.ptr<!rec_H>
// CIR:   cir.copy %[[VIEW]] align(8) to %[[H]] align(4) : !cir.ptr<!rec_H>

// LLVM-LABEL: define dso_local i32 @takeH(i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@takeH(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

int bindH(const union H *p) {
  union H d = retH(p);
  return d.c[0];
}

// CIR-LABEL: cir.func {{.*}} @bindH(%arg0: !cir.ptr<!rec_H> {llvm.noundef} loc({{[^)]+}})) -> !s32i
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR:   %[[D:.+]] = cir.alloca "d" align(4) init : !cir.ptr<!rec_H>
// CIR:   %[[RES:.+]] = cir.call @retH(%{{[^)]+}}) : (!cir.ptr<!rec_H> {llvm.noundef}) -> !u64i
// CIR:   cir.store %[[RES]], %[[SLOT]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_H>
// CIR:   cir.copy %[[VIEW]] align(8) to %[[D]] align(4) : !cir.ptr<!rec_H>

// LLVM-LABEL: define dso_local i32 @bindH(ptr noundef %{{[^,)]+}})
// LLVM:   call i64 @retH(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@bindH(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void storeH(union H *q, const union H *p) { *q = retH(p); }

// CIR-LABEL: cir.func {{.*}} @storeH(%arg0: !cir.ptr<!rec_H> {llvm.noundef} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_H> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR:   %[[Q:.+]] = cir.load deref align(8) %{{[^ ]+}} : !cir.ptr<!cir.ptr<!rec_H>>, !cir.ptr<!rec_H>
// CIR:   %[[RES:.+]] = cir.call @retH(%{{[^)]+}}) : (!cir.ptr<!rec_H> {llvm.noundef}) -> !u64i
// CIR:   cir.store %[[RES]], %[[SLOT]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_H>
// CIR:   cir.copy %[[VIEW]] align(8) to %[[Q]] align(4) : !cir.ptr<!rec_H>

// LLVM-LABEL: define dso_local void @storeH(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call i64 @retH(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@storeH(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

int passH(const union H *p) { return takeH(*p); }

// CIR-LABEL: cir.func {{.*}} @passH(%arg0: !cir.ptr<!rec_H> {llvm.noundef} loc({{[^)]+}})) -> !s32i
// CIR:   %[[TMP:.+]] = cir.alloca "coerce" align(8) : !cir.ptr<!rec_H>
// CIR:   cir.copy %{{[^ ]+}} align(4) to %[[TMP]] align(8) : !cir.ptr<!rec_H>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_H> -> !cir.ptr<!u64i>
// CIR:   %[[ARG:.+]] = cir.load align(8) %[[VIEW]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.call @takeH(%[[ARG]]) : (!u64i) -> !s32i

// LLVM-LABEL: define dso_local i32 @passH(ptr noundef %{{[^,)]+}})
// LLVM:   call i32 @takeH(i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passH(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union H vaH(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  union H h = __builtin_va_arg(ap, union H);
  __builtin_va_end(ap);
  return h;
}

// CIR-LABEL: cir.func {{.*}} @vaH(%arg0: !s32i {llvm.noundef} loc({{[^)]+}}), ...) -> !u64i
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(4) init : !cir.ptr<!rec_H>
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(4) : !cir.ptr<!rec_H>
// CIR:   %[[ADDR:.+]] = cir.ternary(
// CIR:   %[[SRC:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_H>
// CIR:   cir.copy %[[SRC]] align(4) to %[[TMP]] align(4) : !cir.ptr<!rec_H>
// CIR:   cir.copy %[[TMP]] align(4) to %[[RETVAL]] align(4) : !cir.ptr<!rec_H>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_H> -> !cir.ptr<!u64i>
// CIR:   %[[RET:.+]] = cir.load align(4) %[[VIEW]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.return %[[RET]] : !u64i

// LLVM-LABEL: define dso_local i64 @vaH(i32 noundef %{{[^,)]+}}, ...)
// LLVM:   icmp ule i32 %{{.+}}, 40
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{[^,)]+}}, ptr align 4 %{{[^,)]+}}, i64 8, i1 false)

// NOMOVE-LABEL: define {{.*}}@vaH(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

// The storage type of U is a long double that stores 10 of its 16 bytes.
union U retU(const union U *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @retU(%arg0: !cir.ptr<!rec_U> {llvm.noundef} loc({{[^)]+}})) -> !s128i
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_U>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_U>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_U> -> !cir.ptr<!s128i>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!s128i>, !s128i
// CIR:   cir.return %[[RET]] : !s128i

// LLVMCIR-LABEL: define dso_local i128 @retU(ptr noundef %{{[^,)]+}})
// OGCG-LABEL:    define dso_local { i64, i64 } @retU(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@retU(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

__int128 takeU(union U u) { return u.i; }

// CIR-LABEL: cir.func {{.*}} @takeU(%arg0: !s128i loc({{[^)]+}})) -> !s128i
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!s128i>
// CIR:   cir.store %arg0, %[[SLOT]] : !s128i, !cir.ptr<!s128i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!s128i> -> !cir.ptr<!rec_U>
// CIR:   %[[U:.+]] = cir.alloca "u" align(16) init : !cir.ptr<!rec_U>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[U]] align(16) : !cir.ptr<!rec_U>

// LLVMCIR-LABEL: define dso_local i128 @takeU(i128 %{{[^,)]+}})
// OGCG-LABEL:    define dso_local i128 @takeU(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@takeU(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

__int128 bindU(const union U *p) {
  union U d = retU(p);
  return d.i;
}

// CIR-LABEL: cir.func {{.*}} @bindU(%arg0: !cir.ptr<!rec_U> {llvm.noundef} loc({{[^)]+}})) -> !s128i
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!s128i>
// CIR:   %[[D:.+]] = cir.alloca "d" align(16) init : !cir.ptr<!rec_U>
// CIR:   %[[RES:.+]] = cir.call @retU(%{{[^)]+}}) : (!cir.ptr<!rec_U> {llvm.noundef}) -> !s128i
// CIR:   cir.store %[[RES]], %[[SLOT]] : !s128i, !cir.ptr<!s128i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!s128i> -> !cir.ptr<!rec_U>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[D]] align(16) : !cir.ptr<!rec_U>

// LLVM-LABEL: define dso_local i128 @bindU(ptr noundef %{{[^,)]+}})
// LLVMCIR:   call i128 @retU(ptr noundef %{{[^,)]+}})
// OGCG:      call { i64, i64 } @retU(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@bindU(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void storeU(union U *q, const union U *p) { *q = retU(p); }

// CIR-LABEL: cir.func {{.*}} @storeU(%arg0: !cir.ptr<!rec_U> {llvm.noundef} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_U> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!s128i>
// CIR:   %[[Q:.+]] = cir.load deref align(8) %{{[^ ]+}} : !cir.ptr<!cir.ptr<!rec_U>>, !cir.ptr<!rec_U>
// CIR:   %[[RES:.+]] = cir.call @retU(%{{[^)]+}}) : (!cir.ptr<!rec_U> {llvm.noundef}) -> !s128i
// CIR:   cir.store %[[RES]], %[[SLOT]] : !s128i, !cir.ptr<!s128i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!s128i> -> !cir.ptr<!rec_U>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[Q]] align(16) : !cir.ptr<!rec_U>

// LLVM-LABEL: define dso_local void @storeU(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVMCIR:   call i128 @retU(ptr noundef %{{[^,)]+}})
// OGCG:      call { i64, i64 } @retU(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@storeU(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

__int128 passU(const union U *p) { return takeU(*p); }

// CIR-LABEL: cir.func {{.*}} @passU(%arg0: !cir.ptr<!rec_U> {llvm.noundef} loc({{[^)]+}})) -> !s128i
// CIR:   %[[TMP:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_U>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_U>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_U> -> !cir.ptr<!s128i>
// CIR:   %[[ARG:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!s128i>, !s128i
// CIR:   cir.call @takeU(%[[ARG]]) : (!s128i) -> !s128i

// LLVM-LABEL: define dso_local i128 @passU(ptr noundef %{{[^,)]+}})
// LLVMCIR:   call i128 @takeU(i128 %{{[^,)]+}})
// OGCG:      call i128 @takeU(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passU(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union U vaU(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  union U u = __builtin_va_arg(ap, union U);
  __builtin_va_end(ap);
  return u;
}

// CIR-LABEL: cir.func {{.*}} @vaU(%arg0: !s32i {llvm.noundef} loc({{[^)]+}}), ...) -> !s128i
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) init : !cir.ptr<!rec_U>
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(16) : !cir.ptr<!rec_U>
// CIR:   %[[REG:.+]] = cir.alloca "vaarg.reg" align(16) : !cir.ptr<!rec_U>
// CIR:   %[[ADDR:.+]] = cir.ternary(
// CIR:     %[[REGSLOT:.+]] = cir.cast bitcast %{{[^ ]+}} : !cir.ptr<!u8i> -> !cir.ptr<!rec_U>
// CIR:     cir.copy %[[REGSLOT]] align(8) to %[[REG]] align(16) : !cir.ptr<!rec_U>
// CIR:   %[[SRC:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_U>
// CIR:   cir.copy %[[SRC]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_U>
// CIR:   cir.copy %[[TMP]] align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_U>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_U> -> !cir.ptr<!s128i>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!s128i>, !s128i
// CIR:   cir.return %[[RET]] : !s128i

// LLVMCIR-LABEL: define dso_local i128 @vaU(i32 noundef %{{[^,)]+}}, ...)
// OGCG-LABEL:    define dso_local { i64, i64 } @vaU(i32 noundef %{{[^,)]+}}, ...)
// LLVM:   icmp ule i32 %{{.+}}, 32
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 8 %{{[^,)]+}}, i64 16, i1 false)
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@vaU(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

// The storage type of U2 is a long double.  The struct member spans all 16
// bytes.
union U2 retU2(const union U2 *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @retU2(%arg0: !cir.ptr<!rec_U2> {llvm.noundef} loc({{[^)]+}})) -> !rec_anon_struct
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_U2>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_U2>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_U2> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR:   cir.return %[[RET]] : !rec_anon_struct

// LLVM-LABEL: define dso_local { i64, i64 } @retU2(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)
// LLVM:   %[[RET:.+]] = load { i64, i64 }, ptr %{{[^,)]+}}, align 16
// LLVM:   ret { i64, i64 } %[[RET]]

// NOMOVE-LABEL: define {{.*}}@retU2(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

const char *takeU2(union U2 u) { return u.v.p; }

// CIR-LABEL: cir.func {{.*}} @takeU2(%arg0: !u64i loc({{[^)]+}}), %arg1: !u64i loc({{[^)]+}})) -> !cir.ptr<!s8i>
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[SLOT]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   cir.store %arg0, %[[LOP]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[HIP:.+]] = cir.get_member %[[SLOT]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   cir.store %arg1, %[[HIP]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_U2>
// CIR:   %[[U:.+]] = cir.alloca "u" align(16) init : !cir.ptr<!rec_U2>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[U]] align(16) : !cir.ptr<!rec_U2>

// LLVM-LABEL: define dso_local ptr @takeU2(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@takeU2(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

const char *bindU2(const union U2 *p) {
  union U2 d = retU2(p);
  return d.v.p;
}

// CIR-LABEL: cir.func {{.*}} @bindU2(%arg0: !cir.ptr<!rec_U2> {llvm.noundef} loc({{[^)]+}})) -> !cir.ptr<!s8i>
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[D:.+]] = cir.alloca "d" align(16) init : !cir.ptr<!rec_U2>
// CIR:   %[[RES:.+]] = cir.call @retU2(%{{[^)]+}}) : (!cir.ptr<!rec_U2> {llvm.noundef}) -> !rec_anon_struct
// CIR:   cir.store %[[RES]], %[[SLOT]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_U2>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[D]] align(16) : !cir.ptr<!rec_U2>

// LLVM-LABEL: define dso_local ptr @bindU2(ptr noundef %{{[^,)]+}})
// LLVM:   call { i64, i64 } @retU2(ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@bindU2(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void storeU2(union U2 *q, const union U2 *p) { *q = retU2(p); }

// CIR-LABEL: cir.func {{.*}} @storeU2(%arg0: !cir.ptr<!rec_U2> {llvm.noundef} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_U2> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct>
// CIR:   %[[Q:.+]] = cir.load deref align(8) %{{[^ ]+}} : !cir.ptr<!cir.ptr<!rec_U2>>, !cir.ptr<!rec_U2>
// CIR:   %[[RES:.+]] = cir.call @retU2(%{{[^)]+}}) : (!cir.ptr<!rec_U2> {llvm.noundef}) -> !rec_anon_struct
// CIR:   cir.store %[[RES]], %[[SLOT]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!rec_U2>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[Q]] align(16) : !cir.ptr<!rec_U2>

// LLVM-LABEL: define dso_local void @storeU2(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call { i64, i64 } @retU2(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@storeU2(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

const char *passU2(const union U2 *p) { return takeU2(*p); }

// CIR-LABEL: cir.func {{.*}} @passU2(%arg0: !cir.ptr<!rec_U2> {llvm.noundef} loc({{[^)]+}})) -> !cir.ptr<!s8i>
// CIR:   %[[TMP:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_U2>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_U2>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_U2> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[VIEW]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[LO:.+]] = cir.load %[[LOP]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[HIP:.+]] = cir.get_member %[[VIEW]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[HI:.+]] = cir.load %[[HIP]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.call @takeU2(%[[LO]], %[[HI]]) : (!u64i, !u64i) -> !cir.ptr<!s8i>

// LLVM-LABEL: define dso_local ptr @passU2(ptr noundef %{{[^,)]+}})
// LLVM:   call ptr @takeU2(i64 %{{[^,)]+}}, i64 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passU2(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union U2 vaU2(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  union U2 u = __builtin_va_arg(ap, union U2);
  __builtin_va_end(ap);
  return u;
}

// CIR-LABEL: cir.func {{.*}} @vaU2(%arg0: !s32i {llvm.noundef} loc({{[^)]+}}), ...) -> !rec_anon_struct
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) init : !cir.ptr<!rec_U2>
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(16) : !cir.ptr<!rec_U2>
// CIR:   %[[REG:.+]] = cir.alloca "vaarg.reg" align(16) : !cir.ptr<!rec_U2>
// CIR:   %[[ADDR:.+]] = cir.ternary(
// CIR:     %[[REGSLOT:.+]] = cir.cast bitcast %{{[^ ]+}} : !cir.ptr<!u8i> -> !cir.ptr<!rec_U2>
// CIR:     cir.copy %[[REGSLOT]] align(8) to %[[REG]] align(16) : !cir.ptr<!rec_U2>
// CIR:   %[[SRC:.+]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_U2>
// CIR:   cir.copy %[[SRC]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_U2>
// CIR:   cir.copy %[[TMP]] align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_U2>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_U2> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[VIEW]] : !cir.ptr<!rec_anon_struct>, !rec_anon_struct
// CIR:   cir.return %[[RET]] : !rec_anon_struct

// LLVM-LABEL: define dso_local { i64, i64 } @vaU2(i32 noundef %{{[^,)]+}}, ...)
// LLVM:   icmp ule i32 %{{.+}}, 32
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 8 %{{[^,)]+}}, i64 16, i1 false)
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@vaU2(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union V takeV(union V u) { return u; }

// CIR-LABEL: cir.func {{.*}} @takeV(%arg0: !cir.ptr<!rec_V> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_V, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_V> {llvm.align = 16 : i64, llvm.byval = !rec_V, llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[U:.+]] = cir.alloca "u" align(16) init : !cir.ptr<!rec_V>
// CIR:   cir.copy %arg1 align(16) to %[[U]] align(16) : !cir.ptr<!rec_V>
// CIR:   cir.copy %[[U]] align(16) to %arg0 align(16) : !cir.ptr<!rec_V>

// LLVM-LABEL: define dso_local void @takeV(ptr dead_on_unwind noalias writable sret(%union.V) align 16 %{{[^,)]+}}, ptr noundef byval(%union.V) align 16 %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@takeV(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union V passV(const union V *p) { return takeV(*p); }

// CIR-LABEL: cir.func {{.*}} @passV(%arg0: !cir.ptr<!rec_V> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_V, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_V> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[BYVAL:.+]] = cir.alloca "byval" align(16) : !cir.ptr<!rec_V>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[BYVAL]] align(16) : !cir.ptr<!rec_V>
// CIR:   cir.call @takeV(%arg0, %[[BYVAL]]) : (!cir.ptr<!rec_V> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.sret = !rec_V, llvm.writable}, !cir.ptr<!rec_V> {llvm.align = 16 : i64, llvm.byval = !rec_V, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @passV(ptr dead_on_unwind noalias writable sret(%union.V) align 16 %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @takeV(ptr dead_on_unwind writable sret(%union.V) align 16 %{{[^,)]+}}, ptr noundef byval(%union.V) align 16 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passV(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union VI takeVI(union VI u) { return u; }

// CIR-LABEL: cir.func {{.*}} @takeVI(%arg0: !u64i loc({{[^)]+}}), %arg1: !cir.double loc({{[^)]+}})) -> !rec_anon_struct1
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct1>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!rec_VI>
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_VI>
// CIR:   %[[U:.+]] = cir.alloca "u" align(16) init : !cir.ptr<!rec_VI>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[U]] align(16) : !cir.ptr<!rec_VI>
// CIR:   cir.copy %[[U]] align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_VI>
// CIR:   %[[RETVIEW:.+]] = cir.cast bitcast %[[RETVAL]] : !cir.ptr<!rec_VI> -> !cir.ptr<!rec_anon_struct1>
// CIR:   %[[RET:.+]] = cir.load align(16) %[[RETVIEW]] : !cir.ptr<!rec_anon_struct1>, !rec_anon_struct1
// CIR:   cir.return %[[RET]] : !rec_anon_struct1

// LLVM-LABEL: define dso_local { i64, double } @takeVI(i64 %{{[^,)]+}}, double %{{[^,)]+}})
// LLVM:   %[[RET:.+]] = load { i64, double }, ptr %{{[^,)]+}}, align 16
// LLVM:   ret { i64, double } %[[RET]]

// NOMOVE-LABEL: define {{.*}}@takeVI(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union VI passVI(const union VI *p) { return takeVI(*p); }

// CIR-LABEL: cir.func {{.*}} @passVI(%arg0: !cir.ptr<!rec_VI> {llvm.noundef} loc({{[^)]+}})) -> !rec_anon_struct1
// CIR:   %[[RSLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_anon_struct1>
// CIR:   %[[TMP:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!rec_VI>
// CIR:   %[[RETVAL:.+]] = cir.alloca "__retval" align(16) : !cir.ptr<!rec_VI>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_VI>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_VI> -> !cir.ptr<!rec_anon_struct1>
// CIR:   %[[LOP:.+]] = cir.get_member %[[VIEW]][0] {name = ""} : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!u64i>
// CIR:   %[[LO:.+]] = cir.load %[[LOP]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[HIP:.+]] = cir.get_member %[[VIEW]][1] {name = ""} : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!cir.double>
// CIR:   %[[HI:.+]] = cir.load %[[HIP]] : !cir.ptr<!cir.double>, !cir.double
// CIR:   %[[RES:.+]] = cir.call @takeVI(%[[LO]], %[[HI]]) : (!u64i, !cir.double) -> !rec_anon_struct1
// CIR:   cir.store %[[RES]], %[[RSLOT]] : !rec_anon_struct1, !cir.ptr<!rec_anon_struct1>
// CIR:   %[[RVIEW:.+]] = cir.cast bitcast %[[RSLOT]] : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!rec_VI>
// CIR:   cir.copy %[[RVIEW]] align(16) to %[[RETVAL]] align(16) : !cir.ptr<!rec_VI>

// LLVM-LABEL: define dso_local { i64, double } @passVI(ptr noundef %{{[^,)]+}})
// LLVM:   call { i64, double } @takeVI(i64 %{{[^,)]+}}, double %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passVI(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union C takeC(union C u) { return u; }

// CIR-LABEL: cir.func {{.*}} @takeC(%arg0: !cir.ptr<!rec_C> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_C, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_C> {llvm.align = 16 : i64, llvm.byval = !rec_C, llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[U:.+]] = cir.alloca "u" align(16) init : !cir.ptr<!rec_C>
// CIR:   cir.copy %arg1 align(16) to %[[U]] align(16) : !cir.ptr<!rec_C>
// CIR:   cir.copy %[[U]] align(16) to %arg0 align(16) : !cir.ptr<!rec_C>

// LLVM-LABEL: define dso_local void @takeC(ptr dead_on_unwind noalias writable sret(%union.C) align 16 %{{[^,)]+}}, ptr noundef byval(%union.C) align 16 %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 32, i1 false)

// NOMOVE-LABEL: define {{.*}}@takeC(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union C passC(const union C *p) { return takeC(*p); }

// CIR-LABEL: cir.func {{.*}} @passC(%arg0: !cir.ptr<!rec_C> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_C, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_C> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[BYVAL:.+]] = cir.alloca "byval" align(16) : !cir.ptr<!rec_C>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[BYVAL]] align(16) : !cir.ptr<!rec_C>
// CIR:   cir.call @takeC(%arg0, %[[BYVAL]]) : (!cir.ptr<!rec_C> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.sret = !rec_C, llvm.writable}, !cir.ptr<!rec_C> {llvm.align = 16 : i64, llvm.byval = !rec_C, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @passC(ptr dead_on_unwind noalias writable sret(%union.C) align 16 %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @takeC(ptr dead_on_unwind writable sret(%union.C) align 16 %{{[^,)]+}}, ptr noundef byval(%union.C) align 16 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passC(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

// M is 32 bytes, so it is passed and returned in memory.
union M retM(const union M *p) { return *p; }

// CIR-LABEL: cir.func {{.*}} @retM(%arg0: !cir.ptr<!rec_M> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_M, llvm.writable} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_M> {llvm.noundef} loc({{[^)]+}}))
// CIR:   cir.copy %{{[^ ]+}} align(16) to %arg0 align(16) : !cir.ptr<!rec_M>

// LLVM-LABEL: define dso_local void @retM(ptr dead_on_unwind noalias writable sret(%union.M) align 16 %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 32, i1 false)

// NOMOVE-LABEL: define {{.*}}@retM(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

char takeM(union M m) { return m.buf[31]; }

// CIR-LABEL: cir.func {{.*}} @takeM(%arg0: !cir.ptr<!rec_M> {llvm.align = 16 : i64, llvm.byval = !rec_M, llvm.noundef} loc({{[^)]+}})) -> !s8i
// CIR:   %[[M:.+]] = cir.alloca "m" align(16) init : !cir.ptr<!rec_M>
// CIR:   cir.copy %arg0 align(16) to %[[M]] align(16) : !cir.ptr<!rec_M>

// LLVM-LABEL: define dso_local i8 @takeM(ptr noundef byval(%union.M) align 16 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@takeM(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void storeM(union M *q, const union M *p) { *q = retM(p); }

// CIR-LABEL: cir.func {{.*}} @storeM(%arg0: !cir.ptr<!rec_M> {llvm.noundef} loc({{[^)]+}}), %arg1: !cir.ptr<!rec_M> {llvm.noundef} loc({{[^)]+}}))
// CIR:   cir.call @retM(%{{[^,]+}}, %{{[^)]+}}) : (!cir.ptr<!rec_M> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.sret = !rec_M, llvm.writable}, !cir.ptr<!rec_M> {llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @storeM(ptr noundef %{{[^,)]+}}, ptr noundef %{{[^,)]+}})
// LLVM:   call void @retM(ptr dead_on_unwind writable sret(%union.M) align 16 %{{[^,)]+}}, ptr noundef %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@storeM(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

char passM(const union M *p) { return takeM(*p); }

// CIR-LABEL: cir.func {{.*}} @passM(%arg0: !cir.ptr<!rec_M> {llvm.noundef} loc({{[^)]+}})) -> !s8i
// CIR:   %[[BYVAL:.+]] = cir.alloca "byval" align(16) : !cir.ptr<!rec_M>
// CIR:   cir.copy %{{[^ ]+}} align(16) to %[[BYVAL]] align(16) : !cir.ptr<!rec_M>
// CIR:   cir.call @takeM(%[[BYVAL]]) : (!cir.ptr<!rec_M> {llvm.align = 16 : i64, llvm.byval = !rec_M, llvm.noundef}) -> !s8i

// LLVM-LABEL: define dso_local i8 @passM(ptr noundef %{{[^,)]+}})
// LLVM:   call i8 @takeM(ptr noundef byval(%union.M) align 16 %{{[^,)]+}})

// NOMOVE-LABEL: define {{.*}}@passM(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

union M vaM(int n, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, n);
  union M m = __builtin_va_arg(ap, union M);
  __builtin_va_end(ap);
  return m;
}

// CIR-LABEL: cir.func {{.*}} @vaM(%arg0: !cir.ptr<!rec_M> {llvm.align = 16 : i64, llvm.dead_on_unwind, llvm.noalias, llvm.sret = !rec_M, llvm.writable} loc({{[^)]+}}), %arg1: !s32i {llvm.noundef} loc({{[^)]+}}), ...)
// CIR:   %[[TMP:.+]] = cir.alloca "vaarg.tmp" align(16) : !cir.ptr<!rec_M>
// CIR-NOT: gp_offset
// CIR:   %[[ALIGNED:.+]] = cir.ptr_mask %{{.+}}, %{{.+}} : (!cir.ptr<!u8i>, !s64i) -> !cir.ptr<!u8i>
// CIR:   %[[SRC:.+]] = cir.cast bitcast %[[ALIGNED]] : !cir.ptr<!u8i> -> !cir.ptr<!rec_M>
// CIR:   cir.copy %[[SRC]] align(16) to %[[TMP]] align(16) : !cir.ptr<!rec_M>
// CIR:   cir.copy %[[TMP]] align(16) to %arg0 align(16) : !cir.ptr<!rec_M>

// LLVM-LABEL: define dso_local void @vaM(ptr dead_on_unwind noalias writable sret(%union.M) align 16 %{{[^,)]+}}, i32 noundef %{{[^,)]+}}, ...)
// LLVM-NOT: icmp
// LLVM:   call ptr @llvm.ptrmask.p0.i64(ptr %{{.+}}, i64 -16)
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 %{{[^,)]+}}, ptr align 16 %{{[^,)]+}}, i64 32, i1 false)

// NOMOVE-LABEL: define {{.*}}@vaM(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

void takeS16(struct S16);

// The argument is read in place from the align-4 local as two i64 eightbytes.
void passS16(struct S16 *p) {
  struct S16 s = *p;
  takeS16(s);
}

// CIR-LABEL: cir.func {{.*}} @passS16(%arg0: !cir.ptr<!rec_S16> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[S:.+]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_S16>
// CIR:   cir.copy %{{[^ ]+}} align(4) to %[[S]] align(4) : !cir.ptr<!rec_S16>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[S]] : !cir.ptr<!rec_S16> -> !cir.ptr<!rec_anon_struct>
// CIR:   %[[LOP:.+]] = cir.get_member %[[VIEW]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[LO:.+]] = cir.load align(4) %[[LOP]] : !cir.ptr<!u64i>, !u64i
// CIR:   %[[HIP:.+]] = cir.get_member %[[VIEW]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!u64i>
// CIR:   %[[HI:.+]] = cir.load align(4) %[[HIP]] : !cir.ptr<!u64i>, !u64i
// CIR:   cir.call @takeS16(%[[LO]], %[[HI]]) : (!u64i, !u64i) -> ()

// LLVM-LABEL: define dso_local void @passS16(ptr noundef %{{[^,)]+}})
// LLVM:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S:[^,]+]], ptr align 4 %{{[^,)]+}}, i64 16, i1 false)
// LLVM:   %[[LOP:.+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[S]], i32 0, i32 0
// LLVM:   %[[LO:.+]] = load i64, ptr %[[LOP]], align 4
// LLVM:   %[[HIP:.+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[S]], i32 0, i32 1
// LLVM:   %[[HI:.+]] = load i64, ptr %[[HIP]], align 4
// LLVM:   call void @takeS16(i64 %[[LO]], i64 %[[HI]])

// NOMOVE-LABEL: define {{.*}}@passS16(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* (x86_fp80|<[0-9]+ x x86_fp80>|<?\{ ?x86_fp80|%struct\.|%union\.)}}

// An X87-class union result stored through a pointer is copied whole out of
// the slot its x86_fp80 is stored into.
union X87U { long double a, b; };
union X87U makeX87U(void);

void storeX87U(union X87U *q) { *q = makeX87U(); }

// CIR-LABEL: cir.func {{.*}} @storeX87U(%arg0: !cir.ptr<!rec_X87U> {llvm.noundef} loc({{[^)]+}}))
// CIR:   %[[SLOT:.+]] = cir.alloca "coerce" align(16) : !cir.ptr<!cir.f80>
// CIR:   %[[Q:.+]] = cir.load deref align(8) %{{[^ ]+}} : !cir.ptr<!cir.ptr<!rec_X87U>>, !cir.ptr<!rec_X87U>
// CIR:   %[[RES:.+]] = cir.call @makeX87U() : () -> !cir.f80
// CIR:   cir.store %[[RES]], %[[SLOT]] : !cir.f80, !cir.ptr<!cir.f80>
// CIR:   %[[VIEW:.+]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!cir.f80> -> !cir.ptr<!rec_X87U>
// CIR:   cir.copy %[[VIEW]] align(16) to %[[Q]] align(16) : !cir.ptr<!rec_X87U>

// LLVM-LABEL: define dso_local void @storeX87U(ptr noundef %{{[^,)]+}})
// LLVMCIR: %[[TMP:.+]] = alloca x86_fp80, align 16
// OGCG:    %[[TMP:.+]] = alloca %union.X87U, align 16
// LLVM:    %[[Q:.+]] = load ptr, ptr %{{[^,)]+}}, align 8
// LLVM:    %[[RES:.+]] = call x86_fp80 @makeX87U()
// LLVM:    store x86_fp80 %[[RES]], ptr %[[TMP]], align 16
// LLVM:    call void @llvm.memcpy.p0.p0.i64(ptr align 16 %[[Q]], ptr align 16 %[[TMP]], i64 16, i1 false)

// NOMOVE-LABEL: define {{.*}}@storeX87U(
// NOMOVE-NOT: {{(load|store)( volatile| atomic)* %union\.}}
