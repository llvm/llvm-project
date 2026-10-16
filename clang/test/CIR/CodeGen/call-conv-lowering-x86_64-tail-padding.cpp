// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

struct V { char v; };
struct B { int i; char c; B(int); B() = default; };
B makeB();

struct C : virtual V {
  [[no_unique_address]] B b;
  C();
};

// b is at offset 8 with data size 5, and V is in its tail padding at 13.
C::C() : b(makeB()) {}

// CIR-LABEL: cir.func {{.*}} @_ZN1CC2Ev(
// CIR:         %[[SLOT:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR:         %[[B:.*]] = cir.get_member %{{.*}}[1] {name = "b"}
// CIR:         %[[BPTR:.*]] = cir.cast bitcast %[[B]] : !cir.ptr<!rec_B2Ebase> -> !cir.ptr<!rec_B>
// CIR:         %[[RES:.*]] = cir.call @_Z5makeBv() : () -> !u64i
// CIR:         cir.store %[[RES]], %[[SLOT]] : !u64i, !cir.ptr<!u64i>
// CIR:         %[[VIEW:.*]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_B>
// CIR:         cir.copy %[[VIEW]] align(8) to %[[BPTR]] align(8) skip_tail_padding : !cir.ptr<!rec_B>

// LLVM-LABEL: define dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 8 dereferenceable(13) %{{[^,]+}}, ptr noundef %{{[^,)]+}})
// LLVMCIR:      %[[SLOT:.*]] = alloca i64, align 8
// LLVM:         %[[B:.*]] = getelementptr inbounds nuw %struct.C, ptr %{{[^,]+}}, i32 0, i32 1
// LLVM:         %[[RES:.*]] = call i64 @_Z5makeBv()
// LLVMCIR:      store i64 %[[RES]], ptr %[[SLOT]], align 8
// LLVMCIR:      call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[B]], ptr align 8 %[[SLOT]], i64 5, i1 false)
// OGCG:         %[[TRUNC:.*]] = trunc i64 %[[RES]] to i40
// OGCG:         store i40 %[[TRUNC]], ptr %[[B]], align 8

void localB() { B b = makeB(); }

// CIR-LABEL: cir.func {{.*}} @_Z6localBv()
// CIR:         %[[SLOT:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR:         %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!rec_B>
// CIR:         %[[RES:.*]] = cir.call @_Z5makeBv() : () -> !u64i
// CIR:         cir.store %[[RES]], %[[SLOT]] : !u64i, !cir.ptr<!u64i>
// CIR:         %[[VIEW:.*]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_B>
// CIR:         cir.copy %[[VIEW]] align(8) to %[[B]] align(4) : !cir.ptr<!rec_B>

// LLVM-LABEL: define dso_local void @_Z6localBv()
// LLVMCIR:      %[[SLOT:.*]] = alloca i64, align 8
// LLVM:         %[[B:.*]] = alloca %struct.B, align 4
// LLVM:         %[[RES:.*]] = call i64 @_Z5makeBv()
// LLVMCIR:      store i64 %[[RES]], ptr %[[SLOT]], align 8
// LLVMCIR:      call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[B]], ptr align 8 %[[SLOT]], i64 8, i1 false)
// OGCG:         %[[TRUNC:.*]] = trunc i64 %[[RES]] to i40
// OGCG:         store i40 %[[TRUNC]], ptr %[[B]], align 4

struct S { int a, b; };
S makeS();

void localS() { S s = makeS(); }

// CIR-LABEL: cir.func {{.*}} @_Z6localSv()
// CIR:         %[[SLOT:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR:         %[[S:.*]] = cir.alloca "s" align(4) init : !cir.ptr<!rec_S>
// CIR:         %[[RES:.*]] = cir.call @_Z5makeSv() : () -> !u64i
// CIR:         cir.store %[[RES]], %[[SLOT]] : !u64i, !cir.ptr<!u64i>
// CIR:         %[[VIEW:.*]] = cir.cast bitcast %[[SLOT]] : !cir.ptr<!u64i> -> !cir.ptr<!rec_S>
// CIR:         cir.copy %[[VIEW]] align(8) to %[[S]] align(4) : !cir.ptr<!rec_S>

// LLVM-LABEL: define dso_local void @_Z6localSv()
// LLVMCIR:      %[[SLOT:.*]] = alloca i64, align 8
// LLVM:         %[[S:.*]] = alloca %struct.S, align 4
// LLVM:         %[[RES:.*]] = call i64 @_Z5makeSv()
// LLVMCIR:      store i64 %[[RES]], ptr %[[SLOT]], align 8
// LLVMCIR:      call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[S]], ptr align 8 %[[SLOT]], i64 8, i1 false)
// OGCG:         store i64 %[[RES]], ptr %[[S]], align 4

struct Big { long a, b, c; char d; };

void takeBig(Big x) {}

// CIR-LABEL: cir.func {{.*}} @_Z7takeBig3Big(%arg0: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noundef}
// CIR:         %[[X:.*]] = cir.alloca "x" align(8) init : !cir.ptr<!rec_Big>
// CIR:         cir.copy %arg0 align(8) to %[[X]] align(8) : !cir.ptr<!rec_Big>

// LLVM-LABEL: define dso_local void @_Z7takeBig3Big(ptr noundef byval(%struct.Big) align 8 %{{[^,)]+}})
// LLVMCIR:      %[[X:.*]] = alloca %struct.Big, align 8
// LLVMCIR:      call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[X]], ptr align 8 %{{[^,]+}}, i64 32, i1 false)
// OGCG-NOT:     call void @llvm.memcpy
// LLVM:         ret void
