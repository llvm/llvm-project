// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

#define __maybe_undef __attribute__((maybe_undef))

struct Pad { char c; int i; };
struct Dbl { double a, b; };
struct Big { int a[10]; };
struct NT { NT(const NT &); int x; };

void scalar(int a, int __maybe_undef b, int c);
void variadic(int __maybe_undef x, ...);
void coerced(Pad __maybe_undef p);
void flattened(Dbl __maybe_undef d);
void byval(Big __maybe_undef b);
void bitint(_BitInt(129) __maybe_undef x);
void indirect(NT __maybe_undef n);
void cfloat(_Complex float __maybe_undef c);
void cdouble(_Complex double __maybe_undef c);

void test_scalar() {
  int x, y, z;
  scalar(x, y, z);
  variadic(x, y);
}

// CIR-LABEL: cir.func{{.*}} @_Z11test_scalarv()
// CIR:         %[[X:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[Y:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[Z:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[FY:.*]] = cir.freeze %[[Y]] : !s32i
// CIR-NEXT:    cir.call @_Z6scalariii(%[[X]], %[[FY]], %[[Z]])
// CIR:         %[[X2:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[Y2:.*]] = cir.load{{.*}} : !cir.ptr<!s32i>, !s32i
// CIR:         %[[FX2:.*]] = cir.freeze %[[X2]] : !s32i
// CIR-NEXT:    cir.call @_Z8variadiciz(%[[FX2]], %[[Y2]])

// LLVM-LABEL: define{{.*}} void @_Z11test_scalarv()
// LLVM:         %[[X:.*]] = load i32
// LLVM:         %[[Y:.*]] = load i32
// LLVM:         %[[Z:.*]] = load i32
// LLVM:         %[[FY:.*]] = freeze i32 %[[Y]]
// LLVM-NEXT:    call void @_Z6scalariii(i32 noundef %[[X]], i32 noundef %[[FY]], i32 noundef %[[Z]])
// LLVM:         %[[X2:.*]] = load i32
// LLVM:         %[[Y2:.*]] = load i32
// LLVM:         %[[FX2:.*]] = freeze i32 %[[X2]]
// LLVM-NEXT:    call void (i32, ...) @_Z8variadiciz(i32 noundef %[[FX2]], i32 noundef %[[Y2]])

// The coerced value is frozen, so the padding bytes are frozen too.
void test_coerced(Pad p) { coerced(p); }

// CIR-LABEL: cir.func{{.*}} @_Z12test_coerced3Pad(
// CIR:         %[[V:.*]] = cir.load{{.*}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    %[[F:.*]] = cir.freeze %[[V]] : !u64i
// CIR-NEXT:    cir.call @_Z7coerced3Pad(%[[F]])

// LLVM-LABEL: define{{.*}} void @_Z12test_coerced3Pad(
// LLVM:         %[[V:.*]] = load i64
// LLVM-NEXT:    %[[F:.*]] = freeze i64 %[[V]]
// LLVM-NEXT:    call void @_Z7coerced3Pad(i64 %[[F]])

void test_flattened(Dbl d) { flattened(d); }

// CIR-LABEL: cir.func{{.*}} @_Z14test_flattened3Dbl(
// CIR:         %[[A:.*]] = cir.load{{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR:         %[[B:.*]] = cir.load{{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:    %[[FA:.*]] = cir.freeze %[[A]] : !cir.double
// CIR-NEXT:    %[[FB:.*]] = cir.freeze %[[B]] : !cir.double
// CIR-NEXT:    cir.call @_Z9flattened3Dbl(%[[FA]], %[[FB]])

// LLVM-LABEL: define{{.*}} void @_Z14test_flattened3Dbl(
// LLVM-DAG:     %[[FA:.*]] = freeze double
// LLVM-DAG:     %[[FB:.*]] = freeze double
// LLVM:         call void @_Z9flattened3Dbl(double %[[FA]], double %[[FB]])

// Like classic CodeGen, an indirect argument freezes the pointer.
void test_byval(Big b) { byval(b); }

// CIR-LABEL: cir.func{{.*}} @_Z10test_byval3Big(
// CIR:         %[[F:.*]] = cir.freeze %{{.*}} : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @_Z5byval3Big(%[[F]])

// LLVM-LABEL: define{{.*}} void @_Z10test_byval3Big(
// LLVM:         %[[F:.*]] = freeze ptr %{{.*}}
// LLVM-NEXT:    call void @_Z5byval3Big(ptr noundef byval(%struct.Big) align 8 %[[F]])

void test_bitint(_BitInt(129) x) { bitint(x); }

// CIR-LABEL: cir.func{{.*}} @_Z11test_bitintDB129_(
// CIR:         %[[F:.*]] = cir.freeze %{{.*}} : !cir.ptr<
// CIR-NEXT:    cir.call @_Z6bitintDB129_(%[[F]])

// LLVM-LABEL: define{{.*}} void @_Z11test_bitintDB129_(
// LLVM:         %[[F:.*]] = freeze ptr %{{.*}}
// LLVM:         call void @_Z6bitintDB129_(ptr noundef byval([24 x i8]) align 8 %[[F]])

void test_indirect(NT &n) { indirect(n); }

// CIR-LABEL: cir.func{{.*}} @_Z13test_indirectR2NT(
// CIR:         cir.call @_ZN2NTC1ERKS_(%[[TMP:[0-9]+]],
// CIR-NEXT:    %[[F:.*]] = cir.freeze %[[TMP]] : !cir.ptr<!rec_NT>
// CIR-NEXT:    cir.call @_Z8indirect2NT(%[[F]])

// LLVM-LABEL: define{{.*}} void @_Z13test_indirectR2NT(
// LLVM:         call void @_ZN2NTC1ERKS_(ptr {{.*}} %[[TMP:[^,]+]],
// LLVM-NEXT:    %[[F:.*]] = freeze ptr %[[TMP]]
// LLVM-NEXT:    call void @_Z8indirect2NT(ptr {{.*}} %[[F]])

void test_complex(_Complex float f, _Complex double d) {
  cfloat(f);
  cdouble(d);
}

// CIR-LABEL: cir.func{{.*}} @_Z12test_complexCfCd(
// CIR:         %[[V:.*]] = cir.load{{.*}} : !cir.ptr<!cir.vector<2 x !cir.float>>, !cir.vector<2 x !cir.float>
// CIR-NEXT:    %[[F:.*]] = cir.freeze %[[V]] : !cir.vector<2 x !cir.float>
// CIR-NEXT:    cir.call @_Z6cfloatCf(%[[F]])
// CIR:         %[[R:.*]] = cir.load{{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR:         %[[I:.*]] = cir.load{{.*}} : !cir.ptr<!cir.double>, !cir.double
// CIR-NEXT:    %[[FR:.*]] = cir.freeze %[[R]] : !cir.double
// CIR-NEXT:    %[[FI:.*]] = cir.freeze %[[I]] : !cir.double
// CIR-NEXT:    cir.call @_Z7cdoubleCd(%[[FR]], %[[FI]])

// LLVM-LABEL: define{{.*}} void @_Z12test_complexCfCd(
// LLVM:         %[[V:.*]] = load <2 x float>
// LLVM-NEXT:    %[[F:.*]] = freeze <2 x float> %[[V]]
// LLVM-NEXT:    call void @_Z6cfloatCf(<2 x float> noundef %[[F]])
// LLVM-DAG:     %[[FR:.*]] = freeze double
// LLVM-DAG:     %[[FI:.*]] = freeze double
// LLVM:         call void @_Z7cdoubleCd(double noundef %[[FR]], double noundef %[[FI]])
