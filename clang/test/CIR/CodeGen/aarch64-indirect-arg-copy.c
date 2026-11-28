// RUN: %clang_cc1 -triple aarch64-linux-gnu -fexperimental-max-bitint-width=1024 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fexperimental-max-bitint-width=1024 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fexperimental-max-bitint-width=1024 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

// AAPCS64 passes a record larger than 16 bytes, and a _BitInt wider than 128
// bits, indirectly without byval.  The callee may write the memory it is
// passed, so the caller passes a copy.

typedef struct { long a, b, c, d; } Big;
struct Outer { int x; Big inner; };

void take(Big);
void use(Big *);
Big make(void);
Big global;

// The callee's writes to its argument do not reach s.
void local_then_use(void) {
  Big s = {1, 2, 3, 4};
  take(s);
  use(&s);
}

// CIR-LABEL: cir.func{{.*}} @local_then_use()
// CIR:         %[[S:.*]] = cir.alloca "s" align(8) init : !cir.ptr<!rec_Big>
// CIR:         %[[TMP:.*]] = cir.alloca "indirect.tmp" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.copy %[[S]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @take(%[[TMP]]) : (!cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.dead_on_return = -1 : i64, llvm.dereferenceable = 32 : i64, llvm.nofreeobj, llvm.noundef}) -> ()
// CIR-NEXT:    cir.call @use(%[[S]])

// LLVM-LABEL: define dso_local void @local_then_use()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[S:[^,]+]], ptr align 8 @__const.local_then_use.s, i64 32, i1 false)
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %[[S]], i64 32, i1 false)
// LLVM-NEXT:    call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[TMP]])
// LLVM-NEXT:    call void @use(ptr noundef %[[S]])

void through_pointer(Big *p) { take(*p); }

// CIR-LABEL: cir.func{{.*}} @through_pointer(
// CIR:         %[[P:.*]] = cir.load deref align(8) %{{.*}} : !cir.ptr<!cir.ptr<!rec_Big>>, !cir.ptr<!rec_Big>
// CIR-NEXT:    %[[TMP:.*]] = cir.alloca "indirect.tmp" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.copy %[[P]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @take(%[[TMP]])

// LLVM-LABEL: define dso_local void @through_pointer(
// LLVM:         %[[P:.*]] = load ptr, ptr %{{.*}}, align 8
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %[[P]], i64 32, i1 false)
// LLVM-NEXT:    call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[TMP]])

void from_global(void) { take(global); }

// CIR-LABEL: cir.func{{.*}} @from_global()
// CIR:         %[[G:.*]] = cir.get_global @global : !cir.ptr<!rec_Big>
// CIR-NEXT:    %[[TMP:.*]] = cir.alloca "indirect.tmp" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.copy %[[G]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @take(%[[TMP]])

// LLVM-LABEL: define dso_local void @from_global()
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 @global, i64 32, i1 false)
// LLVM-NEXT:    call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[TMP]])

void from_member(struct Outer *o) { take(o->inner); }

// CIR-LABEL: cir.func{{.*}} @from_member(
// CIR:         %[[M:.*]] = cir.get_member %{{.*}}[1] {name = "inner"} : !cir.ptr<!rec_Outer> -> !cir.ptr<!rec_Big>
// CIR-NEXT:    %[[TMP:.*]] = cir.alloca "indirect.tmp" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.copy %[[M]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @take(%[[TMP]])

// LLVM-LABEL: define dso_local void @from_member(
// LLVM:         %[[M:.*]] = getelementptr inbounds nuw %struct.Outer, ptr %{{.*}}, i32 0, i32 1
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %[[M]], i64 32, i1 false)
// LLVM-NEXT:    call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[TMP]])

// Classic codegen passes the call's result slot.  CIR copies it.
void from_call(void) { take(make()); }

// CIR-LABEL: cir.func{{.*}} @from_call()
// CIR:         %[[RES:.*]] = cir.alloca "agg.tmp0" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @make(%[[RES]])
// CIR-NEXT:    %[[TMP:.*]] = cir.alloca "indirect.tmp" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.copy %[[RES]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @take(%[[TMP]])

// LLVM-LABEL: define dso_local void @from_call()
// LLVM:         call void @make(ptr dead_on_unwind writable sret(%struct.Big) align 8 %[[RES:[^)]+]])
// LLVMCIR:      call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %[[RES]], i64 32, i1 false)
// LLVMCIR-NEXT: call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[TMP]])
// OGCG-NEXT:    call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[RES]])

// The incoming argument is the caller's copy, which this function copies
// again for its own callee.
void pass_param_on(Big b) { take(b); }

// CIR-LABEL: cir.func{{.*}} @pass_param_on(
// CIR-SAME:    %[[B:[a-z0-9]+]]: !cir.ptr<!rec_Big>
// CIR-NEXT:    %[[TMP:.*]] = cir.alloca "indirect.tmp" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.copy %[[B]] align(8) to %[[TMP]] align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.call @take(%[[TMP]])

// LLVM-LABEL: define dso_local void @pass_param_on(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %{{.*}})
// LLVM:         call void @llvm.memcpy.p0.p0.i64(ptr align 8 %[[TMP:[^,]+]], ptr align 8 %{{.*}}, i64 32, i1 false)
// LLVM-NEXT:    call void @take(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %[[TMP]])

void take_bitint(_BitInt(129));
void bitint_through_pointer(_BitInt(129) *p) { take_bitint(*p); }

// CIR-LABEL: cir.func{{.*}} @bitint_through_pointer(
// CIR:         %[[V:.*]] = cir.load align(16) %{{.*}} : !cir.ptr<!cir.int<s, 129, bitint>>, !cir.int<s, 129, bitint>
// CIR-NEXT:    %[[TMP:.*]] = cir.alloca "indirect.tmp" align(16) : !cir.ptr<!cir.int<s, 129, bitint>>
// CIR-NEXT:    cir.store %[[V]], %[[TMP]] : !cir.int<s, 129, bitint>, !cir.ptr<!cir.int<s, 129, bitint>>
// CIR-NEXT:    cir.call @take_bitint(%[[TMP]]) : (!cir.ptr<!cir.int<s, 129, bitint>> {llvm.align = 16 : i64, llvm.dead_on_return = -1 : i64, llvm.dereferenceable = 32 : i64, llvm.nofreeobj, llvm.noundef}) -> ()

// LLVM-LABEL: define dso_local void @bitint_through_pointer(
// LLVM:         %[[EXT:.*]] = sext i129 %{{.*}} to i256
// LLVM-NEXT:    store i256 %[[EXT]], ptr %[[TMP:[^,]+]], align 16
// LLVM-NEXT:    call void @take_bitint(ptr nofreeobj noundef align 16 dead_on_return dereferenceable(32) %[[TMP]])
