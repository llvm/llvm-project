// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fclangir-call-conv-lowering -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-CIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVM-OGCG --input-file=%t.ll %s

typedef struct { int x; int y; } Pair2;
typedef struct { long a; long b; } Pair16;
typedef struct { long a, b, c, d; } Big;
typedef struct { __int128 w; } Wide;
typedef struct { __int128 w; char c; } WideChar;

int vf(Pair2 p, ...);

// CIR: cir.func private @vf(!u64i, ...) -> !s32i

int call_scalar(Pair2 p, int a, double d) { return vf(p, a, d); }

// CIR-LABEL: cir.func {{.*}}@call_scalar(%arg0: !u64i loc({{.+}}), %arg1: !s32i {llvm.noundef} loc({{.+}}), %arg2: !cir.double {llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         cir.store %arg1, %[[ASLOT:[0-9]+]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.store %arg2, %[[DSLOT:[0-9]+]] : !cir.double, !cir.ptr<!cir.double>
// CIR:         %[[AV:[0-9]+]] = cir.load align(4) %[[ASLOT]] : !cir.ptr<!s32i>, !s32i
// CIR:         %[[DV:[0-9]+]] = cir.load align(8) %[[DSLOT]] : !cir.ptr<!cir.double>, !cir.double
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         cir.call @vf(%[[PV]], %[[AV]], %[[DV]]) : (!u64i, !s32i {llvm.noundef}, !cir.double {llvm.noundef}) -> !s32i

// LLVM-LABEL: define dso_local i32 @call_scalar(
// LLVM-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i32 noundef %[[A:[0-9a-zA-Z._]+]], double noundef %[[D:[0-9a-zA-Z._]+]])
// LLVM:         store i32 %[[A]], ptr %[[ASLOT:[0-9a-zA-Z._]+]], align 4
// LLVM:         store double %[[D]], ptr %[[DSLOT:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[AV:[0-9a-zA-Z._]+]] = load i32, ptr %[[ASLOT]], align 4
// LLVM:         %[[DV:[0-9a-zA-Z._]+]] = load double, ptr %[[DSLOT]], align 8
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM:         call i32 (i64, ...) @vf(i64 %[[PV]], i32 noundef %[[AV]], double noundef %[[DV]])

// A two-eightbyte record at the ellipsis is flattened into two INTEGER
// registers while registers remain.
int call_small(Pair2 p, Pair16 q) { return vf(p, q); }

// CIR-LABEL: cir.func {{.*}}@call_small(%arg0: !u64i loc({{.+}}), %arg1: !s64i loc({{.+}}), %arg2: !s64i loc({{.+}})) -> !s32i
// CIR:         %[[IN0:[0-9]+]] = cir.get_member %[[IN:[0-9]+]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s64i>
// CIR:         cir.store %arg1, %[[IN0]] : !s64i, !cir.ptr<!s64i>
// CIR:         %[[IN1:[0-9]+]] = cir.get_member %[[IN]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s64i>
// CIR:         cir.store %arg2, %[[IN1]] : !s64i, !cir.ptr<!s64i>
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         %[[OUT0:[0-9]+]] = cir.get_member %[[OUT:[0-9]+]][0] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s64i>
// CIR:         %[[Q0:[0-9]+]] = cir.load %[[OUT0]] : !cir.ptr<!s64i>, !s64i
// CIR:         %[[OUT1:[0-9]+]] = cir.get_member %[[OUT]][1] {name = ""} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!s64i>
// CIR:         %[[Q1:[0-9]+]] = cir.load %[[OUT1]] : !cir.ptr<!s64i>, !s64i
// CIR:         cir.call @vf(%[[PV]], %[[Q0]], %[[Q1]]) : (!u64i, !s64i, !s64i) -> !s32i

// LLVM-LABEL: define dso_local i32 @call_small(
// LLVM-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i64 %[[Q0:[0-9a-zA-Z._]+]], i64 %[[Q1:[0-9a-zA-Z._]+]])
// LLVM:         %[[S0:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[SLOT:[0-9a-zA-Z._]+]], i32 0, i32 0
// LLVM:         store i64 %[[Q0]], ptr %[[S0]], align 8
// LLVM:         %[[S1:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[SLOT]], i32 0, i32 1
// LLVM:         store i64 %[[Q1]], ptr %[[S1]], align 8
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM:         %[[L0:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[REC:[0-9a-zA-Z._]+]], i32 0, i32 0
// LLVM:         %[[A0:[0-9a-zA-Z._]+]] = load i64, ptr %[[L0]], align 8
// LLVM:         %[[L1:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[REC]], i32 0, i32 1
// LLVM:         %[[A1:[0-9a-zA-Z._]+]] = load i64, ptr %[[L1]], align 8
// LLVM:         call i32 (i64, ...) @vf(i64 %[[PV]], i64 %[[A0]], i64 %[[A1]])

// Larger than two eightbytes is MEMORY regardless of register availability.
int call_big(Pair2 p, Big b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@call_big(%arg0: !u64i loc({{.+}}), %arg1: !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noalias, llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         %{{[0-9]+}} = cir.load %arg1 : !cir.ptr<!rec_Big>, !rec_Big
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    %[[COPY:[0-9]+]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_Big>
// CIR-NEXT:    cir.store %{{[0-9]+}}, %[[COPY]] : !rec_Big, !cir.ptr<!rec_Big>
// CIR-NEXT:    %{{[0-9]+}} = cir.call @vf(%[[PV]], %[[COPY]]) : (!u64i, !cir.ptr<!rec_Big> {llvm.align = 8 : i64, llvm.byval = !rec_Big, llvm.noalias, llvm.noundef}) -> !s32i

// CIR copies the incoming byval slot before forwarding it.  OGCG does not.
// LLVM-CIR-LABEL: define dso_local i32 @call_big(
// LLVM-CIR-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], ptr noalias noundef byval(%struct.Big) align 8 %[[B:[0-9a-zA-Z._]+]])
// LLVM-CIR:       %{{[0-9a-zA-Z._]+}} = load %struct.Big, ptr %[[B]], align 8
// LLVM-CIR:       %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 8
// LLVM-CIR-NEXT:  %[[COPY:[0-9a-zA-Z._]+]] = alloca %struct.Big, align 8
// LLVM-CIR-NEXT:  store %struct.Big %{{[0-9a-zA-Z._]+}}, ptr %[[COPY]], align 8
// LLVM-CIR-NEXT:  %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], ptr noalias noundef byval(%struct.Big) align 8 %[[COPY]])

// LLVM-OGCG-LABEL: define dso_local i32 @call_big(
// LLVM-OGCG-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], ptr noundef byval(%struct.Big) align 8 %[[B:[0-9a-zA-Z._]+]])
// LLVM-OGCG:       %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 4
// LLVM-OGCG-NEXT:  %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], ptr noundef byval(%struct.Big) align 8 %[[B]])

// The same Pair16 that went to registers in call_small goes to memory here:
// the named parameter and four longs leave only one INTEGER register, and a
// two-eightbyte record cannot be split across a register and the stack.
int call_exhausted(Pair2 p, long a, long b, long c, long d, Pair16 q) {
  return vf(p, a, b, c, d, q);
}

// CIR-LABEL: cir.func {{.*}}@call_exhausted(%arg0: !u64i loc({{.+}}), %arg1: !s64i {llvm.noundef} loc({{.+}}), %arg2: !s64i {llvm.noundef} loc({{.+}}), %arg3: !s64i {llvm.noundef} loc({{.+}}), %arg4: !s64i {llvm.noundef} loc({{.+}}), %arg5: !cir.ptr<!rec_Pair16> {llvm.align = 8 : i64, llvm.byval = !rec_Pair16, llvm.noalias, llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         cir.store %arg1, %[[AS:[0-9]+]] : !s64i, !cir.ptr<!s64i>
// CIR:         cir.store %arg2, %[[BS:[0-9]+]] : !s64i, !cir.ptr<!s64i>
// CIR:         cir.store %arg3, %[[CS:[0-9]+]] : !s64i, !cir.ptr<!s64i>
// CIR:         cir.store %arg4, %[[DS:[0-9]+]] : !s64i, !cir.ptr<!s64i>
// CIR:         %[[AV:[0-9]+]] = cir.load align(8) %[[AS]] : !cir.ptr<!s64i>, !s64i
// CIR:         %[[BV:[0-9]+]] = cir.load align(8) %[[BS]] : !cir.ptr<!s64i>, !s64i
// CIR:         %[[CV:[0-9]+]] = cir.load align(8) %[[CS]] : !cir.ptr<!s64i>, !s64i
// CIR:         %[[DV:[0-9]+]] = cir.load align(8) %[[DS]] : !cir.ptr<!s64i>, !s64i
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    %[[COPY:[0-9]+]] = cir.alloca "byval" align(8) : !cir.ptr<!rec_Pair16>
// CIR-NEXT:    cir.store %{{[0-9]+}}, %[[COPY]] : !rec_Pair16, !cir.ptr<!rec_Pair16>
// CIR-NEXT:    %{{[0-9]+}} = cir.call @vf(%[[PV]], %[[AV]], %[[BV]], %[[CV]], %[[DV]], %[[COPY]]) : (!u64i, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !s64i {llvm.noundef}, !cir.ptr<!rec_Pair16> {llvm.align = 8 : i64, llvm.byval = !rec_Pair16, llvm.noalias, llvm.noundef}) -> !s32i

// LLVM-CIR-LABEL: define dso_local i32 @call_exhausted(
// LLVM-CIR-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i64 noundef %[[A:[0-9a-zA-Z._]+]], i64 noundef %[[B:[0-9a-zA-Z._]+]], i64 noundef %[[C:[0-9a-zA-Z._]+]], i64 noundef %[[D:[0-9a-zA-Z._]+]], ptr noalias noundef byval(%struct.Pair16) align 8 %[[Q:[0-9a-zA-Z._]+]])
// LLVM-OGCG-LABEL: define dso_local i32 @call_exhausted(
// LLVM-OGCG-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i64 noundef %[[A:[0-9a-zA-Z._]+]], i64 noundef %[[B:[0-9a-zA-Z._]+]], i64 noundef %[[C:[0-9a-zA-Z._]+]], i64 noundef %[[D:[0-9a-zA-Z._]+]], ptr noundef byval(%struct.Pair16) align 8 %[[Q:[0-9a-zA-Z._]+]])
// LLVM:         store i64 %[[A]], ptr %[[AS:[0-9a-zA-Z._]+]], align 8
// LLVM:         store i64 %[[B]], ptr %[[BS:[0-9a-zA-Z._]+]], align 8
// LLVM:         store i64 %[[C]], ptr %[[CS:[0-9a-zA-Z._]+]], align 8
// LLVM:         store i64 %[[D]], ptr %[[DS:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[AV:[0-9a-zA-Z._]+]] = load i64, ptr %[[AS]], align 8
// LLVM:         %[[BV:[0-9a-zA-Z._]+]] = load i64, ptr %[[BS]], align 8
// LLVM:         %[[CV:[0-9a-zA-Z._]+]] = load i64, ptr %[[CS]], align 8
// LLVM:         %[[DV:[0-9a-zA-Z._]+]] = load i64, ptr %[[DS]], align 8
// LLVM-CIR:       %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 8
// LLVM-CIR-NEXT:  %[[COPY:[0-9a-zA-Z._]+]] = alloca %struct.Pair16, align 8
// LLVM-CIR-NEXT:  store %struct.Pair16 %{{[0-9a-zA-Z._]+}}, ptr %[[COPY]], align 8
// LLVM-CIR-NEXT:  %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], i64 noundef %[[AV]], i64 noundef %[[BV]], i64 noundef %[[CV]], i64 noundef %[[DV]], ptr noalias noundef byval(%struct.Pair16) align 8 %[[COPY]])
// LLVM-OGCG:      %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 4
// LLVM-OGCG-NEXT: %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], i64 noundef %[[AV]], i64 noundef %[[BV]], i64 noundef %[[CV]], i64 noundef %[[DV]], ptr noundef byval(%struct.Pair16) align 8 %[[Q]])

// A 128-bit integer spans two eightbytes but is still passed whole.
int call_int128(Pair2 p, __int128 w) { return vf(p, w); }

// CIR-LABEL: cir.func {{.*}}@call_int128(%arg0: !u64i loc({{.+}}), %arg1: !s128i {llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         cir.store %arg1, %[[WSLOT:[0-9]+]] : !s128i, !cir.ptr<!s128i>
// CIR:         %[[WV:[0-9]+]] = cir.load align(16) %[[WSLOT]] : !cir.ptr<!s128i>, !s128i
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         cir.call @vf(%[[PV]], %[[WV]]) : (!u64i, !s128i {llvm.noundef}) -> !s32i

// LLVM-LABEL: define dso_local i32 @call_int128(
// LLVM-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i128 noundef %[[W:[0-9a-zA-Z._]+]])
// LLVM:         store i128 %[[W]], ptr %[[WSLOT:[0-9a-zA-Z._]+]], align 16
// LLVM:         %[[WV:[0-9a-zA-Z._]+]] = load i128, ptr %[[WSLOT]], align 16
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM:         call i32 (i64, ...) @vf(i64 %[[PV]], i128 noundef %[[WV]])

// Wrapping it in a record does not change the class: both eightbytes are
// INTEGER, so the record is coerced back to a bare i128.
int call_wide(Pair2 p, Wide w) { return vf(p, w); }

// CIR-LABEL: cir.func {{.*}}@call_wide(%arg0: !u64i loc({{.+}}), %arg1: !s128i loc({{.+}})) -> !s32i
// CIR:         cir.store %arg1, %[[IN:[0-9]+]] : !s128i, !cir.ptr<!s128i>
// CIR:         %[[INCAST:[0-9]+]] = cir.cast bitcast %[[IN]] : !cir.ptr<!s128i> -> !cir.ptr<!rec_Wide>
// CIR:         %{{[0-9]+}} = cir.load %[[INCAST]] : !cir.ptr<!rec_Wide>, !rec_Wide
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         cir.store %{{[0-9]+}}, %[[OUT:[0-9]+]] : !rec_Wide, !cir.ptr<!rec_Wide>
// CIR:         %[[OUTCAST:[0-9]+]] = cir.cast bitcast %[[OUT]] : !cir.ptr<!rec_Wide> -> !cir.ptr<!s128i>
// CIR:         %[[WV:[0-9]+]] = cir.load %[[OUTCAST]] : !cir.ptr<!s128i>, !s128i
// CIR:         cir.call @vf(%[[PV]], %[[WV]]) : (!u64i, !s128i) -> !s32i

// LLVM-LABEL: define dso_local i32 @call_wide(
// LLVM-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i128 %[[W:[0-9a-zA-Z._]+]])
// LLVM-CIR:       store i128 %[[W]], ptr %[[IN:[0-9a-zA-Z._]+]], align 16
// LLVM-CIR:       %{{[0-9a-zA-Z._]+}} = load %struct.Wide, ptr %[[IN]], align 16
// LLVM-CIR:       %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 8
// LLVM-CIR:       store %struct.Wide %{{[0-9a-zA-Z._]+}}, ptr %[[OUT:[0-9a-zA-Z._]+]], align 16
// LLVM-CIR-NEXT:  %[[WV:[0-9a-zA-Z._]+]] = load i128, ptr %[[OUT]], align 16

// LLVM-OGCG:      %[[DIVE:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw %struct.Wide, ptr %[[WSLOT:[0-9a-zA-Z._]+]], i32 0, i32 0
// LLVM-OGCG-NEXT: store i128 %[[W]], ptr %[[DIVE]], align 16
// LLVM-OGCG:      %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 4
// LLVM-OGCG-NEXT: %[[DIVE1:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw %struct.Wide, ptr %[[WSLOT]], i32 0, i32 0
// LLVM-OGCG-NEXT: %[[WV:[0-9a-zA-Z._]+]] = load i128, ptr %[[DIVE1]], align 16

// LLVM-NEXT:      %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], i128 %[[WV]])

// One trailing byte pushes the record past two eightbytes, so it goes to
// memory, and the 128-bit member keeps the slot at 16-byte alignment.
int call_wide_char(Pair2 p, WideChar w) { return vf(p, w); }

// CIR-LABEL: cir.func {{.*}}@call_wide_char(%arg0: !u64i loc({{.+}}), %arg1: !cir.ptr<!rec_WideChar> {llvm.align = 16 : i64, llvm.byval = !rec_WideChar, llvm.noalias, llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         %{{[0-9]+}} = cir.load %arg1 : !cir.ptr<!rec_WideChar>, !rec_WideChar
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    %[[COPY:[0-9]+]] = cir.alloca "byval" align(16) : !cir.ptr<!rec_WideChar>
// CIR-NEXT:    cir.store %{{[0-9]+}}, %[[COPY]] : !rec_WideChar, !cir.ptr<!rec_WideChar>
// CIR-NEXT:    %{{[0-9]+}} = cir.call @vf(%[[PV]], %[[COPY]]) : (!u64i, !cir.ptr<!rec_WideChar> {llvm.align = 16 : i64, llvm.byval = !rec_WideChar, llvm.noalias, llvm.noundef}) -> !s32i

// LLVM-CIR-LABEL: define dso_local i32 @call_wide_char(
// LLVM-CIR-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], ptr noalias noundef byval(%struct.WideChar) align 16 %[[W:[0-9a-zA-Z._]+]])
// LLVM-CIR:       %{{[0-9a-zA-Z._]+}} = load %struct.WideChar, ptr %[[W]], align 16
// LLVM-CIR:       %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 8
// LLVM-CIR-NEXT:  %[[COPY:[0-9a-zA-Z._]+]] = alloca %struct.WideChar, align 16
// LLVM-CIR-NEXT:  store %struct.WideChar %{{[0-9a-zA-Z._]+}}, ptr %[[COPY]], align 16
// LLVM-CIR-NEXT:  %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], ptr noalias noundef byval(%struct.WideChar) align 16 %[[COPY]])

// LLVM-OGCG-LABEL: define dso_local i32 @call_wide_char(
// LLVM-OGCG-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], ptr noundef byval(%struct.WideChar) align 16 %[[W:[0-9a-zA-Z._]+]])
// LLVM-OGCG:       %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align 4
// LLVM-OGCG-NEXT:  %{{[0-9a-zA-Z._]+}} = call i32 (i64, ...) @vf(i64 %[[PV]], ptr noundef byval(%struct.WideChar) align 16 %[[W]])

// A _BitInt narrower than a register is extended at the ellipsis, same as a
// declared parameter.
int ell_bitint17(Pair2 p, _BitInt(17) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint17(%arg0: !u64i loc({{.+}}), %arg1: !cir.int<s, 17, bitint> {llvm.signext} loc({{.+}})) -> !s32i
// CIR:         cir.store %arg1, %[[BSLOT:[0-9]+]] : !cir.int<s, 17, bitint>, !cir.ptr<!cir.int<s, 17, bitint>>
// CIR:         %[[BV:[0-9]+]] = cir.load align(4) %[[BSLOT]] : !cir.ptr<!cir.int<s, 17, bitint>>, !cir.int<s, 17, bitint>
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         cir.call @vf(%[[PV]], %[[BV]]) : (!u64i, !cir.int<s, 17, bitint> {llvm.signext}) -> !s32i

// LLVM-CIR-LABEL: define dso_local i32 @ell_bitint17(
// LLVM-CIR-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i17 signext %[[B:[0-9a-zA-Z._]+]])
// LLVM-OGCG-LABEL: define dso_local i32 @ell_bitint17(
// LLVM-OGCG-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i17 noundef signext %[[B:[0-9a-zA-Z._]+]])
// LLVM:         %[[EXT:[0-9a-zA-Z._]+]] = sext i17 %[[B]] to i32
// LLVM:         store i32 %[[EXT]], ptr %[[BSLOT:[0-9a-zA-Z._]+]], align 4
// LLVM:         %[[RE:[0-9a-zA-Z._]+]] = load i32, ptr %[[BSLOT]], align 4
// LLVM:         %[[TR:[0-9a-zA-Z._]+]] = trunc i32 %[[RE]] to i17
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM-CIR:     call i32 (i64, ...) @vf(i64 %[[PV]], i17 signext %[[TR]])
// LLVM-OGCG:    call i32 (i64, ...) @vf(i64 %[[PV]], i17 noundef signext %[[TR]])

// A width between 33 and 63 widens to one register.
int ell_bitint48(Pair2 p, _BitInt(48) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint48(%arg0: !u64i loc({{.+}}), %arg1: !u64i {llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         %[[BV:[0-9]+]] = cir.load align(8) %{{[0-9]+}} : !cir.ptr<!cir.int<s, 48, bitint>>, !cir.int<s, 48, bitint>
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         %[[OUTCAST:[0-9]+]] = cir.cast bitcast %[[OUT:[0-9]+]] : !cir.ptr<!u64i> -> !cir.ptr<!cir.int<s, 48, bitint>>
// CIR-NEXT:    cir.store %[[BV]], %[[OUTCAST]] : !cir.int<s, 48, bitint>, !cir.ptr<!cir.int<s, 48, bitint>>
// CIR-NEXT:    %[[ARG:[0-9]+]] = cir.load %[[OUT]] : !cir.ptr<!u64i>, !u64i
// CIR-NEXT:    %{{[0-9]+}} = cir.call @vf(%[[PV]], %[[ARG]]) : (!u64i, !u64i {llvm.noundef}) -> !s32i

// LLVM-LABEL: define dso_local i32 @ell_bitint48(
// LLVM-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i64 noundef %[[B:[0-9a-zA-Z._]+]])
// LLVM:         %[[TR:[0-9a-zA-Z._]+]] = trunc i64 %{{[0-9a-zA-Z._]+}} to i48
// LLVM:         %[[EXT:[0-9a-zA-Z._]+]] = sext i48 %[[TR]] to i64
// LLVM:         store i64 %[[EXT]], ptr %[[BSLOT:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[RE:[0-9a-zA-Z._]+]] = load i64, ptr %[[BSLOT]], align 8
// LLVM:         %[[TR2:[0-9a-zA-Z._]+]] = trunc i64 %[[RE]] to i48
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM:         %[[EXT2:[0-9a-zA-Z._]+]] = sext i48 %[[TR2]] to i64
// LLVM:         store i64 %[[EXT2]], ptr %[[CSLOT:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[ARG:[0-9a-zA-Z._]+]] = load i64, ptr %[[CSLOT]], align 8
// LLVM:         call i32 (i64, ...) @vf(i64 %[[PV]], i64 noundef %[[ARG]])

// A width between 65 and 127 coerces to a register pair, and both halves are
// passed through the ellipsis.
int ell_bitint96(Pair2 p, _BitInt(96) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint96(%arg0: !u64i loc({{.+}}), %arg1: !u64i loc({{.+}}), %arg2: !u64i loc({{.+}})) -> !s32i
// CIR:         %[[IN0:[0-9]+]] = cir.get_member %[[IN:[0-9]+]][0] {name = ""} : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!u64i>
// CIR:         cir.store %arg1, %[[IN0]] : !u64i, !cir.ptr<!u64i>
// CIR:         %[[IN1:[0-9]+]] = cir.get_member %[[IN]][1] {name = ""} : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!u64i>
// CIR:         cir.store %arg2, %[[IN1]] : !u64i, !cir.ptr<!u64i>
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         %[[OUT0:[0-9]+]] = cir.get_member %[[OUT:[0-9]+]][0] {name = ""} : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!u64i>
// CIR:         %[[B0:[0-9]+]] = cir.load %[[OUT0]] : !cir.ptr<!u64i>, !u64i
// CIR:         %[[OUT1:[0-9]+]] = cir.get_member %[[OUT]][1] {name = ""} : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!u64i>
// CIR:         %[[B1:[0-9]+]] = cir.load %[[OUT1]] : !cir.ptr<!u64i>, !u64i
// CIR:         cir.call @vf(%[[PV]], %[[B0]], %[[B1]]) : (!u64i, !u64i, !u64i) -> !s32i

// LLVM-CIR-LABEL: define dso_local i32 @ell_bitint96(
// LLVM-CIR-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i64 %[[B0:[0-9a-zA-Z._]+]], i64 %[[B1:[0-9a-zA-Z._]+]])
// LLVM-OGCG-LABEL: define dso_local i32 @ell_bitint96(
// LLVM-OGCG-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i64 noundef %[[B0:[0-9a-zA-Z._]+]], i64 noundef %[[B1:[0-9a-zA-Z._]+]])
// LLVM:         %[[S0:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[SLOT:[0-9a-zA-Z._]+]], i32 0, i32 0
// LLVM:         store i64 %[[B0]], ptr %[[S0]], align 8
// LLVM:         %[[S1:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[SLOT]], i32 0, i32 1
// LLVM:         store i64 %[[B1]], ptr %[[S1]], align 8
// LLVM:         %[[TR:[0-9a-zA-Z._]+]] = trunc i128 %{{[0-9a-zA-Z._]+}} to i96
// LLVM:         %[[EXT:[0-9a-zA-Z._]+]] = sext i96 %[[TR]] to i128
// LLVM:         store i128 %[[EXT]], ptr %[[BSLOT:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[RE:[0-9a-zA-Z._]+]] = load i128, ptr %[[BSLOT]], align 8
// LLVM:         %[[TR2:[0-9a-zA-Z._]+]] = trunc i128 %[[RE]] to i96
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM:         %[[EXT2:[0-9a-zA-Z._]+]] = sext i96 %[[TR2]] to i128
// LLVM:         store i128 %[[EXT2]], ptr %[[COERCE:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[C0:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[COERCE]], i32 0, i32 0
// LLVM:         %[[A0:[0-9a-zA-Z._]+]] = load i64, ptr %[[C0]], align 8
// LLVM:         %[[C1:[0-9a-zA-Z._]+]] = getelementptr inbounds nuw { i64, i64 }, ptr %[[COERCE]], i32 0, i32 1
// LLVM:         %[[A1:[0-9a-zA-Z._]+]] = load i64, ptr %[[C1]], align 8
// LLVM-CIR:     call i32 (i64, ...) @vf(i64 %[[PV]], i64 %[[A0]], i64 %[[A1]])
// LLVM-OGCG:    call i32 (i64, ...) @vf(i64 %[[PV]], i64 noundef %[[A0]], i64 noundef %[[A1]])

// At exactly 128 bits it stays in its natural type.
int ell_bitint128(Pair2 p, _BitInt(128) b) { return vf(p, b); }

// CIR-LABEL: cir.func {{.*}}@ell_bitint128(%arg0: !u64i loc({{.+}}), %arg1: !s128i_bitint {llvm.noundef} loc({{.+}})) -> !s32i
// CIR:         cir.store %arg1, %[[BSLOT:[0-9]+]] : !s128i_bitint, !cir.ptr<!s128i_bitint>
// CIR:         %[[BV:[0-9]+]] = cir.load align(8) %[[BSLOT]] : !cir.ptr<!s128i_bitint>, !s128i_bitint
// CIR:         %[[PV:[0-9]+]] = cir.load %{{[0-9]+}} : !cir.ptr<!u64i>, !u64i
// CIR:         cir.call @vf(%[[PV]], %[[BV]]) : (!u64i, !s128i_bitint {llvm.noundef}) -> !s32i

// LLVM-LABEL: define dso_local i32 @ell_bitint128(
// LLVM-SAME:    i64 %[[P:[0-9a-zA-Z._]+]], i128 noundef %[[B:[0-9a-zA-Z._]+]])
// LLVM:         store i128 %[[B]], ptr %[[BSLOT:[0-9a-zA-Z._]+]], align 8
// LLVM:         %[[BV:[0-9a-zA-Z._]+]] = load i128, ptr %[[BSLOT]], align 8
// LLVM:         %[[PV:[0-9a-zA-Z._]+]] = load i64, ptr %{{[0-9a-zA-Z._]+}}, align
// LLVM:         call i32 (i64, ...) @vf(i64 %[[PV]], i128 noundef %[[BV]])
