// This is a clone of a file of the same name, but only the x86 parts, since
// that is all we support in CIR right now.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM,OGCG

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

// CIR-LABEL: cir.func{{.*}} @main() -> !s32i
// CIR-SAME: attributes {[[ATTRS:.*]]} {
// CIR:      cir.call @__cpu_indicator_init() : () -> ()
// CIR-NEXT: cir.scope {
// CIR-NEXT:   %[[CPU_MODEL:.*]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT:   %[[CPU_FEAT:.*]] = cir.get_member %[[CPU_MODEL]][3] {name = "__cpu_features"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT:   %[[FEAT0:.*]] = cir.get_element %[[CPU_FEAT]][%[[ZERO]] : !u32i] : !cir.ptr<!cir.array<!u32i x 1>> -> !cir.ptr<!u32i>
// CIR-NEXT:   %[[LOAD_FEAT0:.*]] = cir.load {{.*}} %[[FEAT0]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[MASK:.*]] = cir.const #cir.int<256> : !u32i
// CIR-NEXT:   %[[AND:.*]] = cir.and %[[LOAD_FEAT0]], %[[MASK]] : !u32i
// CIR-NEXT:   %[[RES:.*]] = cir.cmp eq %[[AND]], %[[MASK]] : !u32i
// CIR-NEXT:   cir.if %[[RES]] {
// CIR-NEXT:     %[[STR:.*]] = cir.get_global 
// CIR-NEXT:     %[[STR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[STR]] : !cir.ptr<!cir.array<!s8i x 7>> -> !cir.ptr<!s8i>
// CIR-NEXT:     cir.call @a(%[[STR_DECAY]]) : (!cir.ptr<!s8i> {llvm.noundef}) -> ()
// CIR-NEXT:   }
// CIR-NEXT: }
// CIR-NEXT: cir.scope {
// CIR-NEXT:   %[[CPU_FEAT2:.*]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT:   %[[FEAT2_0:.*]] = cir.get_element %[[CPU_FEAT2]][%[[ZERO]] : !u32i] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT:   %[[LOAD_FEAT2_0:.*]] = cir.load {{.*}} %[[FEAT2_0]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT:   %[[MASK:.*]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT:   %[[AND:.*]] = cir.and %[[LOAD_FEAT2_0]], %[[MASK]] : !u32i
// CIR-NEXT:   %[[RES:.*]] = cir.cmp eq %[[AND]], %[[MASK]] : !u32i
// CIR-NEXT:   cir.if %[[RES]] {
// CIR-NEXT:     %[[STR:.*]] = cir.get_global 
// CIR-NEXT:     %[[STR_DECAY:.*]] = cir.cast array_to_ptrdecay %[[STR]] : !cir.ptr<!cir.array<!s8i x 5>> -> !cir.ptr<!s8i>
// CIR-NEXT:     cir.call @a(%[[STR_DECAY]]) : (!cir.ptr<!s8i> {llvm.noundef}) -> ()
// CIR-NEXT:   }
// CIR-NEXT: }

// LLVM-LABEL: define dso_local i32 @main(
// LLVM-SAME: ) #[[ATTR0:[0-9]+]] {
// OGCG-NEXT:  entry:
// LLVM-NEXT:    [[RETVAL:%.*]] = alloca i32
// LLVM-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// LLVM-NEXT:    call void @__cpu_indicator_init()
//
// CIR leaves an extra branch/newline/label here.
// LLVMCIR-NEXT:     br label %[[BRANCH:.*]]
// LLVMCIR:          [[BRANCH]]:
//
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 256
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 256
//
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    br i1 [[TMP3]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
// LLVMCIR-NEXT: br i1 [[TMP2]], label %[[IF_THEN:.*]], label %[[IF_END:.*]]
//
// LLVM:       [[IF_THEN]]:
// LLVM-NEXT:    call void @a(ptr noundef @.str)
// CIR has a meaningless set of empty blocks here.
// LLVM:         br label %[[IF_END]]
// LLVM:       [[IF_END]]:
// CIR has more meaningless empty blocks here.
// LLVM:         [[TMP4:%.*]] = load i32, ptr @__cpu_features2, align 4
// LLVM-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], 1
// LLVM-NEXT:    [[TMP6:%.*]] = icmp eq i32 [[TMP5]], 1
//
// OGCG-NEXT:    [[TMP7:%.*]] = and i1 true, [[TMP6]]
// OGCG-NEXT:    br i1 [[TMP7]], label %[[IF_THEN1:.*]], label %[[IF_END1:.*]]
// LLVMCIR-NEXT: br i1 [[TMP6]], label %[[IF_THEN1:.*]], label %[[IF_END1:.*]]

//
// LLVM:       [[IF_THEN1]]:
// LLVM-NEXT:    call void @a(ptr noundef @.str.1)

int main(void) {
  __builtin_cpu_init();

  if (__builtin_cpu_supports("sse4.2"))
    a("sse4.2");


  if (__builtin_cpu_supports("gfni"))
    a("gfni");


  return 0;
}


// CIR-LABEL: cir.func{{.*}} @baseline() -> !s32i
// CIR-SAME: attributes {[[ATTRS]]} {
// CIR-NEXT: %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR-NEXT: %[[CPU_FEAT2:.*]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: %[[FEAT2_ELT:.*]] = cir.get_element %[[CPU_FEAT2]][%[[ONE]] : !u32i] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: %[[FEAT2_ELT_LOAD:.*]] = cir.load {{.*}} %[[FEAT2_ELT]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[MASK:.*]] = cir.const #cir.int<2147483648> : !u32i
// CIR-NEXT: %[[AND:.*]] = cir.and %[[FEAT2_ELT_LOAD]], %[[MASK]] : !u32i
// CIR-NEXT: %[[RES:.*]] = cir.cmp eq %[[AND]], %[[MASK]] : !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast bool_to_int %[[RES]] : !cir.bool -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i

// LLVM-LABEL: define dso_local i32 @baseline(
// LLVM-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// LLVMCIR-NEXT: [[RETVAL:%.*]] = alloca i32
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], -2147483648
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], -2147483648
//
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
//
// LLVMCIR-NEXT: [[EXT:%.*]] = zext i1 [[TMP2]] to i32
// LLVMCIR-NEXT: store i32 [[EXT]], ptr [[RETVAL]]
// LLVMCIR-NEXT: [[LOAD_RET:%.*]] = load i32, ptr [[RETVAL]]
// LLVMCIR-NEXT: ret i32 [[LOAD_RET]]
//
int baseline() { return __builtin_cpu_supports("x86-64"); }

// CIR-LABEL: cir.func{{.*}} @v2() -> !s32i
// CIR-SAME: attributes {[[ATTRS]]} {
// CIR-NEXT: %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR-NEXT: %[[CPU_FEAT2:.*]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: %[[FEAT2_ELT:.*]] = cir.get_element %[[CPU_FEAT2]][%[[TWO]] : !u32i] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: %[[FEAT2_ELT_LOAD:.*]] = cir.load {{.*}} %[[FEAT2_ELT]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[MASK:.*]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: %[[AND:.*]] = cir.and %[[FEAT2_ELT_LOAD]], %[[MASK]] : !u32i
// CIR-NEXT: %[[RES:.*]] = cir.cmp eq %[[AND]], %[[MASK]] : !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast bool_to_int %[[RES]] : !cir.bool -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i

// LLVM-LABEL: define dso_local i32 @v2(
// LLVM-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// LLVMCIR-NEXT: [[RETVAL:%.*]] = alloca i32
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 1
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 1
//
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
//
// LLVMCIR-NEXT: [[EXT:%.*]] = zext i1 [[TMP2]] to i32
// LLVMCIR-NEXT: store i32 [[EXT]], ptr [[RETVAL]]
// LLVMCIR-NEXT: [[LOAD_RET:%.*]] = load i32, ptr [[RETVAL]]
// LLVMCIR-NEXT: ret i32 [[LOAD_RET]]
//
int v2() { return __builtin_cpu_supports("x86-64-v2"); }

// CIR-LABEL: cir.func{{.*}} @v3() -> !s32i
// CIR-SAME: attributes {[[ATTRS]]} {
// CIR-NEXT: %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR-NEXT: %[[CPU_FEAT2:.*]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: %[[FEAT2_ELT:.*]] = cir.get_element %[[CPU_FEAT2]][%[[TWO]] : !u32i] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: %[[FEAT2_ELT_LOAD:.*]] = cir.load {{.*}} %[[FEAT2_ELT]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[MASK:.*]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: %[[AND:.*]] = cir.and %[[FEAT2_ELT_LOAD]], %[[MASK]] : !u32i
// CIR-NEXT: %[[RES:.*]] = cir.cmp eq %[[AND]], %[[MASK]] : !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast bool_to_int %[[RES]] : !cir.bool -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i

// LLVM-LABEL: define dso_local i32 @v3(
// LLVM-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// LLVMCIR-NEXT: [[RETVAL:%.*]] = alloca i32
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 2
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 2
//
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
//
// LLVMCIR-NEXT: [[EXT:%.*]] = zext i1 [[TMP2]] to i32
// LLVMCIR-NEXT: store i32 [[EXT]], ptr [[RETVAL]]
// LLVMCIR-NEXT: [[LOAD_RET:%.*]] = load i32, ptr [[RETVAL]]
// LLVMCIR-NEXT: ret i32 [[LOAD_RET]]
//
int v3() { return __builtin_cpu_supports("x86-64-v3"); }

// CIR-LABEL: cir.func{{.*}} @v4() -> !s32i
// CIR-SAME: attributes {[[ATTRS]]} {
// CIR-NEXT: %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!s32i>
// CIR-NEXT: %[[CPU_FEAT2:.*]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: %[[TWO:.*]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: %[[FEAT2_ELT:.*]] = cir.get_element %[[CPU_FEAT2]][%[[TWO]] : !u32i] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: %[[FEAT2_ELT_LOAD:.*]] = cir.load {{.*}} %[[FEAT2_ELT]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: %[[MASK:.*]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: %[[AND:.*]] = cir.and %[[FEAT2_ELT_LOAD]], %[[MASK]] : !u32i
// CIR-NEXT: %[[RES:.*]] = cir.cmp eq %[[AND]], %[[MASK]] : !u32i
// CIR-NEXT: %[[CAST:.*]] = cir.cast bool_to_int %[[RES]] : !cir.bool -> !s32i
// CIR-NEXT: cir.store %[[CAST]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT: %[[LOAD_RET:.*]] = cir.load %[[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT: cir.return %[[LOAD_RET]] : !s32i

// LLVM-LABEL: define dso_local i32 @v4(
// LLVM-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// LLVMCIR-NEXT: [[RETVAL:%.*]] = alloca i32
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 4
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 4
//
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
//
// LLVMCIR-NEXT: [[EXT:%.*]] = zext i1 [[TMP2]] to i32
// LLVMCIR-NEXT: store i32 [[EXT]], ptr [[RETVAL]]
// LLVMCIR-NEXT: [[LOAD_RET:%.*]] = load i32, ptr [[RETVAL]]
// LLVMCIR-NEXT: ret i32 [[LOAD_RET]]
int v4() { return __builtin_cpu_supports("x86-64-v4"); }
