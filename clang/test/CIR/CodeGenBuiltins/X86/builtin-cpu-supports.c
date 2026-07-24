// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
//
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=OGCG

// Test that we have the structure definition, the gep offsets, the name of the
// global, the bit grab, and the icmp correct.
extern void a(const char *);

// CIR-LABEL: cir.func no_inline dso_local @main() -> !s32i
// CIR: cir.call @__cpu_indicator_init()
// CIR: [[TRUE:%.*]] = cir.const #true
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_model : !cir.ptr<!rec_anon_struct>
// CIR-NEXT: [[CPUTYPE:%.]] = cir.get_member [[GLOBAL]][3] {name = "__cpu_feature"} : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.array<!u32i x 1>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: {{.*}} = cir.ptr_stride [[CPUTYPE]], [[IDX]] : (!cir.ptr<!cir.array<!u32i x 1>>, !u32i) -> !cir.ptr<!cir.array<!u32i x 1>>
// CIR: [[VALUE0:%.]] = cir.load align(4) {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE]] : !cir.bool
// CIR: [[TRUE2:%.*]] = cir.const #true
// CIR: [[GLOBAL2:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX2:%.]] = cir.const #cir.int<0> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL2]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX2]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE2]] : !cir.bool

// LLVM-LABEL: define dso_local i32 @main(
// LLVM-SAME: ) #[[ATTR0:[0-9]+]] {
// LLVM-NEXT:    [[RETVAL:%.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    call void @__cpu_indicator_init()
// LLVM:         [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 256
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 256
// LLVM-NEXT:    [[TMP3:%.*]] = and i1 [[TMP2]], true
// LLVM-NEXT:    br i1 [[TMP3]], label [[IF_THEN:%.*]], label [[IF_END:%.*]]
// LLVM:    [[TMP4:%.*]] = load i32, ptr @__cpu_features2, align 4
// LLVM-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], 1
// LLVM-NEXT:    [[TMP6:%.*]] = icmp eq i32 [[TMP5]], 1
// LLVM-NEXT:    [[TMP7:%.*]] = and i1 [[TMP6]], true
// LLVM-NEXT:    br i1 [[TMP7]], label [[IF_THEN1:%.*]], label [[IF_END2:%.*]]
// LLVM:    store i32 0, ptr [[RETVAL]], align 4

// OGCG-LABEL: define dso_local i32 @main(
// OGCG-SAME: ) #[[ATTR0:[0-9]+]] {
// OGCG-NEXT:  entry:
// OGCG-NEXT:    [[RETVAL:%.*]] = alloca i32, align 4
// OGCG-NEXT:    store i32 0, ptr [[RETVAL]], align 4
// OGCG-NEXT:    call void @__cpu_indicator_init()
// OGCG-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_model, i64 12), align 4
// OGCG-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 256
// OGCG-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 256
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    br i1 [[TMP3]], label [[IF_THEN:%.*]], label [[IF_END:%.*]]
// OGCG:       if.then:
// OGCG-NEXT:    call void @a(ptr noundef @.str)
// OGCG-NEXT:    br label [[IF_END]]
// OGCG:       if.end:
// OGCG-NEXT:    [[TMP4:%.*]] = load i32, ptr @__cpu_features2, align 4
// OGCG-NEXT:    [[TMP5:%.*]] = and i32 [[TMP4]], 1
// OGCG-NEXT:    [[TMP6:%.*]] = icmp eq i32 [[TMP5]], 1
// OGCG-NEXT:    [[TMP7:%.*]] = and i1 true, [[TMP6]]
// OGCG-NEXT:    br i1 [[TMP7]], label [[IF_THEN1:%.*]], label [[IF_END2:%.*]]
// OGCG:       if.then1:
// OGCG-NEXT:    call void @a(ptr noundef @.str.1)
// OGCG-NEXT:    br label [[IF_END2]]
// OGCG:       if.end2:
// OGCG-NEXT:    ret i32 0
int main(void) {
  __builtin_cpu_init();

  if (__builtin_cpu_supports("sse4.2"))
    a("sse4.2");


  if (__builtin_cpu_supports("gfni"))
    a("gfni");

  return 0;
}

// CIR-LABEL: cir.func no_inline dso_local @baseline() -> !s32i
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2147483648> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE2]] : !cir.bool

// LLVM-LABEL: define dso_local i32 @baseline(
// LLVM-SAME: ) #[[ATTR0]] {
// LLVM-NEXT:    [[RETVAL:%.]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], -2147483648
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], -2147483648
// LLVM-NEXT:    [[TMP3:%.*]] = and i1 [[TMP2]], true
// LLVM-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// LLVM-NEXT:    store i32 [[CONV]], ptr [[RETVAL]], align 4
// LLVM-NEXT:    [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM-NEXT:    ret i32 [[RET]]

// OGCG-LABEL: define dso_local i32 @baseline(
// OGCG-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// OGCG-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 4), align 4
// OGCG-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], -2147483648
// OGCG-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], -2147483648
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
int baseline() { return __builtin_cpu_supports("x86-64"); }

// CIR-LABEL: cir.func no_inline dso_local @v2() -> !s32i
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<1> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE2]] : !cir.bool

// LLVM-LABEL: define dso_local i32 @v2(
// LLVM-SAME: ) #[[ATTR0]] {
// LLVM-NEXT:    [[RETVAL:%.]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 1
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 1
// LLVM-NEXT:    [[TMP3:%.*]] = and i1 [[TMP2]], true
// LLVM-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// LLVM-NEXT:    store i32 [[CONV]], ptr [[RETVAL]], align 4
// LLVM-NEXT:    [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM-NEXT:    ret i32 [[RET]]

// OGCG-LABEL: define dso_local i32 @v2(
// OGCG-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// OGCG-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// OGCG-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 1
// OGCG-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 1
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
int v2() { return __builtin_cpu_supports("x86-64-v2"); }

// CIR-LABEL: cir.func no_inline dso_local @v3() -> !s32i
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE2]] : !cir.bool

// LLVM-LABEL: define dso_local i32 @v3(
// LLVM-SAME: ) #[[ATTR0]] {
// LLVM-NEXT:    [[RETVAL:%.]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 2
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 2
// LLVM-NEXT:    [[TMP3:%.*]] = and i1 [[TMP2]], true
// LLVM-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// LLVM-NEXT:    store i32 [[CONV]], ptr [[RETVAL]], align 4
// LLVM-NEXT:    [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM-NEXT:    ret i32 [[RET]]

// OGCG-LABEL: define dso_local i32 @v3(
// OGCG-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// OGCG-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// OGCG-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 2
// OGCG-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 2
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
int v3() { return __builtin_cpu_supports("x86-64-v3"); }

// CIR-LABEL: cir.func no_inline dso_local @v4() -> !s32i
// CIR: [[GLOBAL:%.]] = cir.get_global @__cpu_features2 : !cir.ptr<!cir.array<!u32i x 3>>
// CIR-NEXT: [[IDX:%.]] = cir.const #cir.int<2> : !u32i
// CIR-NEXT: [[ARRPTR:%.]] = cir.cast array_to_ptrdecay [[GLOBAL]] : !cir.ptr<!cir.array<!u32i x 3>> -> !cir.ptr<!u32i>
// CIR-NEXT: [[VALPTR:%.]] = cir.ptr_stride [[ARRPTR]], [[IDX]] : (!cir.ptr<!u32i>, !u32i) -> !cir.ptr<!u32i> 
// CIR-NEXT: [[VALUE0:%.]] = cir.load align(4) [[VALPTR]] : !cir.ptr<!u32i>, !u32i
// CIR-NEXT: [[MASK:%.]] = cir.const #cir.int<4> : !u32i
// CIR-NEXT: [[VALUE1:%.]] = cir.and [[VALUE0]], [[MASK]] : !u32i
// CIR-NEXT: [[RES:%.]] = cir.cmp eq [[VALUE1]], [[MASK]] : !u32i
// CIR-NEXT: cir.and [[RES]], [[TRUE2]] : !cir.bool

// LLVM-LABEL: define dso_local i32 @v4(
// LLVM-SAME: ) #[[ATTR0]] {
// LLVM-NEXT:    [[RETVAL:%.]] = alloca i32, i64 1, align 4
// LLVM-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// LLVM-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 4
// LLVM-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 4
// LLVM-NEXT:    [[TMP3:%.*]] = and i1 [[TMP2]], true
// LLVM-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// LLVM-NEXT:    store i32 [[CONV]], ptr [[RETVAL]], align 4
// LLVM-NEXT:    [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM-NEXT:    ret i32 [[RET]]

// OGCG-LABEL: define dso_local i32 @v4(
// OGCG-SAME: ) #[[ATTR0]] {
// OGCG-NEXT:  entry:
// OGCG-NEXT:    [[TMP0:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr @__cpu_features2, i64 8), align 4
// OGCG-NEXT:    [[TMP1:%.*]] = and i32 [[TMP0]], 4
// OGCG-NEXT:    [[TMP2:%.*]] = icmp eq i32 [[TMP1]], 4
// OGCG-NEXT:    [[TMP3:%.*]] = and i1 true, [[TMP2]]
// OGCG-NEXT:    [[CONV:%.*]] = zext i1 [[TMP3]] to i32
// OGCG-NEXT:    ret i32 [[CONV]]
int v4() { return __builtin_cpu_supports("x86-64-v4"); }

