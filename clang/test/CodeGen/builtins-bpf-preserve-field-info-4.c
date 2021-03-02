// REQUIRES: bpf-registered-target
// RUN: %clang -target bpf -emit-llvm -S -g -Xclang -disable-llvm-passes %s -o - | FileCheck %s

#define _(x, y) (__builtin_preserve_enum_value((x), (y)))

enum AA {
  VAL0 = 0,
  VAL1 = 2,
  VAL2 = 0xffffffff80000000UL,
};
typedef enum { VAL00, VAL10 = -2, VAL11 = 0xffff8000, }  __BB;

unsigned unit1() {
  return _(*(enum AA *)VAL1, 0) + _(*(__BB *)VAL10, 1);
}

unsigned unit2() {
  return _(*(enum AA *)VAL2, 0) + _(*(__BB *)VAL11, 1);
}

unsigned unit3() {
  return _(*(enum AA *)VAL0, 0) + _(*(__BB *)VAL00, 1);
}

// CHECK: @0 = private unnamed_addr constant [7 x i8] c"VAL1:2\00", align 1
// CHECK: @1 = private unnamed_addr constant [9 x i8] c"VAL10:-2\00", align 1
// CHECK: @2 = private unnamed_addr constant [17 x i8] c"VAL2:-2147483648\00", align 1
// CHECK: @3 = private unnamed_addr constant [17 x i8] c"VAL11:4294934528\00", align 1
// CHECK: @4 = private unnamed_addr constant [7 x i8] c"VAL0:0\00", align 1
// CHECK: @5 = private unnamed_addr constant [8 x i8] c"VAL00:0\00", align 1

// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 0, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @0, i32 0, i32 0), i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM_AA:[0-9]+]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 1, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @1, i32 0, i32 0), i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_ENUM:[0-9]+]]

// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 2, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @2, i32 0, i32 0), i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM_AA]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 3, i8* getelementptr inbounds ([17 x i8], [17 x i8]* @3, i32 0, i32 0), i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_ENUM]]

// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 4, i8* getelementptr inbounds ([7 x i8], [7 x i8]* @4, i32 0, i32 0), i64 0), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[ENUM_AA]]
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 5, i8* getelementptr inbounds ([8 x i8], [8 x i8]* @5, i32 0, i32 0), i64 1), !dbg !{{[0-9]+}}, !llvm.preserve.access.index ![[TYPEDEF_ENUM]]

// CHECK: ![[ENUM_AA]] = !DICompositeType(tag: DW_TAG_enumeration_type, name: "AA"
// CHECK: ![[TYPEDEF_ENUM]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__BB"
