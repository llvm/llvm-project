// RUN: %clang_cc1 -triple aarch64-linux-gnu -fblocks -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -fblocks -fexperimental-abi-lowering -emit-llvm -o - %s 2>&1 | FileCheck %s --implicit-check-not="not yet implemented"

// Objective-C object pointers and block pointers are pointer representations,
// but they are not ordinary C/C++ pointers or references and therefore use
// integer coercions for aggregate arguments.

@class Object;

typedef struct {
  id value;
} ObjCIdAgg;
void arg_objc_id(ObjCIdAgg a) {}
// CHECK: define{{.*}} void @arg_objc_id(i64 %{{.*}})

typedef struct {
  Class value;
} ObjCClassAgg;
void arg_objc_class(ObjCClassAgg a) {}
// CHECK: define{{.*}} void @arg_objc_class(i64 %{{.*}})

typedef struct {
  SEL value;
} ObjCSelAgg;
void arg_objc_sel(ObjCSelAgg a) {}
// SEL canonicalizes to an ordinary pointer type.
// CHECK: define{{.*}} void @arg_objc_sel(ptr %{{.*}})

typedef struct {
  Object *value;
} ObjCObjectPointerAgg;
void arg_objc_object_pointer(ObjCObjectPointerAgg a) {}
// CHECK: define{{.*}} void @arg_objc_object_pointer(i64 %{{.*}})

typedef struct {
  id first;
  id second;
} ObjCIdPairAgg;
void arg_objc_id_pair(ObjCIdPairAgg a) {}
// CHECK: define{{.*}} void @arg_objc_id_pair([2 x i64] %{{.*}})

typedef struct {
  void (^value)(void);
} BlockPointerAgg;
void arg_block_pointer(BlockPointerAgg a) {}
// CHECK: define{{.*}} void @arg_block_pointer(i64 %{{.*}})
