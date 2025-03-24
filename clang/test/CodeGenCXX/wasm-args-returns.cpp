// RUN: %clang_cc1 -no-enable-noundef-analysis -O1 -triple wasm32-unknown-unknown -emit-llvm -o - %s \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -O1 -triple wasm64-unknown-unknown -emit-llvm -o - %s \
// RUN:   | FileCheck %s

#define concat_(x, y) x##y
#define concat(x, y) concat_(x, y)

#define test(T)                \
  T forward(T x) { return x; } \
  void use(T x);               \
  T concat(def_, T)(void);     \
  void concat(test_, T)(void) { use(concat(def_, T)()); }

struct one_field {
  double d;
};
test(one_field);
// CHECK: define double @_Z7forward9one_field(double returned %{{.*}})
//
// CHECK: define void @_Z14test_one_fieldv()
// CHECK: %[[call:.*]] = tail call double @_Z13def_one_fieldv()
// CHECK: call void @_Z3use9one_field(double %[[call]])
// CHECK: ret void
//
// CHECK: declare void @_Z3use9one_field(double)
// CHECK: declare double @_Z13def_one_fieldv()

struct two_fields {
  double d, e;
};
test(two_fields);
// CHECK: define void @_Z7forward10two_fields(ptr dead_on_unwind noalias writable writeonly sret(%struct.two_fields) align 8 captures(none) initializes((0, 16)) %{{.*}}, ptr readonly byval(%struct.two_fields) align 8 captures(none) %{{.*}})
//
// CHECK: define void @_Z15test_two_fieldsv()
// CHECK: %[[tmp:.*]] = alloca %struct.two_fields, align 8
// CHECK: call void @_Z14def_two_fieldsv(ptr dead_on_unwind nonnull writable sret(%struct.two_fields) align 8 %[[tmp]])
// CHECK: call void @_Z3use10two_fields(ptr nonnull byval(%struct.two_fields) align 8 %[[tmp]])
// CHECK: ret void
//
// CHECK: declare void @_Z3use10two_fields(ptr byval(%struct.two_fields) align 8)
// CHECK: declare void @_Z14def_two_fieldsv(ptr dead_on_unwind writable sret(%struct.two_fields) align 8)

struct copy_ctor {
  double d;
  copy_ctor(copy_ctor const &);
};
test(copy_ctor);
// CHECK: define void @_Z7forward9copy_ctor(ptr dead_on_unwind noalias {{[^,]*}} sret(%struct.copy_ctor) align 8 %{{.*}}, ptr nonnull %{{.*}})
//
// CHECK: declare ptr @_ZN9copy_ctorC1ERKS_(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align 8 dereferenceable(8))
//
// CHECK: define void @_Z14test_copy_ctorv()
// CHECK: %[[tmp:.*]] = alloca %struct.copy_ctor, align 8
// CHECK: call void @_Z13def_copy_ctorv(ptr dead_on_unwind nonnull writable sret(%struct.copy_ctor) align 8 %[[tmp]])
// CHECK: call void @_Z3use9copy_ctor(ptr nonnull %[[tmp]])
// CHECK: ret void
//
// CHECK: declare void @_Z3use9copy_ctor(ptr)
// CHECK: declare void @_Z13def_copy_ctorv(ptr dead_on_unwind writable sret(%struct.copy_ctor) align 8)

struct __attribute__((aligned(16))) aligned_copy_ctor {
  double d, e;
  aligned_copy_ctor(aligned_copy_ctor const &);
};
test(aligned_copy_ctor);
// CHECK: define void @_Z7forward17aligned_copy_ctor(ptr dead_on_unwind noalias {{[^,]*}} sret(%struct.aligned_copy_ctor) align 16 %{{.*}}, ptr nonnull %{{.*}})
//
// CHECK: declare ptr @_ZN17aligned_copy_ctorC1ERKS_(ptr {{[^,]*}} returned {{[^,]*}}, ptr nonnull align 16 dereferenceable(16))
//
// CHECK: define void @_Z22test_aligned_copy_ctorv()
// CHECK: %[[tmp:.*]] = alloca %struct.aligned_copy_ctor, align 16
// CHECK: call void @_Z21def_aligned_copy_ctorv(ptr dead_on_unwind nonnull writable sret(%struct.aligned_copy_ctor) align 16 %[[tmp]])
// CHECK: call void @_Z3use17aligned_copy_ctor(ptr nonnull %[[tmp]])
// CHECK: ret void
//
// CHECK: declare void @_Z3use17aligned_copy_ctor(ptr)
// CHECK: declare void @_Z21def_aligned_copy_ctorv(ptr dead_on_unwind writable sret(%struct.aligned_copy_ctor) align 16)

struct empty {};
test(empty);
// CHECK: define void @_Z7forward5empty()
//
// CHECK: define void @_Z10test_emptyv()
// CHECK: call void @_Z9def_emptyv()
// CHECK: call void @_Z3use5empty()
// CHECK: ret void
//
// CHECK: declare void @_Z3use5empty()
// CHECK: declare void @_Z9def_emptyv()

struct one_bitfield {
  int d : 3;
};
test(one_bitfield);
// CHECK: define i32 @_Z7forward12one_bitfield(i32 returned %{{.*}})
//
// CHECK: define void @_Z17test_one_bitfieldv()
// CHECK: %[[call:.*]] = tail call i32 @_Z16def_one_bitfieldv()
// CHECK: call void @_Z3use12one_bitfield(i32 %[[call]])
// CHECK: ret void
//
// CHECK: declare void @_Z3use12one_bitfield(i32)
// CHECK: declare i32 @_Z16def_one_bitfieldv()
