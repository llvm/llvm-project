// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s

struct Floats {
  float a, b;
};
struct AtomicFloats {
  _Atomic(float) a, b;
};
struct NestedAtomicFloats {
  struct AtomicFloats value;
};
typedef struct {
  char data[3];
} ThreeBytes;

// Two ordinary floats share an eightbyte and pass in an SSE register.
void take_floats(struct Floats s) {}

// CHECK-LABEL: define dso_local void @take_floats(
// CHECK-SAME: <2 x float> %{{.*}})

// A scalar atomic still has the value type's evaluation kind, so it is a
// direct float argument.
void take_atomic_float(_Atomic(float) value) {}

// CHECK-LABEL: define dso_local void @take_atomic_float(
// CHECK-SAME: float %{{.*}})

// Atomic fields classify Memory rather than inheriting the underlying float's
// SSE class, so the record is passed on the stack.
void take_atomic_floats(struct AtomicFloats s) {}

// CHECK-LABEL: define dso_local void @take_atomic_floats(
// CHECK-SAME: ptr noundef byval(%struct.AtomicFloats) align 8 %{{.*}})

// The same classification applies when that record is nested.
void take_nested_atomic_floats(struct NestedAtomicFloats s) {}

// CHECK-LABEL: define dso_local void @take_nested_atomic_floats(
// CHECK-SAME: ptr noundef byval(%struct.NestedAtomicFloats) align 8 %{{.*}})

// Atomic types preserve any size inflation relative to their value type.
void take_padded_atomic(_Atomic(ThreeBytes) value) {}

// CHECK-LABEL: define dso_local void @take_padded_atomic(
// CHECK-SAME: ptr noundef byval({ %struct.ThreeBytes, [1 x i8] }) align 8

// Aggregate returns use the same Memory classification.
struct AtomicFloats return_atomic_floats(void);
void call_return_atomic_floats(void) { return_atomic_floats(); }

// CHECK-LABEL: define dso_local void @call_return_atomic_floats(
// CHECK: call void @return_atomic_floats(
// CHECK-SAME: ptr dead_on_unwind writable sret(%struct.AtomicFloats) align 4
// CHECK: declare void @return_atomic_floats(
// CHECK-SAME: ptr dead_on_unwind writable sret(%struct.AtomicFloats) align 4)
