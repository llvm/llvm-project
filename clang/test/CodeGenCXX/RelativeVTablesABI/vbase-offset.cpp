// Check that the pointer adjustment from the virtual base offset is loaded as a
// 32-bit int.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -o - -emit-llvm | FileCheck %s

// CHECK-LABEL: @_ZTv0_n12_N7Derived1fEi(
// CHECK-NEXT:  entry:
// CHECK:        [[vtable:%.+]] = load ptr, ptr %this1, align 8
// CHECK-NEXT:   [[vbase_offset_ptr:%.+]] = getelementptr inbounds i8, ptr [[vtable]], i64 -12
// CHECK-NEXT:   [[vbase_offset:%.+]] = load i32, ptr [[vbase_offset_ptr]], align 4
// CHECK-NEXT:   [[adj_this:%.+]] = getelementptr inbounds i8, ptr %this1, i32 [[vbase_offset]]
// CHECK:        [[call:%.+]] = tail call noundef i32 @_ZN7Derived1fEi(ptr noundef{{[^,]*}} [[adj_this]], i32 noundef {{.*}})
// CHECK:        ret i32 [[call]]

class Base {
public:
  virtual int f(int x);

private:
  long x;
};

class Derived : public virtual Base {
public:
  virtual int f(int x);

private:
  long y;
};

int Base::f(int x) { return x + 1; }
int Derived::f(int x) { return x + 2; }
