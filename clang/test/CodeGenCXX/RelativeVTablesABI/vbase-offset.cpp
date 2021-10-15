// Check that the pointer adjustment from the virtual base offset is loaded as a
// 32-bit int.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-fuchsia -S -o - -emit-llvm | FileCheck %s

// CHECK-LABEL: @_ZTv0_n12_N7Derived1fEi(
// CHECK-NEXT:  entry:
// CHECK:        [[this:%.+]] = bitcast %class.Derived* %this1 to i8*
// CHECK-NEXT:   [[this2:%.+]] = bitcast i8* [[this]] to i8**
// CHECK-NEXT:   [[vtable:%.+]] = load i8*, i8** [[this2]], align 8
// CHECK-NEXT:   [[vbase_offset_ptr:%.+]] = getelementptr inbounds i8, i8* [[vtable]], i64 -12
// CHECK-NEXT:   [[vbase_offset_ptr2:%.+]] = bitcast i8* [[vbase_offset_ptr]] to i32*
// CHECK-NEXT:   [[vbase_offset:%.+]] = load i32, i32* [[vbase_offset_ptr2]], align 4
// CHECK-NEXT:   [[adj_this:%.+]] = getelementptr inbounds i8, i8* [[this]], i32 [[vbase_offset]]
// CHECK-NEXT:   [[adj_this2:%.+]] = bitcast i8* [[adj_this]] to %class.Derived*
// CHECK:        [[call:%.+]] = tail call noundef i32 @_ZN7Derived1fEi(%class.Derived* noundef{{[^,]*}} [[adj_this2]], i32 noundef {{.*}})
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
