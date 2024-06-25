/// Check that the offset to top calculation is adjusted to account for the
/// omitted RTTI entry.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-linux-gnu -fexperimental-omit-vtable-rtti -fno-rtti -o - -emit-llvm | FileCheck -check-prefixes=POINTER %s
// RUN: %clang_cc1 %s -triple=aarch64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -fexperimental-omit-vtable-rtti -fno-rtti -o - -emit-llvm | FileCheck -check-prefixes=RELATIVE %s

/// Some important things to check:
/// - The n16 here represents the virtual thunk size. Normally this would be 24
///   to represent 3 components (offset to top, RTTI component, vcall offset),
///   but since one 8-byte component is removed, this is now 16.
// POINTER-LABEL: @_ZTv0_n16_N7Derived1fEi(
// POINTER-NEXT:  entry:
// POINTER:        [[vtable:%.+]] = load ptr, ptr %this1, align 8

/// Same here - When getting the vbase offset, we subtract 2 pointer sizes
/// instead of 3.
// POINTER-NEXT:   [[vbase_offset_ptr:%.+]] = getelementptr inbounds i8, ptr [[vtable]], i64 -16
// POINTER-NEXT:   [[vbase_offset:%.+]] = load i64, ptr [[vbase_offset_ptr]], align 8
// POINTER-NEXT:   [[adj_this:%.+]] = getelementptr inbounds i8, ptr %this1, i64 [[vbase_offset]]
// POINTER:   [[call:%.+]] = tail call noundef i32 @_ZN7Derived1fEi(ptr noundef{{[^,]*}} [[adj_this]], i32 noundef {{.*}})
// POINTER:   ret i32 [[call]]

/// For relative vtables, it's almost the same except the offset sizes are
/// halved.
// RELATIVE-LABEL: @_ZTv0_n8_N7Derived1fEi(
// RELATIVE-NEXT:  entry:
// RELATIVE:        [[vtable:%.+]] = load ptr, ptr %this1, align 8
// RELATIVE-NEXT:   [[vbase_offset_ptr:%.+]] = getelementptr inbounds i8, ptr [[vtable]], i64 -8
// RELATIVE-NEXT:   [[vbase_offset:%.+]] = load i32, ptr [[vbase_offset_ptr]], align 4
// RELATIVE-NEXT:   [[adj_this:%.+]] = getelementptr inbounds i8, ptr %this1, i32 [[vbase_offset]]
// RELATIVE:        [[call:%.+]] = tail call noundef i32 @_ZN7Derived1fEi(ptr noundef{{[^,]*}} [[adj_this]], i32 noundef {{.*}})
// RELATIVE:        ret i32 [[call]]

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
