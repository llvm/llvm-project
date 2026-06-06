/// Check that -fexperimental-omit-vtable-rtti omits the RTTI component from
/// the vtable.

// RUN: %clang_cc1 %s -triple=aarch64-unknown-linux-gnu -fno-rtti -fexperimental-omit-vtable-rtti -o - -emit-llvm | FileCheck -check-prefixes=POINTER,RTTI %s
// RUN: %clang_cc1 %s -triple=aarch64-unknown-linux-gnu -fexperimental-relative-c++-abi-vtables -fno-rtti -fexperimental-omit-vtable-rtti -o - -emit-llvm | FileCheck -check-prefixes=RELATIVE,RTTI %s

/// Normally, the vtable would contain at least three components:
/// - An offset to top
/// - A pointer to the RTTI struct
/// - A virtual function
///
/// Now vtables should have just two components.
// POINTER: @_ZTV1A = unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr null, ptr @_ZN1A3fooEv] }, align 8
// RELATIVE: @_ZTV1A.local = internal unnamed_addr constant { [2 x i32] } { [2 x i32] [i32 0, i32 trunc (i64 sub (i64 ptrtoint (ptr dso_local_equivalent @_ZN1A3fooEv to i64), i64 ptrtoint (ptr getelementptr inbounds ({ [2 x i32] }, ptr @_ZTV1A.local, i32 0, i32 0, i32 1) to i64)) to i32)] }, align 4
// RELATIVE: @_ZTV1A = unnamed_addr alias { [2 x i32] }, ptr @_ZTV1A.local

/// None of these supplementary symbols should be emitted with -fno-rtti, but
/// as a sanity check lets make sure they're not emitted also.
// RTTI-NOT: @_ZTVN10__cxxabiv117__class_type_infoE
// RTTI-NOT: @_ZTS1A
// RTTI-NOT: @_ZTI1A

class A {
public:
  virtual void foo();
};

void A::foo() {}

void A_foo(A *a) {
  a->foo();
}
