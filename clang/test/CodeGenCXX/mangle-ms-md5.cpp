// RUN: %clang_cc1 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck %s
// Define macros, using token pasting, to build up an identifier of any length
#define C_(P1, P2) P1##P2
#define C2(P1, P2) C_(P1, P2)
#define C4(P1, P2, P3, P4) C2(C2(P1, P2), C2(P3, P4))

#define X2(X) C2(X, X)
#define X4(X) X2(X2(X))
#define X8(X) X2(X4(X))
#define X16(X) X2(X8(X))
#define X32(X) X2(X16(X))
#define X64(X) X2(X32(X))
#define X128(X) X2(X64(X))
#define X256(X) X2(X128(X))
#define X512(X) X2(X256(X))
#define X1024(X) X2(X512(X))
#define X2048(X) X2(X1024(X))
#define X4096(X) X2(X2048(X))

#define X4095(X)                                \
    C2(C2(                                      \
      C4(X,       X2(X),   X4(X),    X8(X)),    \
      C4(X16(X),  X32(X),  X64(X),   X128(X))), \
      C4(X256(X), X512(X), X1024(X), X2048(X)))

int X4095(x);
#define Y4095 X4095(y)
// CHECK-DAG: @"??@bf7ea7b95f260b0b24e7f1e8fc8370ab@" = dso_local global i32 0, align 4

struct Y4095 {
  Y4095 ();
  virtual void f();
};
Y4095::Y4095() {}
// CHECK-DAG: @"??@a6a285da2eea70dba6b578022be61d81@??_R4@" = linkonce_odr constant %rtti.CompleteObjectLocator
// CHECK-DAG: @"??@a6a285da2eea70dba6b578022be61d81@" = unnamed_addr alias

// RUN: %clang_cc1 -DTHROW -fcxx-exceptions -fms-compatibility-version=18.0 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck --check-prefix=HAVECTOR %s
// RUN: %clang_cc1 -DTHROW -fcxx-exceptions -fms-compatibility-version=19.0 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck --check-prefix=OMITCTOR %s
// RUN: %clang_cc1 -DTHROW -fcxx-exceptions -fms-compatibility-version=19.10 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck --check-prefix=OMITCTOR %s
// RUN: %clang_cc1 -DTHROW -fcxx-exceptions -fms-compatibility-version=19.11 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck --check-prefix=OMITCTOR %s
// FIXME: Not known where between 19.11 and 19.14 this changed.
// RUN: %clang_cc1 -DTHROW -fcxx-exceptions -fms-compatibility-version=19.14 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck --check-prefix=HAVECTOR %s
// RUN: %clang_cc1 -DTHROW -fcxx-exceptions -fms-compatibility-version=19.20 -emit-llvm -o - -triple i686-pc-win32 %s | FileCheck --check-prefix=HAVECTOR %s
#ifdef THROW
void g() {
  throw Y4095();
// OMITCTOR: "_CT??@c14087f0ec22b387aea7c59083f4f546@4"
// HAVECTOR: "_CT??@c14087f0ec22b387aea7c59083f4f546@??@4ef4f8979c81f9d2224b32bf327e6bdf@4"
}
#endif

// Verify the threshold where md5 mangling kicks in
// Test an ident with 4088 characters, pre-hash, MangleName.size() is 4095
#define X4088(X)                                \
    C2(C2(                                      \
      C4(X,       X4(X),   X4(X),    X8(X)),    \
      C4(X8(X),   X32(X),  X64(X),   X128(X))), \
      C4(X256(X), X512(X), X1024(X), X2048(X)))
#define Z4088 X4088(z)
// Use initialization to verify mangled name association in the il
int X4088(z) = 1515;
// CHECK-DAG: @"?{{z+}}@@3HA" = dso_local global i32 1515, align 4

// Test an ident with 4089 characters, pre-hash, MangleName.size() is 4096
#define X4089(X)                                \
    C2(C2(                                      \
      C4(X2(X),   X4(X),   X4(X),    X8(X)),    \
      C4(X8(X),   X32(X),  X64(X),   X128(X))), \
      C4(X256(X), X512(X), X1024(X), X2048(X)))
// Use initialization to verify mangled name association in the il
int X4089(z) = 1717;
// CHECK-DAG: @"??@0269945400a3474730d6880df0967d8f@" = dso_local global i32 1717, align 4
