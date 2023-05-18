// RUN: %clang_cc1 -emit-llvm -fblocks -o - -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 %s | FileCheck %s

// N.B.  This test verifies that two blocks which are otherwise
//       indistinguishable receive distinct manglings.
//       We had a bug where the first two blocks in the global block map could
//       get the same unqualified-block mangling because the logic to handle
//       block-ids believed it was handling Itanium-style discriminators.

template<typename T>
int tf() {
    return T::value;
}
int i1 = ^int {
    struct S { enum { value = 1 };};
    // CHECK-DAG: @_Z2tfIZUb_E1SEiv
    return tf<S>();
}();
int i2 = ^int(int p1) {
    struct S { enum { value = 2 };};
    // CHECK-DAG: @_Z2tfIZUb0_E1SEiv
    return tf<S>() + p1;
}(1);
