// RUN: %clang_cc1 -std=hlsl202x -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s


half3x3 write_pi(half3x3 A) {
    //CHECK:  [[MAT_INS:%.*]] = insertelement <9 x half> %{{[0-9]+}}, half 0xH4248, i32 7
    //CHECK-NEXT:  store <9 x half> [[MAT_INS]], ptr %{{.*}}, align 2
    A._m12 = 3.14;
    return A;
}

half read_1x1(half3x3 A) {
    //CHECK:  [[MAT_EXT:%.*]] = extractelement <9 x half> %{{[0-9]+}}, i32 0
    return A._11;
}
half read_m0x0(half3x3 A) {
    //CHECK:  [[MAT_EXT:%.*]] = extractelement <9 x half> %{{[0-9]+}}, i32 0
    return A._m00;
}

half read_3x3(half3x3 A) {
    //CHECK:  [[MAT_EXT:%.*]] = extractelement <9 x half> %{{[0-9]+}}, i32 8
    return A._33;
}
