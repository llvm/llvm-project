// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 -HV 2016 %s 2>&1  | FileCheck -check-prefix=both_invalid %s
// RUN: not %clang_dxc -enable-16bit-types -T lib_6_4 -HV 2017 %s 2>&1 | FileCheck -check-prefix=HV_invalid_2017 %s
// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 /HV 2021 %s 2>&1  | FileCheck -check-prefix=TP_invalid %s
// RUN: %clang_dxc -enable-16bit-types -T lib_6_4 /HV 2018 %s 2>&1 -###   | FileCheck -check-prefix=valid_2018 %s
// RUN: %clang_dxc -enable-16bit-types -T lib_6_4 /HV 2021 %s 2>&1 -###   | FileCheck -check-prefix=valid_2021 %s


// both_invalid: error: '-enable-16bit-types' option requires target HLSL Version >= 2018 and shader model >= 6.2, but HLSL Version is 'hlsl2016' and shader model is '6.0'
// HV_invalid_2017: error: '-enable-16bit-types' option requires target HLSL Version >= 2018 and shader model >= 6.2, but HLSL Version is 'hlsl2017' and shader model is '6.4'
// TP_invalid: error: '-enable-16bit-types' option requires target HLSL Version >= 2018 and shader model >= 6.2, but HLSL Version is 'hlsl2021' and shader model is '6.0'

// valid_2021: "dxilv1.4-unknown-shadermodel6.4-library"
// valid_2021-SAME: "-std=hlsl2021"
// valid_2021-SAME: "-fnative-half-type"

// valid_2018: "dxilv1.4-unknown-shadermodel6.4-library"
// valid_2018-SAME: "-std=hlsl2018"
// valid_2018-SAME: "-fnative-half-type"

[numthreads(1,1,1)]
void main()
{
  return;
}

