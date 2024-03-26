// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 -HV 2016 %s 2>&1  | FileCheck -check-prefix=both_invalid %s
// RUN: not %clang_dxc -enable-16bit-types -T lib_6_4 -HV 2017 %s 2>&1 | FileCheck -check-prefix=HV_invalid %s
// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 /HV 2021 %s 2>&1  | FileCheck -check-prefix=TP_invalid %s
// RUN: %clang_dxc -enable-16bit-types -T lib_6_4 /HV 2021 %s 2>&1 -###   | FileCheck -check-prefix=valid %s


// both_invalid: error: '-enable-16bit-types' option only valid when target shader model [-T] is >= 6.2 and HLSL Version [-HV] is >= hlsl2018, but shader model is '6.0' and HLSL Version is 'hlsl2016'
// HV_invalid: error: '-enable-16bit-types' option only valid when target shader model [-T] is >= 6.2 and HLSL Version [-HV] is >= hlsl2018, but shader model is '6.4' and HLSL Version is 'hlsl2017'
// TP_invalid: error: '-enable-16bit-types' option only valid when target shader model [-T] is >= 6.2 and HLSL Version [-HV] is >= hlsl2018, but shader model is '6.0' and HLSL Version is 'hlsl2021'

// valid: "dxil-unknown-shadermodel6.4-library"
// valid-SAME: "-std=hlsl2021"
// valid-SAME: "-fnative-half-type"

[numthreads(1,1,1)]
void main()
{
  return;
}

