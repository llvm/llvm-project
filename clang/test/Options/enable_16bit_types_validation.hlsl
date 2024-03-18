// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 -HV 2016 %s 2>&1  | FileCheck -check-prefix=both_invalid %s
// RUN: not %clang_dxc -enable-16bit-types -T lib_6_4 -HV 2017 %s 2>&1 | FileCheck -check-prefix=HV_invalid %s
// RUN: not %clang_dxc -enable-16bit-types -T cs_6_0 /HV 2021 %s 2>&1  | FileCheck -check-prefix=TP_invalid %s
// RUN: %clang_dxc -enable-16bit-types -T lib_6_4 /HV 2021 %s 2>&1 -###   | FileCheck -check-prefix=valid %s


// both_invalid: error: enable_16bit_types option only valid when target shader model [-T] is >= 6.2 and Hlsl Version [-HV] is >= 2021
// HV_invalid: error: enable_16bit_types option only valid when target shader model [-T] is >= 6.2 and Hlsl Version [-HV] is >= 2021
// TP_invalid: error: enable_16bit_types option only valid when target shader model [-T] is >= 6.2 and Hlsl Version [-HV] is >= 2021
// valid: "dxil-unknown-shadermodel6.4-library"
// valid-SAME: "-std=hlsl2021"
// valid-SAME: "-fnative-half-type"

[numthreads(1,1,1)]
void main()
{
  return;
}

