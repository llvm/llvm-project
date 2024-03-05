// RUN: %clang_dxc -T lib_6_4 -HV 2016 %s 2>&1 -###   | FileCheck -check-prefix=2016 %s
// RUN: %clang_dxc -T lib_6_4 -HV 2017 %s 2>&1 -###   | FileCheck -check-prefix=2017 %s
// RUN: %clang_dxc -T lib_6_4 /HV 2018 %s 2>&1 -###   | FileCheck -check-prefix=2018 %s
// RUN: %clang_dxc -T lib_6_4 /HV 2021 %s 2>&1 -###   | FileCheck -check-prefix=2021 %s

// 2016: "-std=hlsl2016"
// 2017: "-std=hlsl2017"
// 2018: "-std=hlsl2018"
// 2021: "-std=hlsl2021"
float4 main(float4 a : A) : SV_TARGET
{
  return -a.yxxx;
}