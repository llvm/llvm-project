// RUN: %clang_dxc -T lib_6_4 -all-resources-bound %s 2>&1 -### | FileCheck -check-prefix=ARB %s
// RUN: %clang_dxc -T lib_6_4 %s 2>&1 -### | FileCheck -check-prefix=NO_ARB %s

// ARB: "hlsl-all-resources-bound"
// NO_ARB-NOT: "hlsl-all-resources-bound"
// assert expected CC1 option is present
float4 main(float4 a : A) : SV_TARGET
{
  return -a.yxxx;
}
