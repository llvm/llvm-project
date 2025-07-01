// RUN: %clang_dxc -T lib_6_4 -Gis %s 2>&1 -### | FileCheck -check-prefix=Gis %s
// RUN: %clang_dxc -T lib_6_4 %s 2>&1 -### | FileCheck -check-prefix=NO_Gis %s
// RUN: not %clang_dxc -T lib_6_4 /Gis gibberish -### %s 2>&1 | FileCheck -check-prefix=CHECK-ERR %s

// Gis: "-fmath-errno" "-ffp-contract=off" "-frounding-math" "-ffp-exception-behavior=strict" "-complex-range=full"
// assert expected floating point options are present
// NO_Gis-NOT: "-fmath-errno" "-ffp-contract=off" "-frounding-math" "-ffp-exception-behavior=strict" "-complex-range=full"
// CHECK-ERR: error: no such file or directory: 'gibberish'
float4 main(float4 a : A) : SV_TARGET
{
  return -a.yxxx;
}
