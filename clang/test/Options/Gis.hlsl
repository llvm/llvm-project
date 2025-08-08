// RUN: %clang_dxc -T lib_6_4 -Gis %s 2>&1 -### | FileCheck -check-prefix=Gis %s
// RUN: %clang_dxc -T lib_6_4 %s 2>&1 -### | FileCheck -check-prefix=NO_Gis %s

// Gis: "-ffp-contract=off" "-frounding-math" "-ffp-exception-behavior=strict" "-complex-range=full"
// assert expected floating point options are present
// NO_Gis-NOT: "-ffp-contract=off" 
// NO_Gis-NOT: "-frounding-math" 
// NO_Gis-NOT: "-ffp-exception-behavior=strict" 
// NO_Gis-NOT: "-complex-range=full"
float4 main(float4 a : A) : SV_TARGET
{
  return -a.yxxx;
}
