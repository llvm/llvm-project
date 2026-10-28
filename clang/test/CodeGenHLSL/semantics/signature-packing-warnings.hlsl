// RUN: split-file %s %t && %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=prefix-stable -hlsl-entry vertex -o /dev/null -verify=vertex %t/vertex.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -hlsl-entry pixel -o /dev/null -verify=pixel %t/pixel.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -finclude-default-header -disable-llvm-passes -fdx-semantic-signature-packing-mode=optimized -o /dev/null -verify=library %t/library.hlsl

//--- vertex.hlsl
// Vertex targets warn about their fixed input layout, including empty inputs.
[shader("vertex")]
// vertex-warning@+1 {{semantic signature packing mode 'prefix-stable' is ignored as vertex shader inputs always use stacked packing}}
float4 vertex(float4 position : POSITION) : SV_Position {
  return position;
}

[shader("vertex")]
// vertex-warning@+1 {{semantic signature packing mode 'prefix-stable' is ignored as vertex shader inputs always use stacked packing}}
float4 vertex_output_only() : SV_Position {
  return float4(0, 0, 0, 1);
}

[shader("vertex")]
// vertex-warning@+1 {{semantic signature packing mode 'prefix-stable' is ignored as vertex shader inputs always use stacked packing}}
void vertex_empty() {}

//--- pixel.hlsl
// Pixel targets warn about their fixed output layout, including empty outputs.
[shader("pixel")]
// pixel-warning@+1 {{semantic signature packing mode 'optimized' is ignored as pixel shader outputs always use indexed packing}}
float4 pixel(float4 position : SV_Position) : SV_Target0 {
  return position;
}

[shader("pixel")]
// pixel-warning@+1 {{semantic signature packing mode 'optimized' is ignored as pixel shader outputs always use indexed packing}}
void pixel_input_only(float4 position : SV_Position) {}

[shader("pixel")]
// pixel-warning@+1 {{semantic signature packing mode 'optimized' is ignored as pixel shader outputs always use indexed packing}}
void pixel_empty() {}

//--- library.hlsl
// Library targets do not warn, even for vertex and pixel entry points.
// library-no-diagnostics
#include "vertex.hlsl"
#include "pixel.hlsl"
