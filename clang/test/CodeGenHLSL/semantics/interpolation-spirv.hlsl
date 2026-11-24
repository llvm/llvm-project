// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -Wno-ignored-attributes -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,PS -DAS=7 -DBUILTIN=15
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -DVERTEX -finclude-default-header -Wno-ignored-attributes -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,VS -DAS=8 -DBUILTIN=0

struct Inner {
  float inherited;
  linear float overridden;
};

struct Varyings {
  float a : DEFAULT;
  nointerpolation float b : FLAT;
  linear float c : LINEAR;
  centroid float d : CENTROID;
  noperspective float e : NOPERSPECTIVE;
  linear noperspective centroid float f : NPCENTROID;
  linear sample float g : SAMPLE;
  sample linear noperspective float h : NPSAMPLE;
  center float i : CENTER;
  linear noperspective center float j : NPCENTER;
  center centroid sample float k : PRECEDENCE;
  uint2 integers : INTEGER;
  double wide : DOUBLE;
  sample float4 position : SV_Position;
  nointerpolation Inner nested : NESTED;
  nointerpolation float2 array[2][3] : ARRAY;
};

#ifdef VERTEX
Varyings main(sample float a : ATTRIBUTE,
              nointerpolation uint id : SV_VertexID) {
  Varyings result;
  result.a = a + (float)id;
  return result;
}
#else
sample float4 main(Varyings input) : SV_Target {
  return input.a + input.position;
}
#endif

// The same modifiers describe vertex outputs and fragment inputs. All
// decorations must share one metadata list, preserving locations.
// CHECK-DAG: @DEFAULT0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[DEFAULT:[0-9]+]]
// CHECK-DAG: @FLAT0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[FLAT:[0-9]+]]
// CHECK-DAG: @LINEAR0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[LINEAR:[0-9]+]]
// CHECK-DAG: @CENTROID0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[CENTROID:[0-9]+]]
// CHECK-DAG: @NOPERSPECTIVE0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[NP:[0-9]+]]
// CHECK-DAG: @NPCENTROID0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[NPC:[0-9]+]]
// CHECK-DAG: @SAMPLE0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[SAMPLE:[0-9]+]]
// CHECK-DAG: @NPSAMPLE0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[NPS:[0-9]+]]
// CHECK-DAG: @CENTER0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[CENTER:[0-9]+]]
// CHECK-DAG: @NPCENTER0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[NPCENTER:[0-9]+]]
// CHECK-DAG: @PRECEDENCE0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[PRECEDENCE:[0-9]+]]
// CHECK-DAG: @INTEGER0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[INTEGER:[0-9]+]]
// CHECK-DAG: @DOUBLE0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[DOUBLE:[0-9]+]]
// CHECK-DAG: @SV_Position = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[POSITION:[0-9]+]]
// CHECK-DAG: @NESTED0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[INHERITED:[0-9]+]]
// CHECK-DAG: @NESTED1 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[OVERRIDDEN:[0-9]+]]
// CHECK-DAG: @ARRAY0 = {{.*}}addrspace([[AS]]) {{.*}}, !spirv.Decorations ![[ARRAY:[0-9]+]]

// Vertex inputs and fragment outputs ignore interpolation modifiers, including
// the otherwise mandatory Flat on integer fragment inputs.
// VS-DAG: @ATTRIBUTE0 = {{.*}}addrspace(7) {{.*}}, !spirv.Decorations ![[DEFAULT]]
// VS-DAG: @SV_VertexID = {{.*}}addrspace(7) {{.*}}, !spirv.Decorations ![[VERTEXID:[0-9]+]]
// PS-DAG: @SV_Target0 = {{.*}}addrspace(8) {{.*}}, !spirv.Decorations ![[DEFAULT]]

// CHECK-DAG: ![[DEFAULT]] = !{![[LOC0:[0-9]+]]}
// CHECK-DAG: ![[FLAT]] = !{![[LOC1:[0-9]+]], ![[FLAT_DEC:[0-9]+]]}
// CHECK-DAG: ![[LINEAR]] = !{![[LOC2:[0-9]+]]}
// CHECK-DAG: ![[CENTROID]] = !{![[LOC3:[0-9]+]], ![[CENTROID_DEC:[0-9]+]]}
// CHECK-DAG: ![[NP]] = !{![[LOC4:[0-9]+]], ![[NP_DEC:[0-9]+]]}
// CHECK-DAG: ![[NPC]] = !{![[LOC5:[0-9]+]], ![[NP_DEC]], ![[CENTROID_DEC]]}
// CHECK-DAG: ![[SAMPLE]] = !{![[LOC6:[0-9]+]], ![[SAMPLE_DEC:[0-9]+]]}
// CHECK-DAG: ![[NPS]] = !{![[LOC7:[0-9]+]], ![[NP_DEC]], ![[SAMPLE_DEC]]}
// CHECK-DAG: ![[CENTER]] = !{![[LOC8:[0-9]+]]}
// CHECK-DAG: ![[NPCENTER]] = !{![[LOC9:[0-9]+]], ![[NP_DEC]]}
// CHECK-DAG: ![[PRECEDENCE]] = !{![[LOC10:[0-9]+]], ![[SAMPLE_DEC]]}
// CHECK-DAG: ![[INTEGER]] = !{![[LOC11:[0-9]+]], ![[FLAT_DEC]]}
// CHECK-DAG: ![[DOUBLE]] = !{![[LOC12:[0-9]+]], ![[FLAT_DEC]]}
// CHECK-DAG: ![[POSITION]] = !{![[BUILTIN_DEC:[0-9]+]], ![[SAMPLE_DEC]]}
// CHECK-DAG: ![[INHERITED]] = !{![[LOC13:[0-9]+]], ![[FLAT_DEC]]}
// CHECK-DAG: ![[OVERRIDDEN]] = !{![[LOC14:[0-9]+]]}
// CHECK-DAG: ![[ARRAY]] = !{![[LOC15:[0-9]+]], ![[FLAT_DEC]]}
// VS-DAG: ![[VERTEXID]] = !{![[VERTEXID_DEC:[0-9]+]]}

// CHECK-DAG: ![[LOC0]] = !{i32 30, i32 0}
// CHECK-DAG: ![[LOC1]] = !{i32 30, i32 1}
// CHECK-DAG: ![[LOC2]] = !{i32 30, i32 2}
// CHECK-DAG: ![[LOC3]] = !{i32 30, i32 3}
// CHECK-DAG: ![[LOC4]] = !{i32 30, i32 4}
// CHECK-DAG: ![[LOC5]] = !{i32 30, i32 5}
// CHECK-DAG: ![[LOC6]] = !{i32 30, i32 6}
// CHECK-DAG: ![[LOC7]] = !{i32 30, i32 7}
// CHECK-DAG: ![[LOC8]] = !{i32 30, i32 8}
// CHECK-DAG: ![[LOC9]] = !{i32 30, i32 9}
// CHECK-DAG: ![[LOC10]] = !{i32 30, i32 10}
// CHECK-DAG: ![[LOC11]] = !{i32 30, i32 11}
// CHECK-DAG: ![[LOC12]] = !{i32 30, i32 12}
// CHECK-DAG: ![[LOC13]] = !{i32 30, i32 13}
// CHECK-DAG: ![[LOC14]] = !{i32 30, i32 14}
// CHECK-DAG: ![[LOC15]] = !{i32 30, i32 15}
// CHECK-DAG: ![[FLAT_DEC]] = !{i32 14}
// CHECK-DAG: ![[NP_DEC]] = !{i32 13}
// CHECK-DAG: ![[CENTROID_DEC]] = !{i32 16}
// CHECK-DAG: ![[SAMPLE_DEC]] = !{i32 17}
// CHECK-DAG: ![[BUILTIN_DEC]] = !{i32 11, i32 [[BUILTIN]]}
// VS-DAG: ![[VERTEXID_DEC]] = !{i32 11, i32 42}
