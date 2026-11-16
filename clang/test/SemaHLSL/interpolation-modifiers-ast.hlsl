// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -Wno-duplicate-decl-specifier -Wno-ignored-attributes -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -Wno-duplicate-decl-specifier -Wno-ignored-attributes -emit-pch -o %t %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -include-pch %t -ast-dump-all -x hlsl /dev/null | FileCheck %s

void modifiers(nointerpolation float a, linear float b, centroid float c,
               noperspective float d, sample float e, center float f);
// CHECK-LABEL: FunctionDecl {{.*}} modifiers
// CHECK: ParmVarDecl {{.*}} a 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 1
// CHECK: ParmVarDecl {{.*}} b 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 2
// CHECK: ParmVarDecl {{.*}} c 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 16
// CHECK: ParmVarDecl {{.*}} d 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 4
// CHECK: ParmVarDecl {{.*}} e 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 32
// CHECK: ParmVarDecl {{.*}} f 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 8

void merged(linear sample noperspective float a,
            noperspective sample linear float b,
            center centroid sample linear float c,
            linear linear float d);
// CHECK-LABEL: FunctionDecl {{.*}} merged
// CHECK: ParmVarDecl {{.*}} a 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 38
// CHECK-NOT: HLSLInterpolationModifierAttr
// CHECK: ParmVarDecl {{.*}} b 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 38
// CHECK-NOT: HLSLInterpolationModifierAttr
// CHECK: ParmVarDecl {{.*}} c 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 58
// CHECK-NOT: HLSLInterpolationModifierAttr
// CHECK: ParmVarDecl {{.*}} d 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 2
// CHECK-NOT: HLSLInterpolationModifierAttr

struct Fields {
  centroid float x;
  float noperspective y;
};
// CHECK-LABEL: CXXRecordDecl {{.*}} struct Fields definition
// CHECK: FieldDecl {{.*}} x 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 16
// CHECK: FieldDecl {{.*}} y 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 4

linear float result();
// CHECK-LABEL: FunctionDecl {{.*}} result
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 2

// No entry point is needed to preserve modifiers on a dependent declaration.
template<typename T> void dependent(sample T x) {}
void instantiate() { dependent(1.0f); }
// CHECK-LABEL: FunctionTemplateDecl {{.*}} dependent
// CHECK: ParmVarDecl {{.*}} x 'T'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 32
// CHECK: FunctionDecl {{.*}} dependent 'void (float)' implicit_instantiation
// CHECK: ParmVarDecl {{.*}} x 'float'
// CHECK-NEXT: HLSLInterpolationModifierAttr {{.*}} 32
