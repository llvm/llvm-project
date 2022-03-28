// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-amplification | FileCheck -match-full-lines %s --check-prefixes=CHECK,AMPLIFICATION
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-compute | FileCheck -match-full-lines %s --check-prefixes=CHECK,COMPUTE
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-domain | FileCheck -match-full-lines %s --check-prefixes=CHECK,DOMAIN
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-geometry | FileCheck -match-full-lines %s --check-prefixes=CHECK,GEOMETRY
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-hull | FileCheck -match-full-lines %s --check-prefixes=CHECK,HULL
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-library | FileCheck -match-full-lines %s --check-prefixes=CHECK,LIBRARY
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-mesh | FileCheck -match-full-lines %s --check-prefixes=CHECK,MESH
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-pixel | FileCheck -match-full-lines %s --check-prefixes=CHECK,PIXEL
// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -triple dxil-pc-shadermodel6.0-vertex | FileCheck -match-full-lines %s --check-prefixes=CHECK,VERTEX

// CHECK: #define __HLSL_VERSION 2021
// CHECK: #define __SHADER_STAGE_AMPLIFICATION 14
// CHECK: #define __SHADER_STAGE_COMPUTE 5
// CHECK: #define __SHADER_STAGE_DOMAIN 4
// CHECK: #define __SHADER_STAGE_GEOMETRY 2
// CHECK: #define __SHADER_STAGE_HULL 3
// CHECK: #define __SHADER_STAGE_LIBRARY 6
// CHECK: #define __SHADER_STAGE_MESH 13
// CHECK: #define __SHADER_STAGE_PIXEL 0
// CHECK: #define __SHADER_STAGE_VERTEX 1

// AMPLIFICATION: #define __SHADER_TARGET_STAGE 14
// COMPUTE: #define __SHADER_TARGET_STAGE 5
// DOMAIN: #define __SHADER_TARGET_STAGE 4
// GEOMETRY: #define __SHADER_TARGET_STAGE 2
// HULL: #define __SHADER_TARGET_STAGE 3
// LIBRARY: #define __SHADER_TARGET_STAGE 6
// MESH: #define __SHADER_TARGET_STAGE 13
// PIXEL: #define __SHADER_TARGET_STAGE 0
// VERTEX: #define __SHADER_TARGET_STAGE 1

// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -std=hlsl2015 | FileCheck -match-full-lines %s --check-prefixes=STD2015
// STD2015: #define __HLSL_VERSION 2015

// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -std=hlsl2016 | FileCheck -match-full-lines %s --check-prefixes=STD2016
// STD2016: #define __HLSL_VERSION 2016

// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -std=hlsl2017 | FileCheck -match-full-lines %s --check-prefixes=STD2017
// STD2017: #define __HLSL_VERSION 2017

// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -std=hlsl2018 | FileCheck -match-full-lines %s --check-prefixes=STD2018
// STD2018: #define __HLSL_VERSION 2018

// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -std=hlsl2021 | FileCheck -match-full-lines %s --check-prefixes=STD2021
// STD2021: #define __HLSL_VERSION 2021

// RUN: %clang_cc1 %s -E -dM -o - -x hlsl -std=hlsl202x | FileCheck -match-full-lines %s --check-prefixes=STD202x
// STD202x: #define __HLSL_VERSION 2029
