// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -x hlsl -ast-dump -o - %s | FileCheck %s

typedef vector<float, 4> float4;

// CHECK: -TypeAliasDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 2]]:1, col:83>
// CHECK: -HLSLAttributedResourceType 0x{{[0-9a-f]+}} '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]] {{\[\[}}hlsl::contained_type(int)]]' sugar
using ResourceIntAliasT = __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(int)]];
ResourceIntAliasT h1;

// CHECK: -VarDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 1]]:1, col:82> col:82 h2 '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]] {{\[\[}}hlsl::contained_type(float4)]]':'__hlsl_resource_t'
__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(float4)]] h2;

// ClassTemplateDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:1, line:16:1> line:14:30 S
// TemplateTypeParmDecl 0x{{[0-9a-f]+}} <col:[[# @LINE + 3]], col:20> col:20 referenced typename depth 0 index 0 T
// CXXRecordDecl 0x{{[0-9a-f]+}} <col:23, line:[[# @LINE + 2]]:1> line:14:30 struct S definition
// FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 1]]:3, col:79> col:79 h '__hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(T)]]':'__hlsl_resource_t'
template <typename T> struct S {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::contained_type(T)]] h;
};
