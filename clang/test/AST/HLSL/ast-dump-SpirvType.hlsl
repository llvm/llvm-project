// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: TypedefDecl 0x{{.+}} <{{.+}}:4:1, col:83> col:83 referenced AType 'vk::SpirvOpaqueType<123, RWBuffer<float>, vk::integral_constant<uint, 4>>':'__hlsl_spirv_type<123, 0, 0, RWBuffer<float>, vk::integral_constant<unsigned int, 4>>'
typedef vk::SpirvOpaqueType<123, RWBuffer<float>, vk::integral_constant<uint, 4>> AType;
// CHECK: TypedefDecl 0x{{.+}} <{{.+}}:6:1, col:133> col:133 referenced BType 'vk::SpirvType<12, 2, 4, vk::integral_constant<uint64_t, 4886718345L>, float, vk::Literal<vk::integral_constant<uint, 456>>>':'__hlsl_spirv_type<12, 2, 4, vk::integral_constant<unsigned long, 4886718345>, float, vk::Literal<vk::integral_constant<uint, 456>>>'
typedef vk::SpirvType<12, 2, 4, vk::integral_constant<uint64_t, 0x123456789>, float, vk::Literal<vk::integral_constant<uint, 456>>> BType;

// CHECK: VarDecl 0x{{.+}} <{{.+}}:9:1, col:7> col:7 AValue 'hlsl_constant AType':'hlsl_constant __hlsl_spirv_type<123, 0, 0, RWBuffer<float>, vk::integral_constant<unsigned int, 4>>'
AType AValue;
// CHECK: VarDecl 0x{{.+}} <{{.+}}:11:1, col:7> col:7 BValue 'hlsl_constant BType':'hlsl_constant __hlsl_spirv_type<12, 2, 4, vk::integral_constant<unsigned long, 4886718345>, float, vk::Literal<vk::integral_constant<uint, 456>>>'
BType BValue;

// CHECK: VarDecl 0x{{.+}} <{{.+}}:14:1, col:80> col:80 CValue 'hlsl_constant vk::SpirvOpaqueType<123, vk::Literal<vk::integral_constant<uint, 305419896>>>':'hlsl_constant __hlsl_spirv_type<123, 0, 0, vk::Literal<vk::integral_constant<uint, 305419896>>>'
vk::SpirvOpaqueType<123, vk::Literal<vk::integral_constant<uint, 0x12345678>>> CValue;

// CHECK: TypeAliasDecl 0x{{.+}} <{{.+}}:18:1, col:72> col:7 Array 'vk::SpirvOpaqueType<28, T, vk::integral_constant<uint, L>>':'__hlsl_spirv_type<28U, 0, 0, T, vk::integral_constant<uint, L>>'
template <class T, uint L>
using Array = vk::SpirvOpaqueType<28, T, vk::integral_constant<uint, L>>;

// CHECK: VarDecl 0x{{.+}} <{{.+}}:21:1, col:16> col:16 DValue 'hlsl_constant Array<uint, 5>':'hlsl_constant __hlsl_spirv_type<28, 0, 0, uint, vk::integral_constant<unsigned int, 5>>'
Array<uint, 5> DValue;

[numthreads(1, 1, 1)]
void main() {
// CHECK: VarDecl 0x{{.+}} <col:5, col:11> col:11 EValue 'AType':'__hlsl_spirv_type<123, 0, 0, RWBuffer<float>, vk::integral_constant<unsigned int, 4>>'
    AType EValue;
}
