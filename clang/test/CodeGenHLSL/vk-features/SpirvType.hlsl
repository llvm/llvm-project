// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm -disable-llvm-passes \
// RUN:   -fspv-use-unknown-image-format -o - | FileCheck %s

template<class T, uint64_t Size>
using Array = vk::SpirvOpaqueType</* OpTypeArray */ 28, T, vk::integral_constant<uint64_t, Size>>;

template<uint64_t Size>
using ArrayBuffer = Array<RWBuffer<float>, Size>;

typedef vk::SpirvType</* OpTypeInt */ 21, 4, 32, vk::Literal<vk::integral_constant<uint, 32>>, vk::Literal<vk::integral_constant<bool, false>>> Int;

typedef Array<Int, 5> ArrayInt;

// CHECK: %struct.S = type { target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0), target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32) }
struct S {
    ArrayBuffer<4> b;
    Int i;
};

// CHECK: define hidden spir_func target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0) @_Z14getArrayBufferu17spirv_type_28_0_0U5_TypeN4hlsl8RWBufferIfEEU6_ConstLm4E(target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0) %v) #0
ArrayBuffer<4> getArrayBuffer(ArrayBuffer<4> v) {
    return v;
}

// CHECK: define hidden spir_func target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32) @_Z6getIntu18spirv_type_21_4_32U4_LitLi32EU4_LitLi0E(target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32) %v) #0
Int getInt(Int v) {
    return v;
}

// TODO: uncomment and test once CBuffer handles are implemented for SPIR-V
// ArrayBuffer<4> g_buffers;
// Int g_word;

[numthreads(1, 1, 1)]
void main() {
    // CHECK: [[buffers:%.*]] = alloca target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0), align 4
    ArrayBuffer<4> buffers;

    // CHECK: [[longBuffers:%.*]] = alloca target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 591751049, 1), 28, 0, 0), align 4
    ArrayBuffer<0x123456789> longBuffers;

    // CHECK: [[word:%.*]] = alloca target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32), align 4
    Int word;

    // CHECK: [[words:%.*]] = alloca [4 x target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32)], align 4
    Int words[4];

    // CHECK: [[words2:%.*]] = alloca target("spirv.Type", target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32), target("spirv.IntegralConstant", i64, 5), 28, 0, 0), align 4
    ArrayInt words2;

    // CHECK: [[value:%.*]] = alloca %struct.S, align 1
    S value;

    // CHECK: [[buffers2:%.*]] = alloca target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0), align 4
    // CHECK: [[word2:%.*]] = alloca target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32), align 4


    // CHECK: [[loaded:%[0-9]+]] = load target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0), ptr [[buffers]], align 4
    // CHECK: %call1 = call spir_func target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0) @_Z14getArrayBufferu17spirv_type_28_0_0U5_TypeN4hlsl8RWBufferIfEEU6_ConstLm4E(target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0) [[loaded]])
    // CHECK: store target("spirv.Type", target("spirv.Image", float, 5, 2, 0, 0, 2, 0), target("spirv.IntegralConstant", i64, 4), 28, 0, 0) %call1, ptr [[buffers2]], align 4
    ArrayBuffer<4> buffers2 = getArrayBuffer(buffers);

    // CHECK: [[loaded:%[0-9]+]] = load target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32), ptr [[word]], align 4
    // CHECK: %call2 = call spir_func target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32) @_Z6getIntu18spirv_type_21_4_32U4_LitLi32EU4_LitLi0E(target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32) [[loaded]])
    // CHECK: store target("spirv.Type", target("spirv.Literal", 32), target("spirv.Literal", 0), 21, 4, 32) %call2, ptr [[word2]], align 4
    Int word2 = getInt(word);
}
