// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple spirv-vulkan-library \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | llvm-cxxfilt | FileCheck %s --check-prefixes=CHECK,SPV

// CHECK: %S = type <{ i32 }>

// CHECK: @hlsl::ResourceDescriptorHeap = internal {{.*}}global %"struct.hlsl::__hlsl_resource_descriptor_heap_struct" zeroinitializer, align 1
// CHECK: @hlsl::SamplerDescriptorHeap = internal {{.*}}global %"struct.hlsl::__hlsl_sampler_descriptor_heap_struct" zeroinitializer, align 1

// CHECK-LABEL: testTypedBuffer
export void testTypedBuffer(unsigned Index) {
// CHECK: [[TMP0:%.*]] = alloca %"class.hlsl::__hlsl_heap_resource_info"

// DXIL: call void @hlsl::__hlsl_resource_descriptor_heap_struct::operator[](unsigned int)
// DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP0]],
// DXIL-SAME: ptr {{.*}} @hlsl::ResourceDescriptorHeap, i32 noundef %{{[0-9]+}})

// SPV: call spir_func void @hlsl::__hlsl_resource_descriptor_heap_struct::operator[](unsigned int)
// SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP0]],
// SPV-SAME: ptr {{.*}} addrspacecast ({{.*}} @hlsl::ResourceDescriptorHeap to ptr), i32 noundef %{{[0-9]+}})

// CHECK: call {{(spir_func )*}}void @hlsl::RWBuffer<int>::RWBuffer(hlsl::__hlsl_heap_resource_info)
// CHECK-SAME: (ptr {{.*}} %Buffer, ptr noundef byval(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP0]])
  RWBuffer<int> Buffer = ResourceDescriptorHeap[Index];
}

struct S {
  int Value;
};

// CHECK-LABEL: testConstantBuffer
export void testConstantBuffer(unsigned Index) {

// CHECK: %CBS = alloca %"class.hlsl::ConstantBuffer"
// CHECK: [[TMP1:%.*]] = alloca %"class.hlsl::ConstantBuffer"
// CHECK: [[TMP2:%.*]] = alloca %"class.hlsl::__hlsl_heap_resource_info"
// CHECK: call {{(spir_func )*}}void @hlsl::ConstantBuffer<S>::ConstantBuffer()(ptr {{.*}} %CBS)

// DXIL: call void @hlsl::__hlsl_resource_descriptor_heap_struct::operator[](unsigned int)
// DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP2]],
// DXIL-SAME: ptr {{.*}} @hlsl::ResourceDescriptorHeap, i32 noundef %{{[0-9]+}})

// SPV: call spir_func void @hlsl::__hlsl_resource_descriptor_heap_struct::operator[](unsigned int)
// SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP2]],
// SPV-SAME: ptr {{.*}} addrspacecast ({{.*}} @hlsl::ResourceDescriptorHeap to ptr), i32 noundef %{{[0-9]+}})

// CHECK: call {{(spir_func )*}}void @hlsl::ConstantBuffer<S>::ConstantBuffer(hlsl::__hlsl_heap_resource_info)
// CHECK-SAME: (ptr {{.*}} [[TMP1]], ptr noundef byval(%"class.hlsl::__hlsl_heap_resource_info") align 4 %{{.*}})

// CHECK: call {{.*}} ptr @hlsl::ConstantBuffer<S>::operator=(hlsl::ConstantBuffer<S> const&)
// CHECK-SAME: (ptr {{.*}} %CBS, ptr {{.*}} [[TMP1]])
  ConstantBuffer<S> CBS;
  CBS = ResourceDescriptorHeap[Index];
}

// CHECK: define {{(spir_func )*}}void @testSampler(unsigned int)(ptr {{.*}} sret(%"class.hlsl::SamplerState") align {{(4|8)}} [[RESULT:%.*]], i32 noundef %Index)
// CHECK: [[TMP3:%.*]] = alloca %"class.hlsl::__hlsl_heap_sampler_info"
export SamplerState testSampler(unsigned Index) {

// DXIL: call void @hlsl::__hlsl_sampler_descriptor_heap_struct::operator[](unsigned int)
// DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_sampler_info") align 4 [[TMP3]],
// DXIL-SAME: ptr {{.*}} @hlsl::SamplerDescriptorHeap, i32 noundef %{{[0-9]+}})

// SPV: call spir_func void @hlsl::__hlsl_sampler_descriptor_heap_struct::operator[](unsigned int)
// SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_sampler_info") align 4 [[TMP3]],
// SPV-SAME: ptr {{.*}} addrspacecast ({{.*}} @hlsl::SamplerDescriptorHeap to ptr), i32 noundef %{{[0-9]+}})

// CHECK: call {{(spir_func )*}}void @hlsl::SamplerState::SamplerState(hlsl::__hlsl_heap_sampler_info)
// CHECK-SAME: (ptr {{.*}} [[RESULT]], ptr noundef byval(%"class.hlsl::__hlsl_heap_sampler_info") align 4 [[TMP3]])
  return SamplerDescriptorHeap[Index];
}

void useAppendBuffer(AppendStructuredBuffer<int> Buffer, int Value) {
  Buffer.Append(1);
}

// CHECK-LABEL: testCounterBuffer
export void testCounterBuffer(unsigned Index) {
// CHECK: [[TMP_BUFFER:%.*]] = alloca %"class.hlsl::AppendStructuredBuffer"
// CHECK: [[TMP4:%.*]] = alloca %"class.hlsl::__hlsl_heap_resource_info"

// DXIL: call void @hlsl::__hlsl_resource_descriptor_heap_struct::operator[](unsigned int)
// DXIL-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP4]],
// DXIL-SAME: ptr {{.*}} @hlsl::ResourceDescriptorHeap, i32 noundef %{{[0-9]+}})

// SPV: call spir_func void @hlsl::__hlsl_resource_descriptor_heap_struct::operator[](unsigned int)
// SPV-SAME: (ptr {{.*}} sret(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP4]],
// SPV-SAME: ptr {{.*}} addrspacecast ({{.*}} @hlsl::ResourceDescriptorHeap to ptr), i32 noundef %{{[0-9]+}})

// CHECK: call {{(spir_func )*}}void @hlsl::AppendStructuredBuffer<int>::AppendStructuredBuffer(hlsl::__hlsl_heap_resource_info)
// CHECK-SAME: (ptr {{.*}} [[TMP_BUFFER]], ptr noundef byval(%"class.hlsl::__hlsl_heap_resource_info") align 4 [[TMP4]])

// CHECK: call {{(spir_func )*}}void @useAppendBuffer(hlsl::AppendStructuredBuffer<int>, int)
// CHECK-SAME: (ptr {{.*}} [[TMP_BUFFER]], i32 noundef 10)
  useAppendBuffer(ResourceDescriptorHeap[Index], 10);
}

// DXIL-DAG: call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefromheap{{.*}}(i32 %{{.*}})
// DXIL-DAG: call target("dx.CBuffer", %S) @llvm.dx.resource.handlefromheap{{.*}}(i32 %{{.*}})
// DXIL-DAG: call target("dx.Sampler", 0) @llvm.dx.resource.handlefromheap{{.*}}(i32 %{{.*}})
// DXIL-DAG: call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefromheap{{.*}}(i32 %{{.*}})
// DXIL-NOT: counterhandlefromheap

// SPV-DAG: call target("spirv.SignedImage", i32, 5, 2, 0, 0, 2, 24) @llvm.spv.resource.handlefromheap{{.*}}(i32 %{{.*}})
// SPV-DAG: call target("spirv.VulkanBuffer", %S, 2, 0) @llvm.spv.resource.handlefromheap{{.*}}(i32 %{{.*}})
// SPV-DAG: call target("spirv.Sampler") @llvm.spv.resource.handlefromheap{{.*}}(i32 %{{.*}})
// SPV-DAG: call target("spirv.VulkanBuffer", [0 x i32], 12, 1) @llvm.spv.resource.handlefromheap{{.*}}(i32 %{{.*}})
// SPV-DAG: call target("spirv.VulkanBuffer", i32, 12, 1) @llvm.spv.resource.counterhandlefromheap{{.*}}(target("spirv.VulkanBuffer", [0 x i32], 12, 1) %{{.*}}, i32 %{{.*}}) [ "convergencectrl"(token %{{.*}}) ]
