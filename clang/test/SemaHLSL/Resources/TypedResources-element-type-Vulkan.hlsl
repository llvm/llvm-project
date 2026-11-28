// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -fnative-half-type -fnative-int16-type \
// RUN:   -verify=common,vk,vk16 %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -verify=common,vk %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -fnative-half-type -fnative-int16-type \
// RUN:   -verify=common,dxil %s

// Textures and typed buffers require a 32-bit float or a 32-bit or 64-bit
// integer element type on Vulkan. DXIL accepts the other element types. Without
// native 16-bit types, `half` is a 32-bit type.

// Rejected element types.

Texture2D<half> TexHalf;
// vk16-error@-1 {{constraints not satisfied for class template 'Texture2D' [with element_type = half]}}
// vk16-note@*:* {{because 'half' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(half)' evaluated to false}}
Texture2DArray<half4> TexArrayHalf4;
// vk16-error@-1 {{constraints not satisfied for class template 'Texture2DArray' [with element_type = half4]}}
// vk16-note@*:* {{because 'half4' (aka 'vector<half, 4>') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(vector<half, 4>)' evaluated to false}}
TextureCube<double> TexCubeDouble;
// vk-error@-1 {{constraints not satisfied for class template 'TextureCube' [with element_type = double]}}
// vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
RWTexture2D<double2> RWTexDouble2;
// vk-error@-1 {{constraints not satisfied for class template 'RWTexture2D' [with element_type = double2]}}
// vk-note@*:* {{because 'double2' (aka 'vector<double, 2>') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(vector<double, 2>)' evaluated to false}}
Buffer<double> BufDouble;
// vk-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = double]}}
// vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}

#ifdef __HLSL_ENABLE_16_BIT
Texture2DMS<float16_t2> TexMSFloat16x2;
// vk16-error@-1 {{constraints not satisfied for class template 'Texture2DMS' [with element_type = float16_t2, sample_count = 0]}}
// vk16-note@*:* {{because 'float16_t2' (aka 'vector<float16_t, 2>') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(vector<half, 2>)' evaluated to false}}
RWBuffer<int16_t> RWBufInt16;
// vk16-error@-1 {{constraints not satisfied for class template 'RWBuffer' [with element_type = int16_t]}}
// vk16-note@*:* {{because 'int16_t' (aka 'short') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(short)' evaluated to false}}
Buffer<uint16_t4> BufUInt16x4;
// vk16-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = uint16_t4]}}
// vk16-note@*:* {{because 'uint16_t4' (aka 'vector<uint16_t, 4>') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(vector<unsigned short, 4>)' evaluated to false}}
#endif

// Legal element types.
Texture2D<float> TexFloat;
Texture2D<int> TexInt;
RWTexture2D<uint> RWTexUInt;
Buffer<int64_t> BufInt64;
RWBuffer<uint64_t2> RWBufUInt64x2;

// Vulkan's constraint still enforces the general typed-resource restrictions.
Buffer<bool> BoolBuffer;
// common-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = bool]}}
// vk-note@*:* {{because 'bool' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(bool)' evaluated to false}}
// dxil-note@*:* {{because 'bool' does not satisfy '__is_typed_resource_element_compatible'}}
// dxil-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(bool)' evaluated to false}}
Buffer<float[4]> ArrayElement;
// common-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = float[4]]}}
// vk-note@*:* {{because 'float[4]' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(float[4])' evaluated to false}}
// dxil-note@*:* {{because 'float[4]' does not satisfy '__is_typed_resource_element_compatible'}}
// dxil-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(float[4])' evaluated to false}}
Buffer<uint64_t3> OversizedElement;
// common-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = uint64_t3]}}
// vk-note@*:* {{because 'uint64_t3' (aka 'vector<uint64_t, 3>') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(vector<unsigned long, 3>)' evaluated to false}}
// dxil-note@*:* {{because 'uint64_t3' (aka 'vector<uint64_t, 3>') does not satisfy '__is_typed_resource_element_compatible'}}
// dxil-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<unsigned long, 3>)' evaluated to false}}
struct Element { float Value; };
Texture2D<Element> StructElement;
// common-error@-1 {{constraints not satisfied for class template 'Texture2D' [with element_type = Element]}}
// vk-note@*:* {{because 'Element' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(Element)' evaluated to false}}
// dxil-note@*:* {{because 'Element' does not satisfy '__is_typed_resource_element_compatible'}}
// dxil-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(Element)' evaluated to false}}

// The Vulkan builtin must be a complete alternative, including when queried
// directly rather than through a resource's generated concept.
static_assert(__builtin_hlsl_is_vulkan_typed_resource_element_compatible(float), "");
static_assert(__builtin_hlsl_is_vulkan_typed_resource_element_compatible(uint64_t2), "");
static_assert(!__builtin_hlsl_is_vulkan_typed_resource_element_compatible(bool), "");
static_assert(!__builtin_hlsl_is_vulkan_typed_resource_element_compatible(Element), "");
static_assert(!__builtin_hlsl_is_vulkan_typed_resource_element_compatible(float[4]), "");
static_assert(!__builtin_hlsl_is_vulkan_typed_resource_element_compatible(uint64_t3), "");
static_assert(!__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double), "");

// Structured and byte-address buffers do not lower to images.
StructuredBuffer<half4> SBufHalf4;
StructuredBuffer<double> SBufDouble;
RWStructuredBuffer<double2> RWSBufDouble2;
ByteAddressBuffer BAB;
RWByteAddressBuffer RWBAB;

// Reject invalid types even without a variable declaration or resource use.
typedef Texture2D<double2> DoubleTexture;
// vk-error@-1 {{constraints not satisfied for class template 'Texture2D' [with element_type = double2]}}
// vk-note@*:* {{because 'double2' (aka 'vector<double, 2>') does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(vector<double, 2>)' evaluated to false}}
Buffer<double> resourceResult();
// vk-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = double]}}
// vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
void resourceParameter(Buffer<double> P);
// vk-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = double]}}
// vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}

static RWBuffer<half> StaticBufHalf;
// vk16-error@-1 {{constraints not satisfied for class template 'RWBuffer' [with element_type = half]}}
// vk16-note@*:* {{because 'half' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(half)' evaluated to false}}
Texture2D<double> TexArray[4];
// vk-error@-1 {{constraints not satisfied for class template 'Texture2D' [with element_type = double]}}
// vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
RWBuffer<double> BufArray2D[2][3];
// vk-error@-1 {{constraints not satisfied for class template 'RWBuffer' [with element_type = double]}}
// vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
// vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}

// A field's resource type is checked even if the struct is never used.
struct Unused {
  Texture2D<double> Tex;
  // vk-error@-1 {{constraints not satisfied for class template 'Texture2D' [with element_type = double]}}
  // vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
  // vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
};

template<typename T> struct ResourceMember {
  Texture2D<T> Tex;
  // vk-error@-1 {{constraints not satisfied for class template 'Texture2D' [with element_type = double]}}
  // vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
  // vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
};
ResourceMember<double> InstantiatedMember;
// vk-note@-1 {{in instantiation of template class 'ResourceMember<double>' requested here}}

template<typename T> void staticInTemplate() {
  static Buffer<T> B;
  // vk-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = double]}}
  // vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
  // vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
}

// Constraint failure during deduction must discard the candidate, not produce
// a hard error. In DXIL, Buffer<double> is valid and the template is selected.
template<typename T> Buffer<T> deduceResource(T Value);
float deduceResource(...);
#ifdef __SPIRV__
static_assert(__is_same(decltype(deduceResource(1.0L)), float), "");
#else
static_assert(__is_same(decltype(deduceResource(1.0L)), Buffer<double>), "");
#endif

[numthreads(1, 1, 1)]
void main() {
  staticInTemplate<double>();
  // vk-note@-1 {{in instantiation of function template specialization 'staticInTemplate<double>' requested here}}
  RWBuffer<double> FromHeap = ResourceDescriptorHeap[0];
  // vk-error@-1 {{constraints not satisfied for class template 'RWBuffer' [with element_type = double]}}
  // vk-note@*:* {{because 'double' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
  // vk-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(double)' evaluated to false}}
  Buffer<half> LocalHalf;
  // vk16-error@-1 {{constraints not satisfied for class template 'Buffer' [with element_type = half]}}
  // vk16-note@*:* {{because 'half' does not satisfy '__is_vulkan_typed_resource_element_compatible'}}
  // vk16-note@*:* {{because '__builtin_hlsl_is_vulkan_typed_resource_element_compatible(half)' evaluated to false}}
}
