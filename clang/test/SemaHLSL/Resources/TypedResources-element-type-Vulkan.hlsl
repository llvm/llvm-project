// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -fnative-half-type -fnative-int16-type \
// RUN:   -verify=vk,vk16 %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -verify=vk %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-compute -x hlsl -fsyntax-only \
// RUN:   -finclude-default-header -fnative-half-type -fnative-int16-type \
// RUN:   -verify=dxil %s

// dxil-no-diagnostics

// Textures and typed buffers require a 32-bit float or a 32-bit or 64-bit
// integer element type on Vulkan. DXIL accepts the other element types. Without
// native 16-bit types, `half` is a 32-bit type.

// Rejected element types.

Texture2D<half> TexHalf;
// vk16-error@-1 {{resource type 'Texture2D<half>' with element type 'half' is not supported when targeting Vulkan; 'half' is 16 bits wide, but Vulkan requires a 32-bit type or a 64-bit integer type}}
Texture2DArray<half4> TexArrayHalf4;
// vk16-error-re@-1 {{resource type 'Texture2DArray<half4>' {{.*}}with element type 'half' is not supported}}
TextureCube<double> TexCubeDouble;
// vk-error@-1 {{resource type 'TextureCube<double>' with element type 'double' is not supported when targeting Vulkan; 'double' is 64 bits wide, but Vulkan requires a 32-bit type or a 64-bit integer type}}
RWTexture2D<double2> RWTexDouble2;
// vk-error-re@-1 {{resource type 'RWTexture2D<double2>' {{.*}}with element type 'double' is not supported}}
Buffer<double> BufDouble;
// vk-error@-1 {{resource type 'Buffer<double>' with element type 'double' is not supported}}

#ifdef __HLSL_ENABLE_16_BIT
Texture2DMS<float16_t2> TexMSFloat16x2;
// vk16-error-re@-1 {{resource type 'Texture2DMS<float16_t2>' {{.*}}with element type 'half' is not supported}}
RWBuffer<int16_t> RWBufInt16;
// vk16-error-re@-1 {{resource type 'RWBuffer<int16_t>' {{.*}}with element type 'short' is not supported}}
Buffer<uint16_t4> BufUInt16x4;
// vk16-error-re@-1 {{resource type 'Buffer<uint16_t4>' {{.*}}with element type 'unsigned short' is not supported}}
RasterizerOrderedBuffer<vector<int16_t, 3> > ROBufInt16x3;
// vk16-error@-1 {{resource type 'RasterizerOrderedBuffer<vector<int16_t, 3>>' with element type 'short' is not supported}}
#endif

// Legal element types.
Texture2D<float> TexFloat;
Texture2D<int> TexInt;
RWTexture2D<uint> RWTexUInt;
Buffer<int64_t> BufInt64;
RWBuffer<uint64_t2> RWBufUInt64x2;
// HLSL bool has a 32-bit integer memory representation.
RasterizerOrderedBuffer<bool4> ROBufBool4;

// Structured and byte-address buffers are not images.
StructuredBuffer<half4> SBufHalf4;
StructuredBuffer<double> SBufDouble;
ByteAddressBuffer BAB;

// Declaration forms.

Buffer<double> UsedBufDouble;
// vk-error@-1 {{resource type 'Buffer<double>' with element type 'double' is not supported}}
static RWBuffer<half> StaticBufHalf;
// vk16-error@-1 {{resource type 'RWBuffer<half>' with element type 'half' is not supported}}
typedef Texture2D<double2> DoubleTexture;
DoubleTexture TypedefTex;
// vk-error@-1 {{resource type 'DoubleTexture' (aka 'Texture2D<vector<double, 2>>') with element type 'double' is not supported}}
Texture2D<double> TexArray[4];
// vk-error@-1 {{resource type 'Texture2D<double>' with element type 'double' is not supported}}
RWBuffer<double> BufArray2D[2][3];
// vk-error@-1 {{resource type 'RWBuffer<double>' with element type 'double' is not supported}}

// Embedded resources are diagnosed at the field, once for each global
// declaration that contains them: GlobalStruct, GlobalStructArray, GlobalOuter
// and GlobalDerived.
struct S {
  Texture2D<double> Tex;
  // vk-error@-1 4 {{resource type 'Texture2D<double>' with element type 'double' is not supported}}
  Texture2D<float> Legal;
};
S GlobalStruct;
S GlobalStructArray[2];

struct Outer {
  RWBuffer<double> Bufs[2];
  // vk-error@-1 {{resource type 'RWBuffer<double>' with element type 'double' is not supported}}
  S Inner;
};
Outer GlobalOuter;

struct Derived : S {
  RWBuffer<half> Extra;
  // vk16-error@-1 {{resource type 'RWBuffer<half>' with element type 'half' is not supported}}
};
Derived GlobalDerived;

template<typename T> struct ResourceMember {
  Texture2D<T> Tex;
  // vk-error@-1 {{resource type 'Texture2D<double>' with element type 'double' is not supported}}
};
ResourceMember<double> InstantiatedMember;

// Static resources in a template are diagnosed when it is instantiated.
template<typename T> void staticInTemplate() {
  static Buffer<T> B;
  // vk-error@-1 {{resource type 'Buffer<double>' with element type 'double' is not supported}}
}

void resourceParameter(Buffer<double> P);
// vk-error@-1 {{resource type 'Buffer<double>' with element type 'double' is not supported}}

// A template parameter is checked while the template arguments are deduced,
// so the error makes the candidate non-viable.
template<typename T>
void parameterInTemplate(Buffer<T> P) {}
// vk-note@-1 {{candidate template ignored: substitution failure [with T = double]: resource type 'Buffer<double>' with element type 'double' is not supported}}

[numthreads(1, 1, 1)]
void main() {
  double D = UsedBufDouble[0];
  staticInTemplate<double>();
  // vk-note@-1 {{in instantiation of function template specialization 'staticInTemplate<double>' requested here}}
  parameterInTemplate(UsedBufDouble);
  // vk-error@-1 {{no matching function for call to 'parameterInTemplate'}}

  RWBuffer<double> FromHeap = ResourceDescriptorHeap[0];
  // vk-error@-1 {{resource type 'RWBuffer<double>' with element type 'double' is not supported}}
  // A local copy of a diagnosed global is diagnosed again.
  Buffer<double> Copy = UsedBufDouble;
  // vk-error@-1 {{resource type 'Buffer<double>' with element type 'double' is not supported}}
}
