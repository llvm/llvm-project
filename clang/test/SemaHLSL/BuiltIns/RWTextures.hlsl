// RUN: %clang_cc1 -Wno-hlsl-implicit-binding -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify -DTEXTURE=RWTexture2D %s
// RUN: %clang_cc1 -Wno-hlsl-implicit-binding -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify -DTEXTURE=RWTexture2DArray %s

typedef vector<float, 3> float3;
typedef vector<double, 2> double2;
typedef vector<double, 3> double3;

// expected-error-re@+2 {{class template 'RWTexture{{.*}}' requires template arguments}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWTexture}}
TEXTURE TextureErr1;

// expected-error-re@+2 {{too few template arguments for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWTexture}}
TEXTURE<> TextureErr2;

// test implicit Texture concept
TEXTURE<float3> Tex;
TEXTURE<int> r1;
TEXTURE<float> r2;
TEXTURE<double2> r4;

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note-re@*:* {{because 'RWTexture{{.*}}<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note-re@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(hlsl::RWTexture{{.*}}<int>)' evaluated to false}}
TEXTURE<TEXTURE<int> > r5;

struct s {
    int x;
};

struct Empty {};

template<typename T> struct TemplatedTexture {
    T a;
};

template<typename T> struct TemplatedVector {
    vector<T, 4> v;
};

// structs not allowed
// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 's' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(s)' evaluated to false}}
TEXTURE<s> r6;
// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'Empty' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(Empty)' evaluated to false}}
TEXTURE<Empty> r7;

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'TemplatedTexture<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(TemplatedTexture<int>)' evaluated to false}}
TEXTURE<TemplatedTexture<int> > r8;
// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'TemplatedVector<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(TemplatedVector<int>)' evaluated to false}}
TEXTURE<TemplatedVector<int> > r9;

// arrays not allowed
// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'half[4]' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(half[4])' evaluated to false}}
TEXTURE<half[4]> r10;

typedef vector<int, 8> int8;
// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'int8' (aka 'vector<int, 8>') does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<int, 8>)' evaluated to false}}37
TEXTURE<int8> r11;

typedef int MyInt;
TEXTURE<MyInt> r12;

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'bool' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(bool)' evaluated to false}}
TEXTURE<bool> r13;

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'vector<bool, 2>' (vector of 2 'bool' values) does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<bool, 2>)' evaluated to false}}
TEXTURE<vector<bool, 2>> r14;

enum numbers { one, two, three };

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'numbers' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(numbers)' evaluated to false}}
TEXTURE<numbers> r15;

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'double3' (aka 'vector<double, 3>') does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<double, 3>)' evaluated to false}}
TEXTURE<double3> r16;


struct threeDoubles {
  double a;
  double b;
  double c;
};

// expected-error-re@+3 {{constraints not satisfied for class template 'RWTexture{{.*}}'}}
// expected-note@*:* {{because 'threeDoubles' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(threeDoubles)' evaluated to false}}
TEXTURE<threeDoubles> TextureErr3;


[numthreads(1,1,1)]
void main() {
  (void)Tex.__handle; // expected-error-re {{'__handle' is a private member of 'hlsl::RWTexture{{.*}}<vector<float, 3>>'}}
  // expected-note@* {{implicitly declared private here}}
}

// expected-error-re@+2 {{class template 'RWTexture{{.*}}' requires template arguments}}
// expected-note-re@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWTexture{{.*}} {}}}
void f1(TEXTURE B) {}

// expected-error-re@+2 {{class template 'RWTexture{{.*}}' requires template arguments}}
// expected-note-re@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWTexture{{.*}} {}}}
TEXTURE f2();

struct S {
  // expected-error-re@+2 {{class template 'RWTexture{{.*}}' requires template arguments}}
  // expected-note-re@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWTexture{{.*}} {}}}
  TEXTURE B;
};
