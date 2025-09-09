// RUN: %clang_cc1 -Wno-hlsl-implicit-binding -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef vector<float, 3> float3;
typedef vector<double, 2> double2;
typedef vector<double, 3> double3;


// expected-error@+1 {{class template 'Buffer' requires template arguments}}
Buffer BufferErr1;

// expected-error@+1 {{too few template arguments for class template 'Buffer'}}
Buffer<> BufferErr2;

// test implicit Buffer concept
Buffer<int> r1;
Buffer<float> r2;
Buffer<float3> Buff;
Buffer<double2> r4;

// expected-error@+4 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class Buffer}}
// expected-note@*:* {{because 'Buffer<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(hlsl::Buffer<int>)' evaluated to false}}
Buffer<Buffer<int> > r5;

struct s {
    int x;
};

struct Empty {};

template<typename T> struct TemplatedBuffer {
    T a;
};

template<typename T> struct TemplatedVector {
    vector<T, 4> v;
};

// structs not allowed
// expected-error@+4 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class Buffer}}
// expected-note@*:* {{because 's' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(s)' evaluated to false}}
Buffer<s> r6;
// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'Empty' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(Empty)' evaluated to false}}
Buffer<Empty> r7;

// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'TemplatedBuffer<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(TemplatedBuffer<int>)' evaluated to false}}
Buffer<TemplatedBuffer<int> > r8;
// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'TemplatedVector<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(TemplatedVector<int>)' evaluated to false}}
Buffer<TemplatedVector<int> > r9;

// arrays not allowed
// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'half[4]' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(half[4])' evaluated to false}}
Buffer<half[4]> r10;

typedef vector<int, 8> int8;
// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'int8' (aka 'vector<int, 8>') does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<int, 8>)' evaluated to false}}
Buffer<int8> r11;

typedef int MyInt;
Buffer<MyInt> r12;

// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'bool' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(bool)' evaluated to false}}
Buffer<bool> r13;

// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'vector<bool, 2>' (vector of 2 'bool' values) does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<bool, 2>)' evaluated to false}}
Buffer<vector<bool, 2>> r14;

enum numbers { one, two, three };

// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'numbers' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(numbers)' evaluated to false}}
Buffer<numbers> r15;

// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'double3' (aka 'vector<double, 3>') does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<double, 3>)' evaluated to false}}
Buffer<double3> r16;


struct threeDoubles {
  double a;
  double b;
  double c;
};

// expected-error@+3 {{constraints not satisfied for class template 'Buffer'}}
// expected-note@*:* {{because 'threeDoubles' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(threeDoubles)' evaluated to false}}
Buffer<threeDoubles> BufferErr3;


[numthreads(1,1,1)]
void main() {
  (void)Buff.__handle; // expected-error {{'__handle' is a private member of 'hlsl::Buffer<vector<float, 3>>'}}
  // expected-note@* {{implicitly declared private here}}

  // expected-error@+2 {{cannot assign to return value because function 'operator[]' returns a const value}}
  // expected-note@* {{function 'operator[]' which returns const-qualified type 'vector<float const hlsl_device &, 3>' declared here}}
  Buff[0] = 0.0;
}
