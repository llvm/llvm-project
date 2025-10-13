// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -fsyntax-only -verify %s

typedef vector<float, 3> float3;
typedef vector<double, 2> double2;
typedef vector<double, 3> double3;


// expected-error@+1 {{class template 'RWBuffer' requires template arguments}}
RWBuffer BufferErr1;

// expected-error@+1 {{too few template arguments for class template 'RWBuffer'}}
RWBuffer<> BufferErr2;

// test implicit RWBuffer concept
RWBuffer<int> r1;
RWBuffer<float> r2;
RWBuffer<float3> Buff;
RWBuffer<double2> r4;

// expected-error@+4 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWBuffer}}
// expected-note@*:* {{because 'RWBuffer<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(RWBuffer<int>)' evaluated to false}}
RWBuffer<RWBuffer<int> > r5;

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
// expected-error@+4 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{template declaration from hidden source: template <typename element_type> requires __is_typed_resource_element_compatible<element_type> class RWBuffer}}
// expected-note@*:* {{because 's' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(s)' evaluated to false}}
RWBuffer<s> r6;
// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'Empty' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(Empty)' evaluated to false}}
RWBuffer<Empty> r7;

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'TemplatedBuffer<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(TemplatedBuffer<int>)' evaluated to false}}
RWBuffer<TemplatedBuffer<int> > r8;
// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'TemplatedVector<int>' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(TemplatedVector<int>)' evaluated to false}}
RWBuffer<TemplatedVector<int> > r9;

// arrays not allowed
// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'half[4]' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(half[4])' evaluated to false}}
RWBuffer<half[4]> r10;

typedef vector<int, 8> int8;
// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'int8' (aka 'vector<int, 8>') does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(int8)' evaluated to false}}
RWBuffer<int8> r11;

typedef int MyInt;
RWBuffer<MyInt> r12;

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'bool' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(bool)' evaluated to false}}
RWBuffer<bool> r13;

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'vector<bool, 2>' (vector of 2 'bool' values) does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(vector<bool, 2>)' evaluated to false}}
RWBuffer<vector<bool, 2>> r14;

enum numbers { one, two, three };

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'numbers' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(numbers)' evaluated to false}}
RWBuffer<numbers> r15;

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'double3' (aka 'vector<double, 3>') does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(double3)' evaluated to false}}
RWBuffer<double3> r16;


struct threeDoubles {
  double a;
  double b;
  double c;
};

// expected-error@+3 {{constraints not satisfied for class template 'RWBuffer'}}
// expected-note@*:* {{because 'threeDoubles' does not satisfy '__is_typed_resource_element_compatible'}}
// expected-note@*:* {{because '__builtin_hlsl_is_typed_resource_element_compatible(threeDoubles)' evaluated to false}}
RWBuffer<threeDoubles> BufferErr3;


[numthreads(1,1,1)]
void main() {
  (void)Buff.__handle; // expected-error {{'__handle' is a private member of 'hlsl::RWBuffer<vector<float, 3>>'}}
  // expected-note@* {{implicitly declared private here}}
}
