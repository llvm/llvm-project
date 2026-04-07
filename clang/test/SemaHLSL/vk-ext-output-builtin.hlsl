// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x hlsl -hlsl-entry foo  -finclude-default-header -o - %s -verify

// expected-error@+1 {{'vk::ext_builtin_output' attribute only applies to non-const static globals}}
[[vk::ext_builtin_output(/* Position */ 0)]]
float4 position0;

// expected-error@+1 {{'vk::ext_builtin_output' attribute only applies to non-const static globals}}
[[vk::ext_builtin_output(/* Position */ 0)]]
// expected-error@+1 {{default initialization of an object of const type 'const hlsl_private float4' (aka 'const hlsl_private vector<float, 4>')}}
static const float4 position1;

// expected-error@+1 {{'vk::ext_builtin_output' attribute takes one argument}}
[[vk::ext_builtin_output()]]
static float4 position2;

// expected-error@+1 {{'vk::ext_builtin_output' attribute requires an integer constant}}
[[vk::ext_builtin_output(0.4f)]]
static float4 position3;

// expected-error@+1 {{'vk::ext_builtin_output' attribute only applies to non-const static globals}}
[[vk::ext_builtin_output(0)]]
void some_function() {
}

[numthreads(1,1,1)]
void foo() {
}
