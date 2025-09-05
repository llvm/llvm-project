// RUN: %clang_cc1 -triple spirv-unkown-vulkan1.3-compute -x hlsl -hlsl-entry foo  -finclude-default-header -o - %s -verify

// expected-error@+1 {{'vk::ext_builtin_input' attribute only applies to static const globals}}
[[vk::ext_builtin_input(/* WorkgroupId */ 26)]]
const uint3 groupid1;

// expected-error@+1 {{'vk::ext_builtin_input' attribute only applies to static const globals}}
[[vk::ext_builtin_input(/* WorkgroupId */ 26)]]
static uint3 groupid2;

// expected-error@+1 {{'vk::ext_builtin_input' attribute takes one argument}}
[[vk::ext_builtin_input()]]
// expected-error@+1 {{default initialization of an object of const type 'const hlsl_private uint3' (aka 'const hlsl_private vector<uint, 3>')}}
static const uint3 groupid3;

// expected-error@+1 {{'vk::ext_builtin_input' attribute requires an integer constant}}
[[vk::ext_builtin_input(0.4f)]]
// expected-error@+1 {{default initialization of an object of const type 'const hlsl_private uint3' (aka 'const hlsl_private vector<uint, 3>')}}
static const uint3 groupid4;

// expected-error@+1 {{'vk::ext_builtin_input' attribute only applies to static const globals}}
[[vk::ext_builtin_input(1)]]
void some_function() {
}

[numthreads(1,1,1)]
void foo() {
}

