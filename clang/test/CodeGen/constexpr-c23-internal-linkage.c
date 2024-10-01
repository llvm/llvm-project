/*
 * RUN: %clang_cc1 -std=c23 -emit-llvm -o - %s | FileCheck %s
 */

constexpr int var_int = 1;
constexpr char var_char = 'a';
constexpr float var_float = 2.5;

const int *p_i = &var_int;
const char *p_c = &var_char;
const float *p_f = &var_float;

/*
CHECK: @var_int = internal constant i32 1{{.*}}
CHECK: @var_char = internal constant i8 97{{.*}}
CHECK: @var_float = internal constant float 2.5{{.*}}
*/

