// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -finclude-default-header -std=hlsl202x -verify %s

// Valid: row_major and column_major on matrix types.
row_major float3x3 rm_mat;
column_major float4x4 cm_mat;

row_major int2x3 rm_int_mat;
column_major bool2x2 cm_bool_mat;

// Valid: row_major and column_major on matrix arrays
row_major float3x3 rm_mat_arr[2];
column_major float4x4 cm_mat_arr[3];

// Valid: row_major and column_major on matrix 2d arrays
row_major float3x3 rm_mat_arr_2d[2][3];
column_major float4x4 cm_mat_arr_2d[3][2];

// Valid: on struct fields with matrix type.
struct S {
  row_major float2x2 mat1;
  column_major float3x3 mat2;
};

// Valid: typedef of a matrix type.
typedef row_major float4x4 RowMajorFloat4x4;
typedef column_major float4x4 ColMajorFloat4x4;

// Valid: layout order conversion
row_major float4x4 Row2Col1(float4x4 M) {
    return M;
}

float4x4 Row2Col2(column_major float4x4 M) {
    return M;
}

row_major float4x4 Row2Col(column_major float4x4 M) {
    return M;
}

column_major float4x4 Col2Row(row_major float4x4 M) {
    return M;
}

void bar(row_major float4x4 M, column_major float4x4 M2) {}

//Invalid: 
// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
void foo(column_major float4x4 mat, row_major int i) {}
// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
void foo2(float4x4 mat, row_major int i) {}

// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
row_major int foo(float4x4 mat) {}


template <typename T>
struct Wrapper {
  row_major T mat; // expected-error 2 {{'row_major' attribute can only be applied to a matrix type}}
};

Wrapper<float4x4> valid;

// expected-note@+1 {{in instantiation of template class 'Wrapper<float>' requested here}}
Wrapper<float> invalid;

// expected-note@+1 {{in instantiation of template class 'Wrapper<vector<float, 2>>' requested here}}
Wrapper<float2> invalid2;



// Invalid: scalar types.
// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
row_major float f;
// expected-error@+1 {{'column_major' attribute can only be applied to a matrix type}}
column_major int i;

// Invalid: vector types.
// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
row_major float3 v;
// expected-error@+1 {{'column_major' attribute can only be applied to a matrix type}}
column_major int4 iv;

// Invalid: array types.
// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
row_major float arr[10];

// Invalid: struct types.
// expected-error@+1 {{'column_major' attribute can only be applied to a matrix type}}
column_major S s;

// Invalid: struct field with non-matrix type.
struct S2 {
  // expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
  row_major float x;
  // expected-error@+1 {{'column_major' attribute can only be applied to a matrix type}}
  column_major int y;
};

// Invalid: typedef of non-matrix type.
// expected-error@+1 {{'row_major' attribute can only be applied to a matrix type}}
typedef row_major float ScalarRM;

// Invalid: conflicting row_major and column_major on same decl.
// expected-note@+2 {{conflicting attribute is here}}
// expected-error@+1 {{'column_major' and 'row_major' attributes are not compatible}}
row_major column_major float3x3 conflict_mat;

// Invalid: duplicate row_major.
// expected-note@+2 {{previous attribute is here}}
// expected-warning@+1 {{attribute 'row_major' is already applied}}
row_major row_major float3x3 dup_rm_mat;

// Invalid: duplicate column_major.
// expected-note@+2 {{previous attribute is here}}
// expected-warning@+1 {{attribute 'column_major' is already applied}}
column_major column_major float4x4 dup_cm_mat;
