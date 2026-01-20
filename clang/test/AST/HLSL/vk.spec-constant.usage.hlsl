// RUN: %clang_cc1 -finclude-default-header -fnative-int16-type -triple spirv-unknown-vulkan-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} bool_const 'const hlsl_private bool' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'bool'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'bool (*)(unsigned int, bool) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'bool (unsigned int, bool) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_bool' 'bool (unsigned int, bool) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: CXXBoolLiteralExpr {{.*}} 'bool' true
[[vk::constant_id(1)]]
const bool bool_const = true;

// CHECK: VarDecl {{.*}} short_const 'const hlsl_private short' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'short'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'short (*)(unsigned int, short) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'short (unsigned int, short) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_short' 'short (unsigned int, short) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'short' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
[[vk::constant_id(2)]]
const short short_const = 4;

// CHECK: VarDecl {{.*}} int_const 'const hlsl_private int' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(unsigned int, int) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int (unsigned int, int) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_int' 'int (unsigned int, int) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
[[vk::constant_id(3)]]
const int int_const = 5;

// CHECK: VarDecl {{.*}} long_const 'const hlsl_private long long' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'long long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'long long (*)(unsigned int, long long) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'long long (unsigned int, long long) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_longlong' 'long long (unsigned int, long long) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 4
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'long long' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
[[vk::constant_id(4)]]
const long long long_const = 8;

// CHECK: VarDecl {{.*}} ushort_const 'const hlsl_private unsigned short' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'unsigned short'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned short (*)(unsigned int, unsigned short) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned short (unsigned int, unsigned short) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_ushort' 'unsigned short (unsigned int, unsigned short) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned short' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 10
[[vk::constant_id(5)]]
const unsigned short ushort_const = 10;

// CHECK: VarDecl {{.*}} uint_const 'const hlsl_private unsigned int' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int (*)(unsigned int, unsigned int) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int (unsigned int, unsigned int) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_uint' 'unsigned int (unsigned int, unsigned int) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
[[vk::constant_id(6)]]
const unsigned int uint_const = 12;

// CHECK: VarDecl {{.*}} uint_const_2 'const hlsl_private uint':'const hlsl_private unsigned int' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'unsigned int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int (*)(unsigned int, unsigned int) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned int (unsigned int, unsigned int) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_uint' 'unsigned int (unsigned int, unsigned int) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 6
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 12
[[vk::constant_id(6)]]
const uint uint_const_2 = 12;


// CHECK: VarDecl {{.*}} ulong_const 'const hlsl_private unsigned long long' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'unsigned long long'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long long (*)(unsigned int, unsigned long long) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'unsigned long long (unsigned int, unsigned long long) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_ulonglong' 'unsigned long long (unsigned int, unsigned long long) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 7
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned long long' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 25
[[vk::constant_id(7)]]
const unsigned long long ulong_const = 25;

// CHECK: VarDecl {{.*}} half_const 'const hlsl_private half' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'half'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half (*)(unsigned int, half) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'half (unsigned int, half) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_half' 'half (unsigned int, half) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'half' <FloatingCast>
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 4.040000e+01
[[vk::constant_id(8)]]
const half half_const = 40.4;

// CHECK: VarDecl {{.*}} float_const 'const hlsl_private float' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'float'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float (*)(unsigned int, float) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float (unsigned int, float) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_float' 'float (unsigned int, float) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 8
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 50
[[vk::constant_id(8)]]
const float float_const = 50;

// CHECK: VarDecl {{.*}} double_const 'const hlsl_private double' static cinit
// CHECK-NEXT: CallExpr {{.*}} 'double'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double (*)(unsigned int, double) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'double (unsigned int, double) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_double' 'double (unsigned int, double) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 9
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'double' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 100
[[vk::constant_id(9)]]
const double double_const = 100;

// CHECK: VarDecl {{.*}} enum_const 'const hlsl_private E' static cinit
// CHECK-NEXT: CStyleCastExpr {{.*}} 'E' <IntegralCast>
// CHECK-NEXT: CallExpr {{.*}} 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int (*)(unsigned int, int) noexcept' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int (unsigned int, int) noexcept' lvalue Function {{.*}} '__builtin_get_spirv_spec_constant_int' 'int (unsigned int, int) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'unsigned int' <IntegralCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 10 
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <IntegralCast>
// CHECK-NEXT: DeclRefExpr {{.*}} 'E' EnumConstant {{.*}} 'e2' 'E' 
enum E {
    e0 = 10,
    e1 = 20,
    e2 = 30
};

[[vk::constant_id(10)]]
const E enum_const = e2;

// CHECK-NOT: CXXRecordDecl {{.*}} implicit struct __cblayout_$Globals definition
