// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - | FileCheck %s --check-prefix=CHECK-ROW-MAJOR
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=row-major -o - -DNAMESPACED| FileCheck %s --check-prefix=CHECK-ROW-MAJOR

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - | FileCheck %s --check-prefix=CHECK-COL-MAJOR
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -fnative-half-type -fnative-int16-type \
// RUN:   -emit-llvm -disable-llvm-passes -fmatrix-memory-layout=column-major -o - -DNAMESPACED| FileCheck %s --check-prefix=CHECK-COL-MAJOR

// CHECK-ROW-MAJOR: @bool1x1_Val = external hidden addrspace(2) global [1 x <1 x i32>], align 4
// CHECK-ROW-MAJOR: @bool1x2_Val = external hidden addrspace(2) global [1 x <2 x i32>], align 4
// CHECK-ROW-MAJOR: @bool1x3_Val = external hidden addrspace(2) global [1 x <3 x i32>], align 4
// CHECK-ROW-MAJOR: @bool1x4_Val = external hidden addrspace(2) global [1 x <4 x i32>], align 4
// CHECK-ROW-MAJOR: @bool2x1_Val = external hidden addrspace(2) global [2 x <1 x i32>], align 4
// CHECK-ROW-MAJOR: @bool2x2_Val = external hidden addrspace(2) global [2 x <2 x i32>], align 4
// CHECK-ROW-MAJOR: @bool2x3_Val = external hidden addrspace(2) global [2 x <3 x i32>], align 4
// CHECK-ROW-MAJOR: @bool2x4_Val = external hidden addrspace(2) global [2 x <4 x i32>], align 4
// CHECK-ROW-MAJOR: @bool3x1_Val = external hidden addrspace(2) global [3 x <1 x i32>], align 4
// CHECK-ROW-MAJOR: @bool3x2_Val = external hidden addrspace(2) global [3 x <2 x i32>], align 4
// CHECK-ROW-MAJOR: @bool3x3_Val = external hidden addrspace(2) global [3 x <3 x i32>], align 4
// CHECK-ROW-MAJOR: @bool3x4_Val = external hidden addrspace(2) global [3 x <4 x i32>], align 4
// CHECK-ROW-MAJOR: @bool4x1_Val = external hidden addrspace(2) global [4 x <1 x i32>], align 4
// CHECK-ROW-MAJOR: @bool4x2_Val = external hidden addrspace(2) global [4 x <2 x i32>], align 4
// CHECK-ROW-MAJOR: @bool4x3_Val = external hidden addrspace(2) global [4 x <3 x i32>], align 4
// CHECK-ROW-MAJOR: @bool4x4_Val = external hidden addrspace(2) global [4 x <4 x i32>], align 4

// CHECK-COL-MAJOR: @bool1x1_Val = external hidden addrspace(2) global [1 x <1 x i32>], align 4
// CHECK-COL-MAJOR: @bool1x2_Val = external hidden addrspace(2) global [2 x <1 x i32>], align 4
// CHECK-COL-MAJOR: @bool1x3_Val = external hidden addrspace(2) global [3 x <1 x i32>], align 4
// CHECK-COL-MAJOR: @bool1x4_Val = external hidden addrspace(2) global [4 x <1 x i32>], align 4
// CHECK-COL-MAJOR: @bool2x1_Val = external hidden addrspace(2) global [1 x <2 x i32>], align 4
// CHECK-COL-MAJOR: @bool2x2_Val = external hidden addrspace(2) global [2 x <2 x i32>], align 4
// CHECK-COL-MAJOR: @bool2x3_Val = external hidden addrspace(2) global [3 x <2 x i32>], align 4
// CHECK-COL-MAJOR: @bool2x4_Val = external hidden addrspace(2) global [4 x <2 x i32>], align 4
// CHECK-COL-MAJOR: @bool3x1_Val = external hidden addrspace(2) global [1 x <3 x i32>], align 4
// CHECK-COL-MAJOR: @bool3x2_Val = external hidden addrspace(2) global [2 x <3 x i32>], align 4
// CHECK-COL-MAJOR: @bool3x3_Val = external hidden addrspace(2) global [3 x <3 x i32>], align 4
// CHECK-COL-MAJOR: @bool3x4_Val = external hidden addrspace(2) global [4 x <3 x i32>], align 4
// CHECK-COL-MAJOR: @bool4x1_Val = external hidden addrspace(2) global [1 x <4 x i32>], align 4
// CHECK-COL-MAJOR: @bool4x2_Val = external hidden addrspace(2) global [2 x <4 x i32>], align 4
// CHECK-COL-MAJOR: @bool4x3_Val = external hidden addrspace(2) global [3 x <4 x i32>], align 4
// CHECK-COL-MAJOR: @bool4x4_Val = external hidden addrspace(2) global [4 x <4 x i32>], align 4

#ifdef NAMESPACED
#define TYPE_DECL(T)  hlsl::T T##_Val
#else
#define TYPE_DECL(T)  T T##_Val
#endif

TYPE_DECL( bool1x1 );
TYPE_DECL( bool1x2 );
TYPE_DECL( bool1x3 );
TYPE_DECL( bool1x4 );
TYPE_DECL( bool2x1 );
TYPE_DECL( bool2x2 );
TYPE_DECL( bool2x3 );
TYPE_DECL( bool2x4 );
TYPE_DECL( bool3x1 );
TYPE_DECL( bool3x2 );
TYPE_DECL( bool3x3 );
TYPE_DECL( bool3x4 );
TYPE_DECL( bool4x1 );
TYPE_DECL( bool4x2 );
TYPE_DECL( bool4x3 );
TYPE_DECL( bool4x4 );
