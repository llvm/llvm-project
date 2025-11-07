// This test verifies that _MSVC_TRADITIONAL is defined under the right
// circumstances, microsoft-ext.c is responsible for testing the implementation
// details of the traditional preprocessor.
// FIXME: C11 should eventually use the conforming preprocessor by default.
//
// RUN: %clang_cc1 -E %s | FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -E -fms-extensions %s | FileCheck --check-prefix=CHECK-MS-EXT %s
// RUN: %clang -E -fms-compatibility %s | FileCheck --check-prefix=CHECK-MS-COMPAT %s
// RUN: %clang_cc1 -E -fms-preprocessor-compat %s | FileCheck --check-prefix=CHECK-MS-PREPRO-COMPAT %s
// RUN: %clang_cl -E %s | FileCheck --check-prefix=CHECK-CL %s
// RUN: %clang_cl -E /std:c11 %s | FileCheck --check-prefix=CHECK-C11 %s
// RUN: %clang_cl -E /Zc:preprocessor %s | FileCheck --check-prefix=CHECK-ZC-PREPRO %s
// RUN: %clang_cl -E /clang:-fno-ms-preprocessor-compat %s | FileCheck --check-prefix=CHECK-NO-PREPRO-COMPAT %s

typedef enum {
	NOT_DEFINED,
	IS_ZERO,
	IS_ONE
} State;

#if !defined(_MSVC_TRADITIONAL)
State state = NOT_DEFINED;
#elif _MSVC_TRADITIONAL == 0
State state = IS_ZERO;
#elif _MSVC_TRADITIONAL == 1
State state = IS_ONE;
#endif

// CHECK-DEFAULT: State state = NOT_DEFINED;
// CHECK-MS-EXT: State state = IS_ZERO;
// CHECK-MS-COMPAT: State state = IS_ONE;
// CHECK-MS-PREPRO-COMPAT: State state = IS_ONE;
// CHECK-CL: State state = IS_ONE;
// CHECK-C11: State state = IS_ONE;
// CHECK-ZC-PREPRO: State state = IS_ZERO;
// CHECK-NO-PREPRO-COMPAT: State state = IS_ZERO;
