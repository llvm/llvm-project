// Test that the required #pragma directives are minimized
// RUN: %clang_cc1 -print-dependency-directives-minimized-source %s > %t 2>&1
// RUN: FileCheck %s -input-file %t
// RUN: %clang_cc1 -Eonly %t

#pragma once

// some pragmas not needed in minimized source.
#pragma region TestRegion
#pragma endregion
#pragma warning "message"

// pragmas required in the minimized source.
#pragma push_macro(    "MYMACRO"   )
#pragma pop_macro("MYMACRO")
#if IMPORT
#pragma clang module import mymodule
#endif
#pragma include_alias(<string>,   "mystring.h")

// CHECK:      #pragma once
// CHECK-NEXT: #pragma push_macro("MYMACRO")
// CHECK-NEXT: #pragma pop_macro("MYMACRO")
// CHECK-NEXT: #if IMPORT
// CHECK-NEXT: #pragma clang module import mymodule
// CHECK-NEXT: #endif
// CHECK-NEXT: #pragma include_alias(<string>, "mystring.h")
