// Test that the required #pragma directives are minimized
// RUN: %clang_cc1 -print-dependency-directives-minimized-source %s 2>&1 | FileCheck %s

#pragma once

// some pragmas not needed in minimized source.
#pragma region TestRegion
#pragma endregion
#pragma warning "message"

// pragmas required in the minimized source.
#pragma push_macro(    "MYMACRO"   )
#pragma pop_macro("MYMACRO")
#pragma clang module import mymodule
#pragma include_alias(<string>,   "mystring.h")

// CHECK:      #pragma once
// CHECK-NEXT: #pragma push_macro("MYMACRO")
// CHECK-NEXT: #pragma pop_macro("MYMACRO")
// CHECK-NEXT: #pragma clang module import mymodule
// CHECK-NEXT: #pragma include_alias(<string>, "mystring.h")
