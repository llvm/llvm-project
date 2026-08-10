// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -I %S/Inputs/preprocess -include %S/Inputs/preprocess-prefix.h -E %s | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -I %S/Inputs/preprocess -x objective-c-header -emit-pch %S/Inputs/preprocess-prefix.h -o %t.pch
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -I %S/Inputs/preprocess -include-pch %t.pch -E %s | FileCheck -strict-whitespace %s
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -I %S/Inputs/preprocess -x objective-c++ -include %S/Inputs/preprocess-prefix.h -E %s | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -I %S/Inputs/preprocess -x objective-c++-header -emit-pch %S/Inputs/preprocess-prefix.h -o %t.pch
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs -I %S/Inputs/preprocess -x objective-c++ -include-pch %t.pch -E %s | FileCheck -strict-whitespace %s
#import "diamond_right.h"
#import "diamond_right.h" // to check that imports get their own line
#include "file.h"
void test() {
  top_left_before();
  left_and_right();
}


// CHECK: int left_and_right(int *);{{$}}
// CHECK-NEXT: #pragma clang module import diamond_left /* clang -E: implicit import for #import "diamond_left.h" */{{$}}

// CHECK: #pragma clang module import diamond_right /* clang -E: implicit import for #import "diamond_right.h" */{{$}}
// CHECK: #pragma clang module import diamond_right /* clang -E: implicit import for #import "diamond_right.h" */{{$}}
// CHECK: #pragma clang module import file /* clang -E: implicit import for #include "file.h" */{{$}}
// CHECK-NEXT: void test() {{{$}}
// CHECK-NEXT:    top_left_before();{{$}}
// CHECK-NEXT:    left_and_right();{{$}}
// CHECK-NEXT: }{{$}}
