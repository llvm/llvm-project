/* RUN: %clang_cc1 -std=c89 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c99 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c11 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c17 -emit-llvm -o - %s | FileCheck %s
   RUN: %clang_cc1 -std=c2x -emit-llvm -o - %s | FileCheck %s
 */

/* WG14 DR060:
 * Array initialization from a string literal
 */

const char str[5] = "foo";
const __typeof__(*L"a") wstr[5] = L"foo";

// CHECK: @str = {{.*}}constant [5 x i8] c"foo\00\00"
// CHECK-NEXT: @wstr = {{.*}}constant [5 x i{{16|32}}] [i{{16|32}} 102, i{{16|32}} 111, i{{16|32}} 111,  i{{16|32}} 0, i{{16|32}} 0]

