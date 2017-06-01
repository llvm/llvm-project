#import "import_moduleA.h"
static const int FROM_IMPL = 2;

void test0(void) {
  int x = 
}
// The lines above this point are sensitive to line/column changes.

// ===--- None
// RUN: c-index-test -code-completion-at=%s:5:11 %s -I %S/Inputs | FileCheck %s

// ===--- Modules
// RUN: rm -rf %t && mkdir %t
// RUN: c-index-test -code-completion-at=%s:5:11 %s -I %S/Inputs -fmodules -fmodules-cache-path=%t/mcp | FileCheck %s

// ===--- PCH
// RUN: rm -rf %t && mkdir %t
// RUN: c-index-test -write-pch %t/import_moduleA.pch -x objective-c-header %S/Inputs/import_moduleA.h -I %S/Inputs
// RUN: c-index-test -code-completion-at=%s:5:11 %s -include-pch %t/import_moduleA.pch -I %S/Inputs | FileCheck %s

// ===--- PCH + Modules
// RUN: rm -rf %t && mkdir %t
// RUN: c-index-test -write-pch %t/import_moduleA.pch -x objective-c-header %S/Inputs/import_moduleA.h -fmodules -fmodules-cache-path=%t/mcp -I %S/Inputs
// RUN: c-index-test -code-completion-at=%s:5:11 %s -include-pch %t/import_moduleA.pch -I %S/Inputs -fmodules -fmodules-cache-path=%t/mcp | FileCheck %s

// ===--- Preamble
// RUN: rm -rf %t && mkdir %t
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:5:11 %s -I %S/Inputs | FileCheck %s

// ===--- Preamble + Modules
// RUN: rm -rf %t
// RUN: env CINDEXTEST_EDITING=1 c-index-test -code-completion-at=%s:5:11 %s -I %S/Inputs -fmodules -fmodules-cache-path=%t/mcp | FileCheck %s


// CHECK: FROM_HEADER
// CHECK: FROM_IMPL
// CHECK: FROM_MODULE_A
