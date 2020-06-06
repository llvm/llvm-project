// RUN: %clang_cc1 -triple wasm32-unknown-unknown -ast-dump %s | FileCheck --strict-whitespace %s

// Test that functions can be redeclared and they retain their attributes.

__attribute__((export_name("export_red"))) void red(void) {}
__attribute__((export_name("export_orange"))) void orange(void) {}
__attribute__((export_name("export_yellow"))) void yellow(void) {}

void red(void);
void orange(void);
void yellow(void);

// CHECK: |-FunctionDecl {{.+}} used red 'void (void)'
// CHECK: | |-CompoundStmt {{.+}}
// CHECK: | |-WebAssemblyExportNameAttr {{.+}} "export_red"
// CHECK: | `-UsedAttr {{.+}} Implicit
// CHECK: |-FunctionDecl {{.+}} used orange 'void (void)'
// CHECK: | |-CompoundStmt {{.+}}
// CHECK: | |-WebAssemblyExportNameAttr {{.+}} "export_orange"
// CHECK: | `-UsedAttr {{.+}} Implicit
// CHECK: |-FunctionDecl {{.+}} used yellow 'void (void)'
// CHECK: | |-CompoundStmt {{.+}}
// CHECK: | |-WebAssemblyExportNameAttr {{.+}} "export_yellow"
// CHECK: | `-UsedAttr {{.+}} Implicit
// CHECK: |-FunctionDecl {{.+}} used red 'void (void)'
// CHECK: | |-UsedAttr {{.+}} Inherited Implicit
// CHECK: | `-WebAssemblyExportNameAttr {{.+}} Inherited "export_red"
// CHECK: |-FunctionDecl {{.+}} used orange 'void (void)'
// CHECK: | |-UsedAttr {{.+}} Inherited Implicit
// CHECK: | `-WebAssemblyExportNameAttr {{.+}} Inherited "export_orange"
// CHECK: `-FunctionDecl {{.+}} used yellow 'void (void)'
// CHECK:   |-UsedAttr {{.+}} Inherited Implicit
// CHECK:     `-WebAssemblyExportNameAttr {{.+}} Inherited "export_yellow"
