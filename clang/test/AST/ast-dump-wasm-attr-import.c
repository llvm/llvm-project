// Test without serialization:
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -ast-dump %s \
// RUN: | FileCheck --strict-whitespace %s

// Test with serialization:
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -triple wasm32-unknown-unknown -x c -include-pch %t -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s
//
// Test that functions can be redeclared and they retain their attributes.

__attribute__((import_name("import_red"), import_module("mod"))) void red(void);
__attribute__((import_name("import_orange"), import_module("mod"))) void orange(void);
__attribute__((import_name("import_yellow"), import_module("mod"))) void yellow(void);

void red(void);
void orange(void);
void yellow(void);

void calls(void) {
    red();
    orange();
    yellow();
}

// CHECK: |-FunctionDecl {{.+}} used red 'void (void)'
// CHECK: | |-attrDetails: WebAssemblyImportNameAttr {{.+}} "import_red"
// CHECK: | `-attrDetails: WebAssemblyImportModuleAttr {{.+}} "mod"
// CHECK: |-FunctionDecl {{.+}} used orange 'void (void)'
// CHECK: | |-attrDetails: WebAssemblyImportNameAttr {{.+}} "import_orange"
// CHECK: | `-attrDetails: WebAssemblyImportModuleAttr {{.+}} "mod"
// CHECK: |-FunctionDecl {{.+}} used yellow 'void (void)'
// CHECK: | |-attrDetails: WebAssemblyImportNameAttr {{.+}} "import_yellow"
// CHECK: | `-attrDetails: WebAssemblyImportModuleAttr {{.+}} "mod"
// CHECK: |-FunctionDecl {{.+}} used red 'void (void)'
// CHECK: | |-attrDetails: WebAssemblyImportNameAttr {{.+}} Inherited "import_red"
// CHECK: | `-attrDetails: WebAssemblyImportModuleAttr {{.+}} Inherited "mod"
// CHECK: |-FunctionDecl {{.+}} used orange 'void (void)'
// CHECK: | |-attrDetails: WebAssemblyImportNameAttr {{.+}} Inherited "import_orange"
// CHECK: | `-attrDetails: WebAssemblyImportModuleAttr {{.+}} Inherited "mod"
// CHECK: |-FunctionDecl {{.+}} used yellow 'void (void)'
// CHECK: | |-attrDetails: WebAssemblyImportNameAttr {{.+}} Inherited "import_yellow"
// CHECK: | `-attrDetails: WebAssemblyImportModuleAttr {{.+}} Inherited "mod"
