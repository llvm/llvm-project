// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/ExportDeclNotInModulePurview.cppm -verify -emit-module-interface -o /dev/null
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -verify -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/AddExport.cppm -verify -fmodule-file=A=%t/A.pcm -o /dev/null
//
// RUN: %clang_cc1 -std=c++20 %t/AddExport2.cppm -emit-module-interface -verify -o /dev/null

//--- ExportDeclNotInModulePurview.cppm
// expected-error@* {{missing 'export module' declaration in module interface unit}}
export int b; // expected-error {{export declaration can only be used within a module interface}}

//--- A.cppm
// expected-no-diagnostics
export module A;
export int a;

//--- AddExport.cppm
module A; // #module-decl
export int b; // expected-error {{export declaration can only be used within a module interface}}
// expected-note@#module-decl {{add 'export' here}}

//--- AddExport2.cppm
module A; // expected-error {{missing 'export' specifier in module declaration while building module interface}}
export int a;
