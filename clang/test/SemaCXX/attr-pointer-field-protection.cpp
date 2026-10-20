// RUN: %clang_cc1 -triple aarch64-unknown-linux -fsyntax-only -verify=disabled  %s
// RUN: %clang_cc1 -fexperimental-allow-pointer-field-protection-attr -triple aarch64-unknown-linux -fsyntax-only -verify=enabled %s

struct [[clang::pointer_field_protection]] S {}; // disabled-error {{this attribute is experimental and must be explicitly enabled with flag -fexperimental-allow-pointer-field-protection-attr}}

// enabled-no-diagnostics
