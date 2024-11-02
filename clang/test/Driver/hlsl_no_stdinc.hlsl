// RUN: %clang_dxc  -Tlib_6_7 -fcgl -Fo - %s -### 2>&1 | FileCheck %s --check-prefix=STDINC
// RUN: %clang_dxc  -Tlib_6_7 -hlsl-no-stdinc -fcgl -Fo - %s -### 2>&1 | FileCheck %s --check-prefix=NOSTDINC

// RUN: %clang -cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s -verify

// Make sure hlsl-no-stdinc is translated into finclude-default-header.
// STDINC:"-finclude-default-header"
// NOSTDINC-NOT:"-finclude-default-header"

// Make sure uint not work when finclude-default-header is off.
// expected-error@+1 {{unknown type name 'uint'}}
uint a;
