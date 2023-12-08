
// Test this without pch.
// RUN: %clang_cc1 -ffixed-point -include %S/Inputs/fixed-point-literal.h -fsyntax-only -ast-print -o - %s | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -ffixed-point -emit-pch -o %t %S/Inputs/fixed-point-literal.h
// RUN: %clang_cc1 -ffixed-point -include-pch %t -fsyntax-only -ast-print -o - %s | FileCheck %s

// CHECK: const short _Fract sf = -0.25r;
// CHECK: const _Fract f = 0.75r;
// CHECK: const long _Accum la = 25.25lk;

short _Fract sf2 = sf;
_Fract f2 = f;
long _Accum la2 = la;
