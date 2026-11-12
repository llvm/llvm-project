// Test that the source manager has the "proper" idea about the include stack
// when using PCH.

// RUN: echo 'int x;' > %t.prefix.h
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-show-note-include-stack -include %t.prefix.h %s 2> %t.diags.no_pch.txt
// RUN: %clang_cc1 -emit-pch -o %t.prefix.pch %t.prefix.h
// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-show-note-include-stack -include-pch %t.prefix.pch %s 2> %t.diags.pch.txt
// RUN: diff %t.diags.no_pch.txt %t.diags.pch.txt
// XFAIL: *
// PR5662

float x;
