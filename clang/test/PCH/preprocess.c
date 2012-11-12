// Check that -E mode is invariant when using an implicit PCH.

// RUN: %clang_cc1 -include %S/preprocess.h -E -o %t.orig %s
// RUN: %clang_cc1 -emit-pch -o %t %S/preprocess.h
// RUN: %clang_cc1 -include-pch %t -E -o %t.from_pch %s
// RUN: diff %t.orig %t.from_pch

a_typedef a_value;
