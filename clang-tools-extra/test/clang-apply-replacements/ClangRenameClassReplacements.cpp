// RUN: rm -rf %t
// RUN: mkdir -p %t/fixes
// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=254 -new-name=Bar -export-fixes=%t/fixes/clang-rename.yaml %t.cpp --
// RUN: clang-apply-replacements %t
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s

class Foo {}; // CHECK: class Bar {};

// Use grep -FUbo 'Foo' <file> to get the correct offset of Cla when changing
// this file.
