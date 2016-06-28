// RUN: rm -rf %t
// RUN: mkdir -p %t/fixes
// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=256 -new-name=Hector -export-fixes=%t/fixes/clang-rename.yaml %t.cpp --
// RUN: clang-apply-replacements %t
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
class Cla  // CHECK: class Hector
{
};

// Use grep -FUbo 'Cla' <file> to get the correct offset of Cla when changing
// this file.
