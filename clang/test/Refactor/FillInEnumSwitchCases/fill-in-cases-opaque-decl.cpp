enum IncompleteEnum : int;

enum IncompleteEnum : int {
  A, B
};

enum class IncompleteClassEnum : short;

enum class IncompleteClassEnum : short {
  B, C
};

enum class IncompleteClassEnum2;

enum class IncompleteClassEnum2 {
  D, E
};

void dontInitiateOnIncompleteEnum(IncompleteEnum e1, IncompleteClassEnum e2, IncompleteClassEnum2 e3) {
  switch (e1) {
  }
// CHECK: "case A:\n<#code#>\nbreak;\ncase B:\n<#code#>\nbreak;\n" [[@LINE-1]]:3

  switch (e2) {
  }
// CHECK: "case IncompleteClassEnum::B:\n<#code#>\nbreak;\ncase IncompleteClassEnum::C:\n<#code#>\nbreak;\n" [[@LINE-1]]:3

  switch (e3) {
  }
// CHECK: "case IncompleteClassEnum2::D:\n<#code#>\nbreak;\ncase IncompleteClassEnum2::E:\n<#code#>\nbreak;\n" [[@LINE-1]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:20:3 -at=%s:24:3 -at=%s:28:3 %s -std=c++11 | FileCheck %s
