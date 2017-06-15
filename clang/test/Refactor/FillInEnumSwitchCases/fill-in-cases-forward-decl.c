#ifndef AFTER
enum ForwardEnumDecl;
#endif

enum ForwardEnumDecl {
  A, B
};

#ifdef AFTER
enum ForwardEnumDecl;
#endif

void dontInitiateOnIncompleteEnum(enum ForwardEnumDecl e) {
  switch (e) {
  }
// CHECK: "case A:\n<#code#>\nbreak;\ncase B:\n<#code#>\nbreak;\n" [[@LINE-1]]:3

  switch (e) {
  case A:
    break;
  }
// CHECK: "case B:\n<#code#>\nbreak;\n" [[@LINE-1]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:14:3 -at=%s:18:3 %s | FileCheck %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:14:3 -at=%s:18:3 %s -D AFTER | FileCheck %s
