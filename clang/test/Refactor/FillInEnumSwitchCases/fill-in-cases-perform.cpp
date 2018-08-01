#ifdef NESTEDANON
namespace {
#endif
#ifdef NESTED1
namespace foo {
struct Struct {
#endif

#ifdef ENUMCLASS
enum class Color {
#else
enum Color {
#endif
  Black,
  Blue,
  White,
  Gold
};

#ifdef NESTED1
#define PREFIX foo::Struct::
#else
#define PREFIX
#endif

#ifdef ENUMCLASS
#define CASE(x) PREFIX Color::x
#else
#define CASE(x) PREFIX x
#endif

#ifdef NESTED1
}
#ifndef NESTED1NS
}
#endif
#endif
#ifdef NESTEDANON
}
#endif

void perform1(PREFIX Color c) {
  switch (c) {
  case CASE(Black):
    break;
  }
// CHECK1: "case Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
// CHECK2: "case Color::Blue:\n<#code#>\nbreak;\ncase Color::White:\n<#code#>\nbreak;\ncase Color::Gold:\n<#code#>\nbreak;\n" [[@LINE-2]]:3 -> [[@LINE-2]]:3
// CHECK3: "case foo::Struct::Blue:\n<#code#>\nbreak;\ncase foo::Struct::White:\n<#code#>\nbreak;\ncase foo::Struct::Gold:\n<#code#>\nbreak;\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
// CHECK4: "case foo::Struct::Color::Blue:\n<#code#>\nbreak;\ncase foo::Struct::Color::White:\n<#code#>\nbreak;\ncase foo::Struct::Color::Gold:\n<#code#>\nbreak;\n" [[@LINE-4]]:3 -> [[@LINE-4]]:3
// CHECK5: "case Struct::Blue:\n<#code#>\nbreak;\ncase Struct::White:\n<#code#>\nbreak;\ncase Struct::Gold:\n<#code#>\nbreak;\n" [[@LINE-5]]:3 -> [[@LINE-5]]:3
// CHECK6: "case Struct::Color::Blue:\n<#code#>\nbreak;\ncase Struct::Color::White:\n<#code#>\nbreak;\ncase Struct::Color::Gold:\n<#code#>\nbreak;\n" [[@LINE-6]]:3 -> [[@LINE-6]]:3

  switch (c) {
  case CASE(Black):
    break;
  case (Color)1: // Blue
    break;
  }
// CHECK1: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
// CHECK2: "case Color::White:\n<#code#>\nbreak;\ncase Color::Gold:\n<#code#>\nbreak;\n" [[@LINE-2]]:3 -> [[@LINE-2]]:3

  switch (c) {
  }
// CHECK1: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
// CHECK2: "case Color::Black:\n<#code#>\nbreak;\ncase Color::Blue:\n<#code#>\nbreak;\ncase Color::White:\n<#code#>\nbreak;\ncase Color::Gold:\n<#code#>\nbreak;\n" [[@LINE-2]]:3 -> [[@LINE-2]]:3
}

#ifdef NESTED1NS
}
#endif

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 -at=%s:54:3 -at=%s:63:3 %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 -at=%s:54:3 -at=%s:63:3 %s -std=c++11 -D ENUMCLASS | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 %s -std=c++11 -D NESTED1 | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 %s -std=c++11 -D NESTED1 -D ENUMCLASS | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 %s -std=c++11 -D NESTED1 -D NESTED1NS | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 %s -std=c++11 -D NESTED1 -D NESTED1NS -D ENUMCLASS | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 -at=%s:54:3 -at=%s:63:3 %s -std=c++11 -D NESTEDANON | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:43:3 -at=%s:54:3 -at=%s:63:3 %s -std=c++11 -D NESTEDANON -D ENUMCLASS | FileCheck --check-prefix=CHECK2 %s

#define MACROARG(X) X

void macroArg(PREFIX Color c) {
  // macro-arg: +2:12
  // macro-arg-range-begin: +1:12
  MACROARG(switch (c) {
  }); // MACRO-ARG: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE]]
  // macro-arg-range-end: -1:4
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=macro-arg %s | FileCheck --check-prefix=MACRO-ARG %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -selected=macro-arg-range %s | FileCheck --check-prefix=MACRO-ARG %s
