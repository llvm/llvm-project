
enum Color {
  Black,
  Blue,
  White,
  Gold
};

// Wrap the inserted case bodies in '{' '}' when the majority of others are
// wrapped as well.
void wrapInBraces(Color c) {
  switch (c) {
  case Black: {
    int x = 0;
    break;
  }
  }
// CHECK1: "case Blue: {\n<#code#>\nbreak;\n}\ncase White: {\n<#code#>\nbreak;\n}\ncase Gold: {\n<#code#>\nbreak;\n}\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  case Black: {
    int x = 0;
    break;
  }
  case Blue: {
    int y = 0;
    break;
  }
  }
// CHECK1: "case White: {\n<#code#>\nbreak;\n}\ncase Gold: {\n<#code#>\nbreak;\n}\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:12:3 -at=%s:20:3 %s | FileCheck --check-prefix=CHECK1 %s

void dontWrapInBraces(Color c) {
  switch (c) {
  case Black: {
    int x = 0;
    break;
  }
  case Blue:
    break;
  }
// CHECK2: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  case Black: {
    int x = 0;
    break;
  }
  case Blue:
    break;
  case White:
    break;
  }
// CHECK2: "case Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  }
// CHECK2: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:36:3 -at=%s:46:3 -at=%s:58:3 %s | FileCheck --check-prefix=CHECK2 %s
