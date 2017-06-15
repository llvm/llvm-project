
enum Color {
  Black,
  Blue,
  White,
  Gold
};

void placeBeforeDefault(Color c) {
  switch (c) {
  case Black:
    break;
  case Blue:
    break;
  default:
    break;
  }
// CHECK1: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3

  switch (c) {
  default:
    break;
  }
// CHECK1: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-3]]:3 -> [[@LINE-3]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:10:3 -at=%s:20:3 %s | FileCheck --check-prefix=CHECK1 %s

void dontPlaceBeforeDefault(Color c) {
  switch (c) {
  default:
    break;
  case Black:
    break;
  case Blue:
    break;
  }
// CHECK2: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  case Black:
    break;
  default:
    break;
  case Blue:
    break;
  }
// CHECK2: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:30:3 -at=%s:40:3 %s | FileCheck --check-prefix=CHECK2 %s

void insertAtProperPlaces(Color c) {
  switch (c) {
  case Black:
    break;
  case White:
    break;
#ifdef USEDEFAULT
  default:
    break;
#endif
  }
// CHECK3: "case Blue:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3
// CHECK3-NEXT: "case Gold:\n<#code#>\nbreak;\n" [[@LINE-2]]:3 -> [[@LINE-2]]:3
// CHECK4: "case Blue:\n<#code#>\nbreak;\n" [[@LINE-9]]:3 -> [[@LINE-9]]:3
// CHECK4-NEXT: "case Gold:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3

  switch (c) {
  case White:
    break;
#ifdef USEDEFAULT
  default:
    break;
#endif
  }
// CHECK3: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3
// CHECK3-NEXT: "case Gold:\n<#code#>\nbreak;\n" [[@LINE-2]]:3 -> [[@LINE-2]]:3
// CHECK4: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\n" [[@LINE-9]]:3 -> [[@LINE-9]]:3
// CHECK4-NEXT: "case Gold:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3

  switch (c) {
  case Gold:
    break;
#ifdef USEDEFAULT
  default:
    break;
#endif
  }
// CHECK3: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3
// CHECK4: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\n" [[@LINE-8]]:3 -> [[@LINE-8]]:3

  switch (c) {
  case Blue:
    break;
#ifdef USEDEFAULT
  default:
    break;
#endif
  }
// CHECK3: "case Black:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3
// CHECK3-NEXT: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-2]]:3 -> [[@LINE-2]]:3
// CHECK4: "case Black:\n<#code#>\nbreak;\n" [[@LINE-9]]:3 -> [[@LINE-9]]:3
// CHECK4-NEXT: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-7]]:3 -> [[@LINE-7]]:3

  switch (c) {
  case White:
    break;
  case Gold:
    break;
#ifdef USEDEFAULT
  default:
    break;
#endif
  }
// CHECK3: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\n" [[@LINE-9]]:3 -> [[@LINE-9]]:3
// CHECK4: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\n" [[@LINE-10]]:3 -> [[@LINE-10]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:54:3 -at=%s:69:3 -at=%s:82:3 -at=%s:93:3 -at=%s:106:3 %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:54:3 -at=%s:69:3 -at=%s:82:3 -at=%s:93:3 -at=%s:106:3 %s -D USEDEFAULT | FileCheck --check-prefix=CHECK4 %s

void insertAtEndIfOrderingIsUncertain(Color c) {
  switch (c) {
  case Gold:
    break;
  case White:
    break;
  }
// CHECK5: "case Black:\n<#code#>\nbreak;\ncase Blue:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  case Blue:
    break;
  case Black:
    break;
  }
// CHECK5: "case White:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  case White:
    break;
  case Black:
    break;
  }
// CHECK5: "case Blue:\n<#code#>\nbreak;\ncase Gold:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3

  switch (c) {
  case Gold:
    break;
  case Blue:
    break;
  }
// CHECK5: "case Black:\n<#code#>\nbreak;\ncase White:\n<#code#>\nbreak;\n" [[@LINE-1]]:3 -> [[@LINE-1]]:3
}

// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=%s:124:3 -at=%s:132:3 -at=%s:140:3 -at=%s:148:3 %s | FileCheck --check-prefix=CHECK5 %s
