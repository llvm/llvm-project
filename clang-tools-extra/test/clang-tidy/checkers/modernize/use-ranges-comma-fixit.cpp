// RUN: %check_clang_tidy -std=c++20 -export-fixes=%t.yaml %s modernize-use-ranges %t -- -- -I %S/Inputs/
// RUN: FileCheck -input-file=%t.yaml %s

#include "use-ranges/fake_std.h"

void commaFixIts(std::vector<int> &I, std::vector<int> &J) {
  // Verify raw FixIt replacements cover the full "I.begin(), I.end()" span (18
  // bytes = 11 + 7) rather than skipping the comma token.
  std::reverse(I.begin(), I.end());
  // CHECK-LABEL: - DiagnosticName: modernize-use-ranges
  // CHECK: ReplacementText: 'std::ranges::reverse'
  // CHECK: Offset: [[#R1:]]
  // CHECK-NEXT: Length: 11
  // CHECK-NEXT: ReplacementText: I
  // CHECK: Offset: [[#R1 + 11]]
  // CHECK-NEXT: Length: 7
  // CHECK-NEXT: ReplacementText: ''

  std::reverse(I.begin() , I.end());
  // CHECK-LABEL: - DiagnosticName: modernize-use-ranges
  // CHECK: ReplacementText: 'std::ranges::reverse'
  // CHECK: Offset: [[#R2:]]
  // CHECK-NEXT: Length: 9
  // CHECK-NEXT: ReplacementText: I
  // CHECK: Offset: [[#R2 + 10]]
  // CHECK-NEXT: Length: 9
  // CHECK-NEXT: ReplacementText: ''

  std::find(I.begin(), I.end(), 0);
  // CHECK-LABEL: - DiagnosticName: modernize-use-ranges
  // CHECK: ReplacementText: 'std::ranges::find'
  // CHECK: Offset: [[#R3:]]
  // CHECK-NEXT: Length: 11
  // CHECK-NEXT: ReplacementText: I
  // CHECK: Offset: [[#R3 + 11]]
  // CHECK-NEXT: Length: 7
  // CHECK-NEXT: ReplacementText: ''

  std::equal(I.begin(), I.end(), J.begin(), J.end());
  // CHECK-LABEL: - DiagnosticName: modernize-use-ranges
  // CHECK: ReplacementText: 'std::ranges::equal'
  // CHECK: Offset: [[#R4A:]]
  // CHECK-NEXT: Length: 11
  // CHECK-NEXT: ReplacementText: I
  // CHECK: Offset: [[#R4A + 11]]
  // CHECK-NEXT: Length: 7
  // CHECK-NEXT: ReplacementText: ''
  // CHECK: Offset: [[#R4A + 20]]
  // CHECK-NEXT: Length: 11
  // CHECK-NEXT: ReplacementText: J
  // CHECK: Offset: [[#R4A + 31]]
  // CHECK-NEXT: Length: 7
  // CHECK-NEXT: ReplacementText: ''

#define EMPTY(a, b)
  std::reverse(I.begin() EMPTY(1, 2), I.end());
  // CHECK-LABEL: - DiagnosticName: modernize-use-ranges
  // CHECK: ReplacementText: 'std::ranges::reverse'
  // CHECK: Offset: [[#R5:]]
  // CHECK-NEXT: Length: 9
  // CHECK-NEXT: ReplacementText: I
  // CHECK: Offset: [[#R5 + 21]]
  // CHECK-NEXT: Length: 9
  // CHECK-NEXT: ReplacementText: ''
#undef EMPTY
}
