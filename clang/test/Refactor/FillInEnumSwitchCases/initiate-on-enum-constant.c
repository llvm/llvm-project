// RUN: clang-refactor-test perform -action fill-in-enum-switch-cases -at=enum-c %s | FileCheck %s

enum Enum {
  Enum_a,
  Enum_b,
};
void testEnumConstantInCWithIntType() {
// enum-c: +1:1
switch (Enum_a) {
case Enum_a: break;
} // CHECK: "case Enum_b:\n<#code#>\nbreak;\n" [[@LINE]]:1 -> [[@LINE]]:1
}
