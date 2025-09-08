// RUN: %clang_cc1 -std=c++11 -emit-llvm -debug-info-kind=limited -o - %s | FileCheck %s
// Test that we are emitting debug info and base types for scoped enums.

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Color"
// CHECK-SAME:             baseType: ![[INT:[0-9]+]]
// CHECK: ![[INT]] = !DIBasicType(name: "int"
enum class Color { gray };

void f(Color);
void g() {
  f(Color::gray);
}

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Colour"
// CHECK-SAME:             baseType: ![[INT]]
enum struct Colour { grey };

void h(Colour);
void i() {
  h(Colour::grey);
}

// CHECK: !DICompositeType(tag: DW_TAG_enumeration_type, name: "Couleur"
// CHECK-SAME:             baseType: ![[UCHAR:[0-9]+]]
// CHECK: ![[UCHAR]] = !DIBasicType(name: "unsigned char"
enum class Couleur : unsigned char { gris };

void j(Couleur);
void k() {
  j(Couleur::gris);
}
