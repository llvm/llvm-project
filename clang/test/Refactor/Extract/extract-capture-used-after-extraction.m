void takesInt(int);
void passByPointer() {
  int x = 1;
  int y = 0, z = 2;
  takesInt(x);
  z = 2;
  x, z;
}
// CHECK1: "static void extracted(int *x, int *z) {\n*x = 1;\n  int y = 0; *z = 2;\n  takesInt(*x);\n  *z = 2;\n}\n\n"
// CHECK1-NEXT: "int x;\nint z;\nextracted(&x, &z)"
// RUN: clang-refactor-test perform -action extract -selected=%s:3:3-6:8 %s | FileCheck --check-prefix=CHECK1 %s

typedef struct { int width; int height; } Rectangle;

void handleStructInit() {
  Rectangle a = { 5, 6 };
  Rectangle b = { 1, 2 }, c = { 3, 4 };
  takesInt(a.width);
  c.height = 10;
  a, c;
}
// The produced code is invalid but we can let the user deal with it:
// CHECK2: "static void extracted(Rectangle *a, Rectangle *c) {\n*a = { 5, 6 };\n  Rectangle b = { 1, 2 }; *c = { 3, 4 };\n  takesInt(a->width);\n  c->height = 10;\n}\n\n" 15:1 -> 15:1
// CHECK2: "Rectangle a;\nRectangle c;\nextracted(&a, &c)"
// RUN: clang-refactor-test perform -action extract -selected=%s:16:3-19:16 %s | FileCheck --check-prefix=CHECK2 %s
