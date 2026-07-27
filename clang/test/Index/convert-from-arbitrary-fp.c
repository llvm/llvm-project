// RUN: c-index-test -test-load-source all %s | FileCheck %s

typedef float dst_t;

float convert(unsigned char bits) {
  return __builtin_convert_from_arbitrary_fp(bits, "Float8E5M2", dst_t);
}

// CHECK: convert-from-arbitrary-fp.c:3:15: TypedefDecl=dst_t:3:15 (Definition)
// CHECK: convert-from-arbitrary-fp.c:6:10: UnexposedExpr=
// CHECK-NEXT: convert-from-arbitrary-fp.c:6:66: TypeRef=dst_t:3:15
// CHECK-NEXT: convert-from-arbitrary-fp.c:6:46: UnexposedExpr=bits:5:29
// CHECK-NEXT: convert-from-arbitrary-fp.c:6:46: DeclRefExpr=bits:5:29
// CHECK-NEXT: convert-from-arbitrary-fp.c:6:52: StringLiteral=
