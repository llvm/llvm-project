// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:  -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

void test_locals(void) {
  // CHECK-DAG: name: "__clang_svint8x2_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
  // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY8:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS8x2:[0-9]+]])
  // CHECK-DAG: ![[ELTTY8]] = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
  // CHECK-DAG: ![[ELTS8x2]] = !{![[REALELTS8x2:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS8x2]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 16, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __clang_svint8x2_t s8;

  // CHECK-DAG: name: "__clang_svuint8x2_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
  // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY8:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS8x2]])
  // CHECK-DAG: ![[ELTTY8]] = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
  __clang_svuint8x2_t u8;

  // CHECK-DAG: name: "__clang_svint16x2_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
  // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS16x2:[0-9]+]])
  // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
  // CHECK-DAG: ![[ELTS16x2]] = !{![[REALELTS16x2:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS16x2]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 8, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __clang_svint16x2_t s16;

  // CHECK-DAG: name: "__clang_svuint16x2_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
  // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS16x2]])
  // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
  __clang_svuint16x2_t u16;

  // CHECK-DAG: name: "__clang_svint32x2_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
  // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS32x2:[0-9]+]])
  // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  // CHECK-DAG: ![[ELTS32x2]] = !{![[REALELTS32x2:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS32x2]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 4, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __clang_svint32x2_t s32;

  // CHECK-DAG: name: "__clang_svuint32x2_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
  // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS32x2]])
  // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
  __clang_svuint32x2_t u32;

  // CHECK-DAG: name: "__clang_svint64x2_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
  // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS1x2_64:[0-9]+]])
  // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
  // CHECK-DAG: ![[ELTS1x2_64]] = !{![[REALELTS1x2_64:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS1x2_64]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 2, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __clang_svint64x2_t s64;

  // CHECK-DAG: name: "__clang_svuint64x2_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
  // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS1x2_64]])
  // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
  __clang_svuint64x2_t u64;

  // CHECK:     name: "__clang_svfloat16x2_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
  // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS16x2]])
  // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "__fp16", size: 16, encoding: DW_ATE_float)
  __clang_svfloat16x2_t f16;

  // CHECK:     name: "__clang_svfloat32x2_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
  // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS32x2]])
  // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
  __clang_svfloat32x2_t f32;

  // CHECK:     name: "__clang_svfloat64x2_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
  // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS1x2_64]])
  // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
  __clang_svfloat64x2_t f64;
}
