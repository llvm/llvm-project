// RUN:  %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve \
// RUN:  -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

void test_locals(void) {
  // CHECK-DAG: name: "__SVBool_t",{{.*}}, baseType: ![[CT1:[0-9]+]]
  // CHECK-DAG: ![[CT1]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYU8:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS8:[0-9]+]], bitStride: i64 1)
  // CHECK-DAG: ![[ELTTYU8]] = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
  // CHECK-DAG: ![[ELTS8]] = !{![[REALELTS8:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS8]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 8, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __SVBool_t b8;

  // CHECK-DAG: name: "__SVCount_t",{{.*}}, baseType: ![[CT1_2:[0-9]+]]
  // CHECK-DAG: ![[CT1_2]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYU8]], flags: DIFlagVector, elements: ![[ELTS0_to_1:[0-9]+]])
  // CHECK-DAG: ![[ELTS0_to_1]] = !{![[REALELTS0_to_1:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS0_to_1]] = !DISubrange(lowerBound: 0, upperBound: 1)
  __SVCount_t c8;

  // CHECK-DAG: name: "__SVInt8_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
  // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYS8:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS8:[0-9]+]])
  // CHECK-DAG: ![[ELTTYS8]] = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
  __SVInt8_t s8;

  // CHECK-DAG: name: "__SVUint8_t",{{.*}}, baseType: ![[CT8:[0-9]+]]
  // CHECK-DAG: ![[CT8]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTYU8]], flags: DIFlagVector, elements: ![[ELTS8]])
  __SVUint8_t u8;

  // CHECK-DAG: name: "__SVInt16_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
  // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS16:[0-9]+]])
  // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "short", size: 16, encoding: DW_ATE_signed)
  // CHECK-DAG: ![[ELTS16]] = !{![[REALELTS16:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS16]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 4, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __SVInt16_t s16;

  // CHECK-DAG: name: "__SVUint16_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
  // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS16]])
  // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
  __SVUint16_t u16;

  // CHECK-DAG: name: "__SVInt32_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
  // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS32:[0-9]+]])
  // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
  // CHECK-DAG: ![[ELTS32]] = !{![[REALELTS32:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS32]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 2, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __SVInt32_t s32;

  // CHECK-DAG: name: "__SVUint32_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
  // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS32]])
  // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
  __SVUint32_t u32;

  // CHECK-DAG: name: "__SVInt64_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
  // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS64:[0-9]+]])
  // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
  // CHECK-DAG: ![[ELTS64]] = !{![[REALELTS64:[0-9]+]]}
  // CHECK-DAG: ![[REALELTS64]] = !DISubrange(lowerBound: 0, upperBound: !DIExpression(DW_OP_constu, 1, DW_OP_bregx, 46, 0, DW_OP_mul, DW_OP_constu, 1, DW_OP_minus))
  __SVInt64_t s64;

  // CHECK-DAG: name: "__SVUint64_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
  // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS64]])
  // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "unsigned long", size: 64, encoding: DW_ATE_unsigned)
  __SVUint64_t u64;

  // CHECK:     name: "__SVFloat16_t",{{.*}}, baseType: ![[CT16:[0-9]+]]
  // CHECK-DAG: ![[CT16]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY16:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS16]])
  // CHECK-DAG: ![[ELTTY16]] = !DIBasicType(name: "__fp16", size: 16, encoding: DW_ATE_float)
  __SVFloat16_t f16;

  // CHECK:     name: "__SVFloat32_t",{{.*}}, baseType: ![[CT32:[0-9]+]]
  // CHECK-DAG: ![[CT32]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY32:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS32]])
  // CHECK-DAG: ![[ELTTY32]] = !DIBasicType(name: "float", size: 32, encoding: DW_ATE_float)
  __SVFloat32_t f32;

  // CHECK:     name: "__SVFloat64_t",{{.*}}, baseType: ![[CT64:[0-9]+]]
  // CHECK-DAG: ![[CT64]] = !DICompositeType(tag: DW_TAG_array_type, baseType: ![[ELTTY64:[0-9]+]], flags: DIFlagVector, elements: ![[ELTS64]])
  // CHECK-DAG: ![[ELTTY64]] = !DIBasicType(name: "double", size: 64, encoding: DW_ATE_float)
  __SVFloat64_t f64;
}
