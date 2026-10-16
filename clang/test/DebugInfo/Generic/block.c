// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// Verify that the desired debugging type is generated for a structure
// member that is a pointer to a block.

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, scope
// CHECK-NOT:              line
// CHECK-SAME:             elements: ![[ELEMS1:.*]])
// CHECK: ![[ELEMS1]] = {{.*, .*, .*,}} ![[FPEL1:.*]], {{.*}}
// CHECK: ![[INT:.*]] = !DIBasicType(name: "int"
// CHECK: ![[FPEL1]] = {{.*}}"__FuncPtr", {{.*}}, baseType: ![[FPTY1:[0-9]+]]
// CHECK: ![[FPTY1]] = {{.*}}baseType: ![[FNTY1:[0-9]+]]
// CHECK: ![[FNTY1]] = !DISubroutineType(types: ![[VOIDVOID:[0-9]+]])
// CHECK: ![[VOIDVOID]] = !{null}
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "__block_descriptor"
// CHECK-NOT:              line
// CHECK-SAME:            )

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, scope
// CHECK-NOT:              line
// CHECK-SAME:             elements: ![[ELEMS2:.*]])
// CHECK: ![[ELEMS2]] = {{.*,.*,.*}}, ![[FPEL2:.*]], {{.*}}
// CHECK: ![[FPEL2]] = {{.*}}"__FuncPtr", {{.*}}, baseType: ![[FPTY2:[0-9]+]]
// CHECK: ![[FPTY2]] = {{.*}}baseType: ![[FNTY2:[0-9]+]]
// CHECK: ![[FNTY2]] = !DISubroutineType(types: ![[INTINT:[0-9]+]])
// CHECK: ![[INTINT]] = !{![[INT]], ![[INT]]}
struct inStruct {
  void (^voidBlockPtr)(void);
  int (^intBlockPtr)(int);
} is;
