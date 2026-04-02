// RUN: %clang_cc1 -DSETNODEBUG=0 -gcodeview -emit-llvm -std=c++14 -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=YESINFO
// RUN: %clang_cc1 -DSETNODEBUG=1 -gcodeview -emit-llvm -std=c++14 -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=NOINFO

#if SETNODEBUG
#define NODEBUG __attribute__((nodebug))
#else
#define NODEBUG
#endif

struct t1 {
  using t2 NODEBUG = void;
};
void func6() {
  t1 v1;
}
// YESINFO-DAG: !DIDerivedType(tag: DW_TAG_typedef, name: "t2"
// NOINFO: [[ELEMENTS:!.*]] = !{}
// NOINFO: !DICompositeType(tag: DW_TAG_structure_type, name: "t1", {{.*}}, elements: [[ELEMENTS]],
// NOINFO-NOT:  !DIDerivedType(tag: DW_TAG_typedef, name: "t2"

