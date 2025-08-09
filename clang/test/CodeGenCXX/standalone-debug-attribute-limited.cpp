// RUN: %clang_cc1 -DSETATTR=0 -triple x86_64-unknown-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=DEBUG
// RUN: %clang_cc1 -DSETATTR=1 -triple x86_64-unknown-linux-gnu -emit-llvm -debug-info-kind=limited %s -o - | FileCheck %s --check-prefix=WITHATTR
// Use -debug-info-kind=limited because an unused type will never have a used ctor.

#if SETATTR
#define STANDALONEDEBUGATTR __attribute__((standalone_debug))
#else
#define STANDALONEDEBUGATTR
#endif

struct STANDALONEDEBUGATTR TypeWithNested {
  struct Unused1 {
  };
  struct STANDALONEDEBUGATTR Unused2 {
  };

  int value = 0;
};
void f(TypeWithNested s) {}
// DEBUG:  !DICompositeType(tag: DW_TAG_structure_type, name: "TypeWithNested"
// DEBUG-NOT:  !DICompositeType(tag: DW_TAG_structure_type, name: "Unused1"
// DEBUG-NOT:  !DICompositeType(tag: DW_TAG_structure_type, name: "Unused2"
// WITHATTR:  !DICompositeType(tag: DW_TAG_structure_type, name: "TypeWithNested"
// WITHATTR:  !DICompositeType(tag: DW_TAG_structure_type, name: "Unused1"
// The STANDALONEDEBUGATTR isn't propagated to the nested type by default, so
// it should still be a forward declaration.
// WITHATTR-SAME: DIFlagFwdDecl
// WITHATTR:  !DICompositeType(tag: DW_TAG_structure_type, name: "Unused2"
// WITHATTR-NOT: DIFlagFwdDecl
