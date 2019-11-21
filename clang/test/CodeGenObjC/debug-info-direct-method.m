// RUN: %clang_cc1 -dwarf-version=5 -emit-llvm -debug-info-kind=limited -w -triple x86_64-apple-darwin10 %s -o - | FileCheck %s
// RUN: %clang_cc1 -dwarf-version=4 -emit-llvm -debug-info-kind=limited -w -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

__attribute__((objc_root_class))
@interface Root
@end

@implementation Root
- (int)getInt __attribute__((objc_direct)) {
  return 42;
}
@end

// Test that objc_direct methods are always (even in DWARF < 5) emitted
// as members of their containing class.

// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "Root",
// CHECK-SAME:             elements: ![[MEMBERS:[0-9]+]],
// CHECK-SAME:             runtimeLang: DW_LANG_ObjC)
// CHECK: ![[MEMBERS]] = !{![[GETTER:[0-9]+]]}
// CHECK: ![[GETTER]] = !DISubprogram(name: "-[Root getInt]",
