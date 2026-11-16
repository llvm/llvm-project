// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o %t.ll
// RUN: FileCheck %s < %t.ll
// RUN: FileCheck %s --check-prefix=ROOT < %t.ll
// RUN: FileCheck %s --check-prefix=SAME < %t.ll

// The Objective-C runtime name of a class is recorded in the debug info when
// it differs from the source name.

__attribute__((objc_root_class))
@interface Root
@end

@implementation Root
@end

__attribute__((objc_runtime_name("_$s4main7DefinedC")))
@interface Defined : Root
@end

@implementation Defined
@end

// A class implemented in another TU, such as a Swift class declared in a
// generated header, is completed at the end of the TU and must carry the
// runtime name as well.
__attribute__((objc_runtime_name("_$s4main8DeclaredC")))
@interface Declared : Root
@end

Declared *d;

__attribute__((objc_runtime_name("Same")))
@interface Same : Root
@end

@implementation Same
@end

// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Defined",{{.*}}annotations: ![[DEFINED:[0-9]+]]
// CHECK-DAG: ![[DEFINED]] = !{![[DEFINED_NAME:[0-9]+]]}
// CHECK-DAG: ![[DEFINED_NAME]] = !{!"objc_runtime_name", !"_$s4main7DefinedC"}
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "Declared",{{.*}}annotations: ![[DECLARED:[0-9]+]]
// CHECK-DAG: ![[DECLARED]] = !{![[DECLARED_NAME:[0-9]+]]}
// CHECK-DAG: ![[DECLARED_NAME]] = !{!"objc_runtime_name", !"_$s4main8DeclaredC"}

// ROOT: !DICompositeType(tag: DW_TAG_structure_type, name: "Root",
// ROOT-NOT: annotations:
// ROOT-SAME: {{$}}

// SAME: !DICompositeType(tag: DW_TAG_structure_type, name: "Same",
// SAME-NOT: annotations:
// SAME-SAME: {{$}}
