// Test that synthesized accessors get treated like regular method declarations/definitions.
// I.e.:
// 1. explicitly passed parameter are not marked artificial.
// 2. Each property accessor has a method declaration and definition.

// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -dwarf-version=5 -debug-info-kind=limited %s -o - | FileCheck %s --implicit-check-not "DIFlagArtificial"

@interface Foo
@property int p1;
@end

@implementation Foo
@end

int main(void) {
  Foo *f;
  f.p1 = 2;
  return f.p1;
}

// CHECK: ![[P1_TYPE:[0-9]+]] = !DIBasicType(name: "int"
// CHECK: ![[GETTER_DECL:[0-9]+]] = !DISubprogram(name: "-[Foo p1]"
// CHECK-SAME:                                    type: ![[GETTER_TYPE:[0-9]+]]
// CHECK-SAME:                                    flags: DIFlagArtificial | DIFlagPrototyped
// CHECK-SAME:                                    spFlags: DISPFlagLocalToUnit)

// CHECK: ![[GETTER_TYPE]] = !DISubroutineType(types: ![[GETTER_PARAMS:[0-9]+]])
// CHECK: ![[GETTER_PARAMS]] = !{![[P1_TYPE]], ![[ID_TYPE:[0-9]+]], ![[SEL_TYPE:[0-9]+]]}
// CHECK: ![[ID_TYPE]] = !DIDerivedType(tag: DW_TAG_pointer_type
// CHECK-SAME:                          flags: DIFlagArtificial | DIFlagObjectPointer)
// CHECK: ![[SEL_TYPE]] = !DIDerivedType(tag: DW_TAG_typedef, name: "SEL"
// CHECK-SAME:                           flags: DIFlagArtificial)

// CHECK: ![[SETTER_DECL:[0-9]+]] = !DISubprogram(name: "-[Foo setP1:]"
// CHECK-SAME:                                    type: ![[SETTER_TYPE:[0-9]+]]
// CHECK-SAME:                                    flags: DIFlagArtificial | DIFlagPrototyped
// CHECK-SAME:                                    spFlags: DISPFlagLocalToUnit)
// CHECK: ![[SETTER_TYPE]] = !DISubroutineType(types: ![[SETTER_PARAMS:[0-9]+]])
// CHECK: ![[SETTER_PARAMS]] = !{null, ![[ID_TYPE]], ![[SEL_TYPE]], ![[P1_TYPE]]}

// CHECK: ![[GETTER_DEF:[0-9]+]] = distinct !DISubprogram(name: "-[Foo p1]"
// CHECK-SAME:                                            type: ![[GETTER_TYPE]]
// CHECK-SAME:                                            flags: DIFlagArtificial | DIFlagPrototyped
// CHECK-SAME:                                            spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-SAME:                                            declaration: ![[GETTER_DECL]]

// CHECK: !DILocalVariable(name: "self", arg: 1, scope: ![[GETTER_DEF]]
// CHECK-SAME:             flags: DIFlagArtificial | DIFlagObjectPointer)
//
// CHECK: !DILocalVariable(name: "_cmd", arg: 2, scope: ![[GETTER_DEF]],
// CHECK-SAME:             flags: DIFlagArtificial)

// CHECK: ![[SETTER_DEF:[0-9]+]] = distinct !DISubprogram(name: "-[Foo setP1:]",
// CHECK-SAME:                                            type: ![[SETTER_TYPE]]
// CHECK-SAME:                                            flags: DIFlagArtificial | DIFlagPrototyped
// CHECK-SAME:                                            spFlags: DISPFlagLocalToUnit | DISPFlagDefinition
// CHECK-SAME:                                            declaration: ![[SETTER_DECL]]

// CHECK: !DILocalVariable(name: "self", arg: 1, scope: ![[SETTER_DEF]]
// CHECK-SAME:             flags: DIFlagArtificial | DIFlagObjectPointer
// CHECK: !DILocalVariable(name: "_cmd", arg: 2, scope: ![[SETTER_DEF]]
// CHECK-SAME:             flags: DIFlagArtificial
// CHECK: !DILocalVariable(name: "p1", arg: 3, scope: ![[SETTER_DEF]]
