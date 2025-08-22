// RUN: %clang_cc1 -dwarf-version=5 -emit-llvm -debug-info-kind=limited -w -triple x86_64-apple-darwin10 %s -o - | FileCheck %s --check-prefix CHECK --check-prefix DWARF5
// RUN: %clang_cc1 -dwarf-version=4 -emit-llvm -debug-info-kind=limited -w -triple x86_64-apple-darwin10 %s -o - | FileCheck %s --check-prefix CHECK --check-prefix DWARF4

@interface Foo {
  int integer;
}

- (int)integer;
- (id)integer:(int)_integer;
@end

@implementation Foo
- (int)integer {
  return integer;
}

- (id)integer:(int)_integer {
  integer = _integer;
  return self;
}
@end

@interface Foo (Bar)
+ (id)zero:(Foo *)zeroend;
- (id)add:(Foo *)addend;
@end

@implementation Foo (Bar)
+ (id)zero:(Foo *)zeroend {
  return [self integer:0];
}
- (id)add:(Foo *)addend {
  return [self integer:[self integer] + [addend integer]];
}
@end

// CHECK: ![[STRUCT:.*]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo"

// Verify "not a definition" by showing spFlags doesn't have DISPFlagDefinition.
// DWARF5: !DISubprogram(name: "-[Foo integer]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit)
// DWARF5: !DISubprogram(name: "-[Foo integer:]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit)
// DWARF5: !DISubprogram(name: "+[Foo(Bar) zero:]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit)
// DWARF5: !DISubprogram(name: "-[Foo(Bar) add:]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit)

// DWARF4-NOT: !DISubprogram(name: "-[Foo integer]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit,
// DWARF4-NOT: !DISubprogram(name: "-[Foo integer:]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit,
// DWARF4-NOT: !DISubprogram(name: "+[Foo(Bar) zero:]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit,
// DWARF4-NOT: !DISubprogram(name: "-[Foo(Bar) add:]", scope: ![[STRUCT]], {{.*}} spFlags: DISPFlagLocalToUnit,

// CHECK: = distinct !DISubprogram(name: "-[Foo integer]"{{.*}} DISPFlagDefinition
// CHECK: = distinct !DISubprogram(name: "-[Foo integer:]"{{.*}} DISPFlagDefinition
// CHECK: = distinct !DISubprogram(name: "+[Foo(Bar) zero:]"{{.*}} DISPFlagDefinition
// CHECK: = distinct !DISubprogram(name: "-[Foo(Bar) add:]"{{.*}} DISPFlagDefinition
