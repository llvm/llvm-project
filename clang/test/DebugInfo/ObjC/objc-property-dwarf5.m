// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -dwarf-version=5 %s -o - | FileCheck %s

@protocol NSObject
@end

@interface NSObject <NSObject> {}
@end

struct Bar {};

@protocol BarProto
@property struct Bar *bar;
@end

@interface Foo <BarProto>
@end

@implementation Foo {}
@synthesize bar = _bar;
- (void)f {}
@end

// CHECK: ![[FOO:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "Foo"

// CHECK: ![[DECL:[0-9]+]] = !DISubprogram(name: "-[Foo setBar:]",
// CHECK-SAME:  scope: ![[FOO]]

// CHECK: distinct !DISubprogram(name: "-[Foo setBar:]",
// CHECK-SAME:  declaration: ![[DECL:[0-9]+]]
