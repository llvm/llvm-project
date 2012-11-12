// RUN: %clang_cc1 -triple x86_64-apple-darwin10  -fdiagnostics-parseable-fixits -x objective-c -fobjc-arc %s 2>&1 | FileCheck %s

@interface I
@property id prop;
@property (atomic) id atomic_prop;
- (id) prop;
- (id) atomic_prop;

@property (  ) id prop1;

@property (copy, atomic, readwrite) id atomic_prop1;

@property (copy, readwrite) id prop2;
@end

@implementation I
@synthesize prop, prop1, prop2;
@synthesize atomic_prop, atomic_prop1;
- (id) prop { return 0; }
- (id) prop1 { return 0; }
- (id) prop2 { return 0; }
- (id) atomic_prop { return 0; }
- (id) atomic_prop1 { return 0; }
@end

// CHECK: {4:1-4:10}:"@property (nonatomic) "
// CHECK: {9:1-9:12}:"@property (nonatomic"
// CHECK: {13:1-13:12}:"@property (nonatomic, "

