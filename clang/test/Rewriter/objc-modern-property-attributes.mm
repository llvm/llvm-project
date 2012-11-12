// RUN: %clang_cc1 -E %s -o %t.mm
// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc %t.mm -o - | FileCheck %s

// rdar://11095151

typedef void (^void_block_t)(void);

@interface PropertyClass {
    int q;
    void_block_t __completion;
    PropertyClass* YVAR;
    id ID;
}
@property int q;
@property int r;

@property (copy) void_block_t completionBlock;
@property (retain) PropertyClass* Yblock;
@property (copy) id ID;

@end

@implementation PropertyClass
@synthesize q;  // attributes should be "Ti,Vq"
@dynamic r;     // attributes should be "Ti,D"
@synthesize completionBlock=__completion; // "T@?,C,V__completion"
@synthesize Yblock = YVAR; // "T@\"PropertyClass\",&,VYVAR"
@synthesize  ID; // "T@,C,VID"
@end

// CHECK: Ti,Vq
// CHECK: Ti,D
// CHECK: T@?,C,V__completion
// CHECK: T@\"PropertyClass\",&,VYVAR

@interface Test @end
@interface Test (Category)
@property int q;
@end

@implementation Test (Category)
@dynamic q;
@end

// CHECK: {{"q","Ti,D"}}
