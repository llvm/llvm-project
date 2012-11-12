// RUN: %clang_cc1 -x objective-c++ -Wno-return-type -fblocks -fms-extensions -rewrite-objc -fobjc-runtime=macosx-fragile-10.5 %s -o %t-rw.cpp
// RUN: %clang_cc1 -fsyntax-only -Wno-address-of-temporary  -D"id=void*" -D"SEL=void*" -D"__declspec(X)=" %t-rw.cpp
// radar 7214439

typedef void (^void_block_t)(void);

@interface Y {
    void_block_t __completion;
    Y* YVAR;
    id ID;
}
@property (copy) void_block_t completionBlock;
@property (retain) Y* Yblock;
@property (copy) id ID;
@end

@implementation Y
@synthesize completionBlock=__completion;
@synthesize Yblock = YVAR;
@synthesize  ID;
@end

