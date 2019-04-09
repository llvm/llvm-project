// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: c-index-test -test-load-source all -comments-xml-schema=%S/../../bindings/xml/comment-xml-schema.rng -target x86_64-apple-darwin10 -fobjc-arc %s > %t/out
// RUN: FileCheck %s < %t/out
// rdar://13757500

@class NSString;

@interface NSArray @end

@interface NSMutableArray : NSArray 
{
//! This is the name.
  NSString *Name;
}
//! This is WithLabel comment.
- (NSString *)WithLabel:(NSString * const)label;
// CHECK: <Declaration>- (NSString *)WithLabel:(NSString *const)label;</Declaration> 

//! This is a property to get the Name.
@property (copy) NSString *Name;
// CHECK: <Declaration>@property(atomic, copy, readwrite) NSString *Name;</Declaration>
@end

@implementation NSMutableArray
{
//! This is private ivar
  NSString *NickName;
// CHECK: <Declaration>NSString *NickName</Declaration>
}

- (NSString *)WithLabel:(NSString * const)label {
    return 0;
}
@synthesize Name = Name;
@end
