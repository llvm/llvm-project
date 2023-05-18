// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef signed char BOOL;
typedef struct _NSZone NSZone;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@interface NSObject <NSObject> {}
@end

@class NSString, NSData, NSMutableData, NSMutableDictionary, NSMutableArray;

@interface SCMObject : NSObject <NSCopying> {}
  @property(assign) SCMObject *__attribute__((objc_gc(weak))) parent;
@end

@interface SCMNode : SCMObject
{
  NSString *_name;
}
@property(copy) NSString *name;
@end

@implementation SCMNode
  @synthesize name = _name;
  - (void) setParent:(SCMObject *__attribute__((objc_gc(weak)))) inParent {
    super.parent = inParent;
  }
@end
