// RUN: %check_clang_tidy %s objc-missing-hash %t

typedef _Bool BOOL;
#define YES 1
#define NO 0
typedef unsigned int NSUInteger;
typedef void *id;

@interface NSObject
- (NSUInteger)hash;
- (BOOL)isEqual:(id)object;
@end

@interface MissingHash : NSObject
@end

@implementation MissingHash
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'MissingHash' implements -isEqual: without implementing -hash [objc-missing-hash]

- (BOOL)isEqual:(id)object {
  return YES;
}

@end

@interface HasHash : NSObject
@end

@implementation HasHash

- (NSUInteger)hash {
  return 0;
}

- (BOOL)isEqual:(id)object {
  return YES;
}

@end

@interface NSArray : NSObject
@end

@interface MayHaveInheritedHash : NSArray
@end

@implementation MayHaveInheritedHash

- (BOOL)isEqual:(id)object {
  return YES;
}

@end

@interface AnotherRootClass
@end

@interface NotDerivedFromNSObject : AnotherRootClass
@end

@implementation NotDerivedFromNSObject

- (BOOL)isEqual:(id)object {
  return NO;
}

@end

