// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end // expected-note {{method 'copyWithZone:' declared here}}
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
typedef struct {} NSFastEnumerationState;
@protocol NSFastEnumeration  - (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len; @end
@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>  - (NSUInteger)count; @end
@interface NSMutableDictionary : NSDictionary  - (void)removeObjectForKey:(id)aKey; @end
extern NSString * const NSTaskDidTerminateNotification;

@interface XCPropertyExpansionContext : NSObject <NSCopying> { // expected-note {{required for direct or indirect protocol 'NSCopying'}}
  NSMutableDictionary * _propNamesToPropValuesCache;
} @end

@protocol XCPropertyValues <NSObject, NSCopying>
- (NSString *)evaluateAsStringInContext:(XCPropertyExpansionContext *)context withNestingState:(const void *)state;
@end

@implementation XCPropertyExpansionContext // expected-warning {{incomplete implementation}} \
					   // expected-warning {{method 'copyWithZone:' in protocol not implemented}}
- (NSString *)expandedValueForProperty:(NSString *)property {
  id <XCPropertyValues> cachedValueNode = [_propNamesToPropValuesCache objectForKey:property]; // expected-warning {{method '-objectForKey:' not found (return type defaults to 'id')}}
  if (cachedValueNode == ((void *)0)) { }
  NSString * expandedValue = [cachedValueNode evaluateAsStringInContext:self withNestingState:((void *)0)];
  return expandedValue;
}
@end
