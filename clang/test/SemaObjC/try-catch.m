// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-exceptions %s
typedef signed char BOOL;
typedef struct _NSZone NSZone;

@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;

@protocol NSObject
- (BOOL)isEqual:(id)object;
@end

@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end

@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end

@interface NSObject <NSObject> {}
@end

@class NSData, NSArray, NSDictionary, NSCharacterSet, NSData, NSURL, NSError, NSLocale;

@interface NSException : NSObject <NSCopying, NSCoding> {}
@end

@class ASTNode, XCRefactoringParser, Transform, TransformInstance, XCRefactoringSelectionInfo;

@interface XCRefactoringTransformation : NSObject {}
@end

@implementation XCRefactoringTransformation
- (NSDictionary *)setUpInfoForTransformKey:(NSString *)transformKey outError:(NSError **)outError {
    @try {}
    // the exception name is optional (weird)
    @catch (NSException *) {}
} // expected-warning {{non-void function does not return a value}}

- (NSDictionary *)anotherFunction {
    @try {}
    @finally {}
} // FIXME: This should warn about a missing return too.
@end

int foo() {
  struct s { int a, b; } agg, *pagg;

  @throw 42; // expected-error {{@throw requires an Objective-C object type ('int' invalid)}}
  @throw agg; // expected-error {{@throw requires an Objective-C object type ('struct s' invalid)}}
  @throw pagg; // expected-error {{@throw requires an Objective-C object type ('struct s *' invalid)}}
  @throw; // expected-error {{@throw (rethrow) used outside of a @catch block}}
}
