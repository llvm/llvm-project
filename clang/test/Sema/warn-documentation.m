// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-objc-root-class -Wdocumentation -Wdocumentation-pedantic -verify %s

@class NSString;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@interface Test1
// expected-warning@+2 {{empty paragraph passed to '\brief' command}}
/**
 * \brief\author Aaa
 * \param aaa Aaa
 * \param bbb Bbb
 */
+ (NSString *)test1:(NSString *)aaa suffix:(NSString *)bbb;

// expected-warning@+2 {{parameter 'aab' not found in the function declaration}} expected-note@+2 {{did you mean 'aaa'?}}
/**
 * \param aab Aaa
 */
+ (NSString *)test2:(NSString *)aaa;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@property int test3; // a property: ObjCPropertyDecl

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@property int test4; // a property: ObjCPropertyDecl
@end

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@interface Test1()
@end

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@implementation Test1 // a class implementation : ObjCImplementationDecl
+ (NSString *)test1:(NSString *)aaa suffix:(NSString *)bbb {
  return 0;
}

+ (NSString *)test2:(NSString *)aaa {
  return 0;
}

@synthesize test3; // a property implementation: ObjCPropertyImplDecl
@dynamic test4; // a property implementation: ObjCPropertyImplDecl

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
NSString *_test5;
@end

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@interface Test1(Test1Category) // a category: ObjCCategoryDecl
// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
+ (NSString *)test3:(NSString *)aaa;
@end

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@implementation Test1(Test1Category) // a category implementation: ObjCCategoryImplDecl
+ (NSString *)test3:(NSString *)aaa {
  return 0;
}
@end

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@protocol TestProto1 // a protocol: ObjCProtocolDecl
@end

int a;

// expected-warning@+1 {{empty paragraph passed to '\brief' command}}
/// \brief\author Aaa
@interface Test4
@end

int b;

@interface TestReturns1
/// \returns Aaa
- (int)test1:(NSString *)aaa;

// expected-warning@+1 {{'\returns' command used in a comment that is attached to a method returning void}}
/// \returns Aaa
- (void)test2:(NSString *)aaa;
@end

// expected-warning@+2 {{parameter 'bbb' not found in the function declaration}} expected-note@+2 {{did you mean 'ccc'?}}
/// \param aaa Meow.
/// \param bbb Bbb.
/// \returns aaa.
typedef int (^test_param1)(int aaa, int ccc);

