// RUN: %clang_cc1 -fsyntax-only -fblocks -Wno-objc-root-class -Wdocumentation -Wdocumentation-pedantic -verify %s
// RUN: %clang_cc1 -xobjective-c++ -fsyntax-only -fblocks -Wno-objc-root-class -Wdocumentation -Wdocumentation-pedantic -verify %s

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

// rdar://13094352
// expected-warning@+2 {{'@method' command should be used in a comment attached to an Objective-C method declaration}}
@interface I
/*!	@method Base64EncodeEx
*/
typedef id ID;
- (unsigned) Base64EncodeEx : (ID)Arg;
@end

// rdar://12379114
// expected-warning@+5 {{'@interface' command should not be used in a comment attached to a non-interface declaration}} 
// expected-warning@+5 {{'@classdesign' command should not be used in a comment attached to a non-container declaration}}
// expected-warning@+5 {{'@coclass' command should not be used in a comment attached to a non-container declaration}} 
@interface NSObject @end
/*!
@interface IOCommandGate
@classdesign Multiple paragraphs go here.
@coclass myCoClass 
*/

typedef id OBJ;
@interface IOCommandGate : NSObject {
  OBJ iv;
}
@end

// rdar://12379114
// expected-warning@+4 {{'@methodgroup' command should be used in a comment attached to an Objective-C method declaration}}
// expected-warning@+6 {{'@method' command should be used in a comment attached to an Objective-C method declaratio}}
@interface rdar12379114
/*!
 @methodgroup Creating a request
*/
/*!
 @method initWithTimeout is the 2nd method
*/
typedef unsigned int NSTimeInterval;
- (id)initWithTimeout:(NSTimeInterval)timeout;
@end

// expected-warning@+2 {{'@protocol' command should not be used in a comment attached to a non-protocol declaration}}
/*!
@protocol PROTO
*/
struct S;

/*!
  @interface NSArray This is an array
*/
@class NSArray;
@interface NSArray @end

// expected-warning@+3 {{unknown command tag name}}
/*!
@interface NSMutableArray 
@super NSArray
*/
@interface NSMutableArray : NSArray @end

/*!
  @protocol MyProto
*/
@protocol MyProto @end

// expected-warning@+2 {{'@protocol' command should not be used in a comment attached to a non-protocol declaration}}
/*!
 @protocol MyProto
*/
@interface INTF <MyProto> @end

// expected-warning@+2 {{'@struct' command should not be used in a comment attached to a non-struct declaration}}
/*!
  @struct S1 THIS IS IT
*/
@interface S1 @end

// expected-warning@+1 {{unknown command tag name}}
/// \t bbb IS_DOXYGEN_END
int FooBar(void);

// rdar://13836387
/** \brief Module handling the incoming notifications from the system.
 *
 * This includes:
 * - Network Reachability
 * - Power State
 * - Low Disk
 */
@interface BRC : NSObject
- (void)removeReach:(NSObject*)observer;
@end

@implementation BRC : NSObject
- (void)removeReach:(NSObject*)observer // expected-note {{previous declaration is here}}
{
}
- (void)removeReach:(NSObject*)observer // expected-error {{duplicate declaration of method 'removeReach:'}}
{
}
@end

// rdar://13927330
/// @class Asset  <- '@class' may be used in a comment attached to a an interface declaration
@interface Asset : NSObject
@end

// rdar://14024851 Check that this does not enter an infinite loop
@interface rdar14024851
-(void)meth; // expected-note {{declared here}}
@end

@implementation rdar14024851 // expected-warning {{method definition for 'meth' not found}} expected-note {{previous definition}}
@end

@implementation rdar14024851 // expected-error {{reimplementation}}
/// \brief comment
-(void)meth {}
@end

// rdar://14124644
@interface test_vararg1
/// @param[in] arg something
/// @param[in] ... This is vararg
- (void) VarArgMeth : (id)arg, ...;
@end

@implementation test_vararg1
/// @param[in] arg something
/// @param[in] ... This is vararg
- (void) VarArgMeth : (id)arg, ... {}
@end

/**
 * blockPointerVariable
 *
 * @param i is integer.
 * @returns integer.
 */
int (^blockPointerVariable)(int i);

struct HasFields {
  /**
   * blockPointerField
   *
   * \param i is integer.
   * \returns integer.
   */
  int (^blockPointerFields)(int i);
};

// expected-warning@+5 {{parameter 'p' not found in the function declaration}}
// expected-warning@+5 {{'\returns' command used in a comment that is attached to a function returning void}}
/**
 * functionPointerVariable
 *
 * \param p not here.
 * \returns integer.
 */
void (^_Nullable blockPointerVariableThatLeadsNowhere)(void);

@interface CheckFunctionBlockPointerVars {
  /**
   * functionPointerIVar
   *
   * @param i is integer.
   * @returns integer.
   */
  int (*functionPointerIVar)(int i);

  /**
   * blockPointerIVar
   *
   * \param i is integer.
   * \returns integer.
   */
  int (^blockPointerIVar)(int i);
}

/**
 * functionPointerProperty
 *
 * @param i is integer.
 * @returns integer.
 */
@property int (*functionPointerProperty)(int i);

/**
 * blockPointerProperty
 *
 * \param i is integer.
 * \returns integer.
 */
@property int (^blockPointerProperty)(int i);

/**
 * blockReturnsNothing
 *
 * \returns Nothing, but can allow this as this pattern is used to document the
 * value that the property getter returns.
 */
@property void (^blockReturnsNothing)(void);

@end

/*!
 * Block typedef with variadic params.
 *
 * @param a
 * works
 *
 * @param ...
 * now should work too.
 */
typedef void (^VariadicBlockType)(int a, ...);

// PR42844 - Assertion failures when using typedefed block pointers
typedef void(^VoidBlockType)(void);
typedef VoidBlockType VoidBlockTypeCall(void);
VoidBlockTypeCall *d; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}
VoidBlockTypeCall ^e; ///< \return none
// expected-warning@-1 {{'\return' command used in a comment that is not attached to a function or method declaration}}

#ifdef __cplusplus
@interface HasAnonNamespace @end
@implementation HasAnonNamespace
namespace {}
@end
#endif
