// RUN: %clang_cc1 -fblocks %s -verify

#if !__has_feature(objc_generics)
#  error Compiler does not support Objective-C generics?
#endif

#if !__has_feature(objc_generics_variance)
#  error Compiler does not support co- and contr-variance?
#endif

@protocol NSObject // expected-note{{'NSObject' declared here}}
@end

@protocol NSCopying // expected-note{{'NSCopying' declared here}}
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject> // expected-note{{'NSObject' defined here}}
@end

@interface NSString : NSObject <NSCopying>
@end

// --------------------------------------------------------------------------
// Parsing parameterized classes.
// --------------------------------------------------------------------------

// Parse type parameters with a bound
@interface PC1<T, U : NSObject*> : NSObject // expected-note{{'PC1' declared here}}
// expected-note@-1{{type parameter 'T' declared here}}
// expected-note@-2{{type parameter 'U' declared here}}
// expected-note@-3{{type parameter 'U' declared here}}
@end

// Parse a type parameter with a bound that terminates in '>>'.
@interface PC2<T : id<NSObject>> : NSObject
@end

// Parse multiple type parameters.
@interface PC3<T, U : id> : NSObject
@end

// Parse multiple type parameters--grammatically ambiguous with protocol refs.
@interface PC4<T, U, V> : NSObject // expected-note 2{{'PC4' declared here}}
@end

// Parse a type parameter list without a superclass.
@interface PC5<T : id>
@end

// Parse a type parameter with name conflicts.
@interface PC6<T, U, 
               T> : NSObject // expected-error{{redeclaration of type parameter 'T'}}
@end

// Parse Objective-C protocol references.
@interface PC7<T> // expected-error{{cannot find protocol declaration for 'T'}}
@end

// Parse both type parameters and protocol references.
@interface PC8<T> : NSObject <NSObject>
@end

// Type parameters with improper bounds.
@interface PC9<T : int, // expected-error{{type bound 'int' for type parameter 'T' is not an Objective-C pointer type}}
               U : NSString> : NSObject // expected-error{{missing '*' in type bound 'NSString' for type parameter 'U'}}
@end

// --------------------------------------------------------------------------
// Parsing parameterized forward declarations classes.
// --------------------------------------------------------------------------

// Okay: forward declaration without type parameters.
@class PC10;

// Okay: forward declarations with type parameters.
@class PC10<T, U : NSObject *>, PC11<T : NSObject *, U : id>; // expected-note{{type parameter 'T' declared here}}

// Okay: forward declaration without type parameters following ones
// with type parameters.
@class PC10, PC11;

// Okay: definition of class with type parameters that was formerly
// declared with the same type parameters.
@interface PC10<T, U : NSObject *> : NSObject
@end

// Mismatched parameters in declaration of @interface following @class.
@interface PC11<T, U> : NSObject // expected-error{{missing type bound 'NSObject *' for type parameter 'T' in @interface}}
@end

@interface PC12<T : NSObject *> : NSObject  // expected-note{{type parameter 'T' declared here}}
@end

@class PC12;

// Mismatched parameters in subsequent forward declarations.
@class PC13<T : NSObject *>; // expected-note{{type parameter 'T' declared here}}
@class PC13;
@class PC13<U>; // expected-error{{missing type bound 'NSObject *' for type parameter 'U' in @class}}

// Mismatch parameters in declaration of @class following @interface.
@class PC12<T>; // expected-error{{missing type bound 'NSObject *' for type parameter 'T' in @class}}

// Parameterized forward declaration a class that is not parameterized.
@class NSObject<T>; // expected-error{{forward declaration of non-parameterized class 'NSObject' cannot have type parameters}}
// expected-note@-1{{'NSObject' declared here}}

// Parameterized forward declaration preceding the definition (that is
// not parameterized).
@class NSNumber<T : NSObject *>; // expected-note{{'NSNumber' declared here}}
@interface NSNumber : NSObject // expected-error{{class 'NSNumber' previously declared with type parameters}}
@end

@class PC14;

// Okay: definition of class with type parameters that was formerly
// declared without type parameters.
@interface PC14<T, U : NSObject *> : NSObject
@end

// --------------------------------------------------------------------------
// Parsing parameterized categories and extensions.
// --------------------------------------------------------------------------

// Inferring type bounds
@interface PC1<T, U> (Cat1) <NSObject>
@end

// Matching type bounds
@interface PC1<T : id, U : NSObject *> (Cat2) <NSObject>
@end

// Inferring type bounds
@interface PC1<T, U> () <NSObject>
@end

// Matching type bounds
@interface PC1<T : id, U : NSObject *> () <NSObject>
@end

// Missing type parameters.
@interface PC1<T> () // expected-error{{extension has too few type parameters (expected 2, have 1)}}
@end

// Extra type parameters.
@interface PC1<T, U, V> (Cat3) // expected-error{{category has too many type parameters (expected 2, have 3)}}
@end

// Mismatched bounds.
@interface PC1<T : NSObject *, // expected-error{{type bound 'NSObject *' for type parameter 'T' conflicts with implicit bound 'id'}}
               X : id> () // expected-error{{type bound 'id' for type parameter 'X' conflicts with previous bound 'NSObject *'for type parameter 'U'}}
@end

// Parameterized category/extension of non-parameterized class.
@interface NSObject<T> (Cat1) // expected-error{{category of non-parameterized class 'NSObject' cannot have type parameters}}
@end

@interface NSObject<T> () // expected-error{{extension of non-parameterized class 'NSObject' cannot have type parameters}}
@end

// --------------------------------------------------------------------------
// @implementations cannot have type parameters
// --------------------------------------------------------------------------
@implementation PC1<T : id> // expected-error{{@implementation cannot have type parameters}}
@end

@implementation PC2<T> // expected-error{{@implementation declaration cannot be protocol qualified}}
@end

@implementation PC1<T> (Cat1) // expected-error{{@implementation cannot have type parameters}}
@end

@implementation PC1<T : id> (Cat2) // expected-error{{@implementation cannot have type parameters}}
@end

typedef T undeclaredT; // expected-error{{unknown type name 'T'}}

// --------------------------------------------------------------------------
// Interfaces involving type parameters
// --------------------------------------------------------------------------
@interface PC20<T : id, U : NSObject *, V : NSString *> : NSObject {
  T object;
}

- (U)method:(V)param;
@end

@interface PC20<T, U, V> (Cat1)
- (U)catMethod:(V)param;
@end

@interface PC20<X, Y, Z>()
- (X)extMethod:(Y)param;
@end

// --------------------------------------------------------------------------
// Parsing type arguments.
// --------------------------------------------------------------------------

typedef NSString * ObjCStringRef; // expected-note{{'ObjCStringRef' declared here}}

// Type arguments with a mix of identifiers and type-names.
typedef PC4<id, NSObject *, NSString *> typeArgs1;

// Type arguments with only identifiers.
typedef PC4<id, id, id> typeArgs2;

// Type arguments with only identifiers; one is ambiguous (resolved as
// types).
typedef PC4<NSObject, id, id> typeArgs3; // expected-error{{type argument 'NSObject' must be a pointer (requires a '*')}}

// Type arguments with only identifiers; one is ambiguous (resolved as
// protocol qualifiers).
typedef PC4<NSObject, NSCopying> protocolQuals1;

// Type arguments and protocol qualifiers.
typedef PC4<id, NSObject *, id><NSObject, NSCopying> typeArgsAndProtocolQuals1;

// Type arguments and protocol qualifiers in the wrong order.
typedef PC4<NSObject, NSCopying><id, NSObject *, id> typeArgsAndProtocolQuals2; // expected-error{{protocol qualifiers must precede type arguments}}

// Type arguments and protocol qualifiers (identifiers).
typedef PC4<id, NSObject, id><NSObject, NSCopying> typeArgsAndProtocolQuals3; // expected-error{{type argument 'NSObject' must be a pointer (requires a '*')}}

// Typo correction: protocol bias.
typedef PC4<NSCopying, NSObjec> protocolQuals2; // expected-error{{cannot find protocol declaration for 'NSObjec'; did you mean 'NSObject'?}}

// Typo correction: type bias.
typedef PC4<id, id, NSObjec> typeArgs4; // expected-error{{unknown class name 'NSObjec'; did you mean 'NSObject'?}}
// expected-error@-1{{type argument 'NSObject' must be a pointer (requires a '*')}}

// Typo correction: bias set by correction itself to a protocol.
typedef PC4<NSObject, NSObject, NSCopyin> protocolQuals3; // expected-error{{cannot find protocol declaration for 'NSCopyin'; did you mean 'NSCopying'?}}

// Typo correction: bias set by correction itself to a type.
typedef PC4<NSObject, NSObject, ObjCStringref> typeArgs5; // expected-error{{unknown type name 'ObjCStringref'; did you mean 'ObjCStringRef'?}}
// expected-error@-1{{type argument 'NSObject' must be a pointer (requires a '*')}}
// expected-error@-2{{type argument 'NSObject' must be a pointer (requires a '*')}}

// Type/protocol conflict.
typedef PC4<NSCopying, ObjCStringRef> typeArgsProtocolQualsConflict1; // expected-error{{angle brackets contain both a type ('ObjCStringRef') and a protocol ('NSCopying')}}
typedef PC4<NSCopying, NSString *> typeArgsProtocolQualsConflict2; // expected-error{{angle brackets contain both a type ('NSString') and a protocol ('NSCopying')}}
typedef PC4<NSCopying, UnknownType, NSString *> typeArgsProtocolQualsConflict3; // expected-error{{angle brackets contain both a type ('NSString') and a protocol ('NSCopying')}} expected-error{{unknown type name 'UnknownType'}}
typedef PC4<UnknownType, NSString *> typeArgsProtocolQualsConflict4; // expected-error{{unknown type name 'UnknownType'}}
typedef PC4<NSString, NSCopying, NSString *> typeArgsProtocolQualsConflict5; // expected-error{{angle brackets contain both a type ('NSString') and a protocol ('NSCopying')}}

// Handling the '>>' in type argument lists.
typedef PC4<id<NSCopying>, NSObject *, id<NSObject>> typeArgs6;

// --------------------------------------------------------------------------
// Checking type arguments.
// --------------------------------------------------------------------------

@interface PC15<T : id, U : NSObject *, V : id<NSCopying>> : NSObject
// expected-note@-1{{type parameter 'V' declared here}}
// expected-note@-2{{type parameter 'V' declared here}}
// expected-note@-3{{type parameter 'U' declared here}}
@end

typedef PC4<NSString *> tooFewTypeArgs1; // expected-error{{too few type arguments for class 'PC4' (have 1, expected 3)}}

typedef PC4<NSString *, NSString *, NSString *, NSString *> tooManyTypeArgs1; // expected-error{{too many type arguments for class 'PC4' (have 4, expected 3)}}

typedef PC15<int (^)(int, int), // block pointers as 'id'
             NSString *, // subclass
             NSString *> typeArgs7; // class that conforms to the protocol

typedef PC15<NSObject *, NSObject *, id<NSCopying>> typeArgs8;

typedef PC15<NSObject *, NSObject *,
             NSObject *> typeArgs8b; // expected-error{{type argument 'NSObject *' does not satisfy the bound ('id<NSCopying>') of type parameter 'V'}}

typedef PC15<id,
             id,  // expected-error{{type argument 'id' does not satisfy the bound ('NSObject *') of type parameter 'U'}}
             id> typeArgs9;

typedef PC15<id, NSObject *,
             id> typeArgs10; // expected-error{{type argument 'id' does not satisfy the bound ('id<NSCopying>') of type parameter 'V'}}

typedef PC15<id,
             int (^)(int, int), // okay
             id<NSCopying, NSObject>> typeArgs11;

typedef PC15<id, NSString *, int (^)(int, int)> typeArgs12; // okay

typedef NSObject<id, id> typeArgs13; // expected-error{{type arguments cannot be applied to non-parameterized class 'NSObject'}}

typedef id<id, id> typeArgs14; // expected-error{{type arguments cannot be applied to non-class type 'id'}}

typedef PC1<NSObject *, NSString *> typeArgs15;

typedef PC1<NSObject *, NSString *><NSCopying> typeArgsAndProtocolQuals4;

typedef typeArgs15<NSCopying> typeArgsAndProtocolQuals5;

typedef typeArgs15<NSObject *, NSString *> typeArgs16; // expected-error{{type arguments cannot be applied to already-specialized class type 'typeArgs15' (aka 'PC1<NSObject *,NSString *>')}}

typedef typeArgs15<NSObject> typeArgsAndProtocolQuals6;

void testSpecializedTypePrinting(void) {
  int *ip;

  ip = (typeArgs15*)0; // expected-warning{{'typeArgs15 *' (aka 'PC1<NSObject *,NSString *> *')}}
  ip = (typeArgsAndProtocolQuals4*)0; // expected-warning{{'typeArgsAndProtocolQuals4 *' (aka 'PC1<NSObject *,NSString *><NSCopying> *')}}
  ip = (typeArgsAndProtocolQuals5*)0; // expected-warning{{'typeArgsAndProtocolQuals5 *' (aka 'typeArgs15<NSCopying> *')}}
  ip = (typeArgsAndProtocolQuals6)0; // expected-error{{used type 'typeArgsAndProtocolQuals6' (aka 'typeArgs15<NSObject>')}}
  ip = (typeArgsAndProtocolQuals6*)0;// expected-warning{{'typeArgsAndProtocolQuals6 *' (aka 'typeArgs15<NSObject> *')}}
}

// --------------------------------------------------------------------------
// Specialized superclasses
// --------------------------------------------------------------------------
@interface PC21<T : NSObject *> : PC1<T, T>
@end

@interface PC22<T : NSObject *> : PC1<T> // expected-error{{too few type arguments for class 'PC1' (have 1, expected 2)}}
@end

@interface PC23<T : NSObject *> : PC1<T, U> // expected-error{{unknown type name 'U'}}
@end

@interface PC24<T> : PC1<T, T> // expected-error{{type argument 'T' (aka 'id') does not satisfy the bound ('NSObject *') of type parameter 'U'}}
@end

@interface NSFoo : PC1<NSObject *, NSObject *> // okay
@end

// --------------------------------------------------------------------------
// Co- and contra-variance.
// --------------------------------------------------------------------------
@class Variance1<T, U>;

@class Variance1<__covariant T, __contravariant U>;

@interface Variance1<__covariant T, __contravariant U> : NSObject // expected-note 2{{declared here}}
@end

@interface Variance1<T, U> () // okay, inferred
@end

@interface Variance1<T, U> (Cat1) // okay, inferred
@end

@class Variance1<T, U>; // okay, inferred

@interface Variance1<__covariant T, __contravariant U> () // okay, matches
@end

@interface Variance1<__covariant T, __contravariant U> (Cat2) // okay, matches
@end

@class Variance1<__covariant T, __contravariant U>; // okay, matches

@interface Variance1<__contravariant X, // expected-error{{contravariant type parameter 'X' conflicts with previous covariant type parameter 'T'}}
                     __covariant Y> () // expected-error{{covariant type parameter 'Y' conflicts with previous contravariant type parameter 'U'}}
@end

@class Variance2<__covariant T, __contravariant U>; // expected-note 2{{declared here}}

@interface Variance2<__contravariant T, // expected-error{{contravariant type parameter 'T' conflicts with previous covariant type parameter 'T'}}
                     U> : NSObject // expected-error{{invariant type parameter 'U' conflicts with previous contravariant type parameter 'U'}}
@end
