// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.ObjCGenerics,alpha.core.DynamicTypeChecker -verify -Wno-objc-method-access %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.ObjCGenerics,alpha.core.DynamicTypeChecker -verify -Wno-objc-method-access %s -analyzer-output=plist -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/generics.m.plist -

#if !__has_feature(objc_generics)
#  error Compiler does not support Objective-C generics?
#endif

#if !__has_feature(objc_generics_variance)
#  error Compiler does not support co- and contr-variance?
#endif

#define nil 0
typedef unsigned long NSUInteger;
typedef int BOOL;

@protocol NSObject
+ (id)alloc;
- (id)init;
@end

@protocol NSCopying
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
@end

@interface NSString : NSObject <NSCopying>
@end

@interface NSMutableString : NSString
@end

@interface NSNumber : NSObject <NSCopying>
@end

@interface NSSet : NSObject <NSCopying>
@end

@interface NSArray<__covariant ObjectType> : NSObject
+ (instancetype)arrayWithObjects:(const ObjectType [])objects count:(NSUInteger)count;
+ (instancetype)getEmpty;
+ (NSArray<ObjectType> *)getEmpty2;
- (BOOL)contains:(ObjectType)obj;
- (BOOL)containsObject:(ObjectType)anObject;
- (ObjectType)getObjAtIndex:(NSUInteger)idx;
- (ObjectType)objectAtIndexedSubscript:(NSUInteger)idx;
- (NSArray<ObjectType> *)arrayByAddingObject:(ObjectType)anObject;
@property(readonly) ObjectType firstObject;
@end

@interface NSMutableArray<ObjectType> : NSArray<ObjectType>
- (void)addObject:(ObjectType)anObject;
- (instancetype)init;
@end

@interface MutableArray<ObjectType> : NSArray<ObjectType>
- (void)addObject:(ObjectType)anObject;
@end

@interface LegacyMutableArray : MutableArray
@end

@interface LegacySpecialMutableArray : LegacyMutableArray
@end

@interface BuggyMutableArray<T> : MutableArray
@end

@interface BuggySpecialMutableArray<T> : BuggyMutableArray<T>
@end

@interface MyMutableStringArray : MutableArray<NSString *>
@end

@interface ExceptionalArray<ExceptionType> : MutableArray<NSString *>
- (ExceptionType) getException;
@end

@interface UnrelatedType : NSObject<NSCopying>
@end

int getUnknown();
NSArray *getStuff();
NSArray *getTypedStuff() {
  NSArray<NSNumber *> *c = getStuff();
  return c;
}

void doStuff(NSArray<NSNumber *> *);
void withArrString(NSArray<NSString *> *);
void withArrMutableString(NSArray<NSMutableString *> *);
void withMutArrString(MutableArray<NSString *> *);
void withMutArrMutableString(MutableArray<NSMutableString *> *);

void incompatibleTypesErased(NSArray *a, NSMutableArray<NSString *> *b,
                             NSArray<NSNumber *> *c,
                             NSMutableArray *d) {
  a = b;
  c = a; // expected-warning  {{Conversion from value of type 'NSMutableArray<NSString *> *' to incompatible type 'NSArray<NSNumber *> *'}}
  [a contains: [[NSNumber alloc] init]];
  [a contains: [[NSString alloc] init]];
  doStuff(a); // expected-warning {{Conversion}}

  d = b;
  [d addObject: [[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void crossProceduralErasedTypes() {
  NSArray<NSString *> *a = getTypedStuff(); // expected-warning {{Conversion}}
}

void incompatibleTypesErasedReverseConversion(NSMutableArray *a,
                                              NSMutableArray<NSString *> *b) {
  b = a;
  [a contains: [[NSNumber alloc] init]];
  [a contains: [[NSString alloc] init]];
  doStuff(a); // expected-warning {{Conversion}}

  [a addObject: [[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void idErasedIncompatibleTypesReverseConversion(id a, NSMutableArray<NSString *> *b) {
  b = a;
  [a contains: [[NSNumber alloc] init]];
  [a contains: [[NSString alloc] init]];
  doStuff(a); // expected-warning {{Conversion}}

  [a addObject:[[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void idErasedIncompatibleTypes(id a, NSMutableArray<NSString *> *b,
                               NSArray<NSNumber *> *c) {
  a = b;
  c = a; // expected-warning {{Conversion}}
  [a contains: [[NSNumber alloc] init]];
  [a contains: [[NSString alloc] init]];
  doStuff(a); // expected-warning {{Conversion}}

  [a addObject:[[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void pathSensitiveInference(MutableArray *m, MutableArray<NSString *> *a,
                            MutableArray<NSMutableString *> *b) {
  if (getUnknown() == 5) {
    m = a;  
    [m contains: [[NSString alloc] init]];
  } else {
    m = b;
    [m contains: [[NSMutableString alloc] init]];
  }
  [m addObject: [[NSString alloc] init]]; // expected-warning {{Conversion}}
  [m addObject: [[NSMutableString alloc] init]];
}

void verifyAPIusage(id a, MutableArray<NSString *> *b) {
  b = a;
  doStuff(a); // expected-warning {{Conversion}}
}

void dontInferFromExplicitCastsOnUnspecialized(MutableArray *a,
                        MutableArray<NSMutableString *> *b) {
  b = (MutableArray<NSMutableString *> *)a;
  [a addObject: [[NSString alloc] init]]; // no-warning
}

void dontWarnOnExplicitCastsAfterInference(MutableArray *a) {
  withMutArrString(a);
  withMutArrMutableString((MutableArray<NSMutableString *> *)a); // no-warning
}

void dontDiagnoseOnExplicitCrossCasts(MutableArray<NSSet *> *a,
                        MutableArray<NSMutableString *> *b) {
  // Treat an explicit cast to a specialized type as an indication that
  // Objective-C's type system is not expressive enough to represent a
  // the invariant the programmer wanted. After an explicit cast, do not
  // warn about potential generics shenanigans.
  b = (MutableArray<NSMutableString *> *)a; // no-warning
  [a addObject: [[NSSet alloc] init]]; // no-warning
  [b addObject: [[NSMutableString alloc] init]]; //no-warning
}

void subtypeOfGeneric(id d, MyMutableStringArray *a,
                       MutableArray<NSString *> *b,
                       MutableArray<NSNumber *> *c) {
  d = a;
  b = d;
  c = d; // expected-warning {{Conversion}}
}

void genericSubtypeOfGeneric(id d, ExceptionalArray<NSString *> *a,
                             MutableArray<NSString *> *b,
                             MutableArray<NSNumber *> *c) {
  d = a;
  [d contains: [[NSString alloc] init]];
  [d contains: [[NSNumber alloc] init]];
  b = d;
  c = d; // expected-warning {{Conversion}}

  [d addObject: [[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void genericSubtypeOfGenericReverse(id d, ExceptionalArray<NSString *> *a,
                                    MutableArray<NSString *> *b,
                                    MutableArray<NSNumber *> *c) {
  a = d;
  [d contains: [[NSString alloc] init]];
  [d contains: [[NSNumber alloc] init]];
  b = d;
  c = d; // expected-warning {{Conversion}}

 [d addObject: [[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void inferenceFromAPI(id a) {
  // Here the type parameter is invariant. There should be a warning every time
  // when the type parameter changes during the conversions.
  withMutArrString(a);
  withMutArrMutableString(a); // expected-warning {{Conversion}}
}

void inferenceFromAPI2(id a) {
  withMutArrMutableString(a);
  withMutArrString(a); // expected-warning {{Conversion}}
}

void inferenceFromAPIWithLegacyTypes(LegacyMutableArray *a) {
  withMutArrMutableString(a);
  withMutArrString(a); // expected-warning {{Conversion}}
}

void inferenceFromAPIWithLegacyTypes2(LegacySpecialMutableArray *a) {
  withMutArrString(a);
  withMutArrMutableString(a); // expected-warning {{Conversion}}
}

void inferenceFromAPIWithLegacyTypes3(__kindof NSArray<NSString *> *a) {
  LegacyMutableArray *b = a;
  withMutArrString(b);
  withMutArrMutableString(b); // expected-warning {{Conversion}}
}

void inferenceFromAPIWithBuggyTypes(BuggyMutableArray<NSMutableString *> *a) {
  withMutArrString(a);
  withMutArrMutableString(a); // expected-warning {{Conversion}}
}

void InferenceFromAPIWithBuggyTypes2(BuggySpecialMutableArray<NSMutableString *> *a) {
  withMutArrMutableString(a);
  withMutArrString(a); // expected-warning {{Conversion}}
}

void InferenceFromAPIWithBuggyTypes3(MutableArray<NSMutableString *> *a) {
  id b = a;
  withMutArrMutableString((BuggyMutableArray<NSMutableString *> *)b);
  withMutArrString(b); // expected-warning {{Conversion}}
}

void InferenceFromAPIWithBuggyTypes4(__kindof NSArray<NSString *> *a) {
  BuggyMutableArray<NSMutableString *> *b = a;
  withMutArrString(b);
  withMutArrMutableString(b); // expected-warning {{Conversion}}
}

NSArray<NSString *> *getStrings();
void enforceDynamicRulesInsteadOfStatic(NSArray<NSNumber *> *a) {
  NSArray *b = a;
  // Valid uses of NSArray of NSNumbers.
  b = getStrings();
  // Valid uses of NSArray of NSStrings.
}

void workWithProperties(NSArray<NSNumber *> *a) {
  NSArray *b = a;
  NSString *str = [b getObjAtIndex: 0]; // expected-warning {{Object has a dynamic type 'NSNumber *' which is incompatible with static type 'NSString *'}}
  NSNumber *num = [b getObjAtIndex: 0];
  str = [b firstObject]; // expected-warning {{Object has a dynamic type 'NSNumber *' which is incompatible with static type 'NSString *'}}
  num = [b firstObject];
  str = b.firstObject; // expected-warning {{Object has a dynamic type 'NSNumber *' which is incompatible with static type 'NSString *'}}
  num = b.firstObject;
  str = b[0]; // expected-warning {{Object has a dynamic type 'NSNumber *' which is incompatible with static type 'NSString *'}}
  num = b[0];
}

void findMethodDeclInTrackedType(id m, NSArray<NSMutableString *> *a,
                                 MutableArray<NSMutableString *> *b) {
  a = b;
  if (getUnknown() == 5) {
    m = a;  
    [m addObject: [[NSString alloc] init]]; // expected-warning {{Conversion}}
  } else {
    m = b;
    [m addObject: [[NSMutableString alloc] init]];
  }
}

void findMethodDeclInTrackedType2(__kindof NSArray<NSString *> *a,
                                  MutableArray<NSMutableString *> *b) {
  a = b;
  if (getUnknown() == 5) {
    [a addObject: [[NSString alloc] init]]; // expected-warning {{Conversion}}
  } else {
    [a addObject: [[NSMutableString alloc] init]];
  }
}

void testUnannotatedLiterals() {
  // ObjCArrayLiterals are not specialized in the AST. 
  NSArray *arr = @[@"A", @"B"];
  [arr contains: [[NSNumber alloc] init]];
}

void testAnnotatedLiterals() {
  NSArray<NSString *> *arr = @[@"A", @"B"];
  NSArray *arr2 = arr;
  [arr2 contains: [[NSNumber alloc] init]];
}

void nonExistentMethodDoesNotCrash(id a, MutableArray<NSMutableString *> *b) {
  a = b;
  [a nonExistentMethod];
}

void trackedClassVariables() {
  Class c = [NSArray<NSString *> class];
  NSArray<NSNumber *> *a = [c getEmpty]; // expected-warning {{Conversion}}
  a = [c getEmpty2]; // expected-warning {{Conversion}}
}

void nestedCollections(NSArray<NSArray<NSNumber *> *> *mat, NSArray<NSString *> *row) {
  id temp = row;
  [mat contains: temp]; // expected-warning {{Conversion}}
}

void testMistmatchedTypeCast(MutableArray<NSMutableString *> *a) {
  MutableArray *b = (MutableArray<NSNumber *> *)a;
  [b addObject: [[NSNumber alloc] init]];
  id c = (UnrelatedType *)a;
  [c addObject: [[NSNumber alloc] init]];
  [c addObject: [[NSString alloc] init]];
}

void returnCollectionToIdVariable(NSArray<NSArray<NSString *> *> *arr) {
  NSArray *erased = arr;
  id a = [erased firstObject];
  NSArray<NSNumber *> *res = a; // expected-warning {{Conversion}}
}

void eraseSpecialization(NSArray<NSArray<NSString *> *> *arr) {
  NSArray *erased = arr;
  NSArray* a = [erased firstObject];
  NSArray<NSNumber *> *res = a; // expected-warning {{Conversion}}
}

void returnToUnrelatedType(NSArray<NSArray<NSString *> *> *arr) {
  NSArray *erased = arr;
  NSSet* a = [erased firstObject]; // expected-warning {{Object has a dynamic type 'NSArray<NSString *> *' which is incompatible with static type 'NSSet *'}}
  (void)a;
}

void returnToIdVariable(NSArray<NSString *> *arr) {
  NSArray *erased = arr;
  id a = [erased firstObject];
  NSNumber *res = a; // expected-warning {{Object has a dynamic type 'NSString *' which is incompatible with static type 'NSNumber *'}}
}

@interface UnrelatedTypeGeneric<T> : NSObject<NSCopying>
- (void)takesType:(T)v;
@end

void testGetMostInformativeDerivedForId(NSArray<NSString *> *a,
                                  UnrelatedTypeGeneric<NSString *> *b) {
  id idB = b;
  a = idB; // expected-warning {{Conversion from value of type 'UnrelatedTypeGeneric<NSString *> *' to incompatible type 'NSArray<NSString *> *'}}

  // rdar://problem/26086914 crash here caused by symbolic type being unrelated
  // to compile-time source type of cast.
  id x = a; // Compile-time type is NSArray<>, Symbolic type is UnrelatedTypeGeneric<>.
  [x takesType:[[NSNumber alloc] init]]; // expected-warning {{Conversion from value of type 'NSNumber *' to incompatible type 'NSString *'}}
}

void testArgumentAfterUpcastToRootWithCovariantTypeParameter(NSArray<NSString *> *allStrings, NSNumber *number) {
  NSArray<NSObject *> *allObjects = allStrings; // no-warning
  NSArray<NSObject *> *moreObjects = [allObjects arrayByAddingObject:number]; // no-warning
}

void testArgumentAfterUpcastWithCovariantTypeParameter(NSArray<NSMutableString *> *allMutableStrings, NSNumber *number) {
  NSArray<NSString *> *allStrings = allMutableStrings; // no-warning
  id numberAsId = number;
  NSArray<NSString *> *moreStrings = [allStrings arrayByAddingObject:numberAsId]; // Sema: expected-warning {{Object has a dynamic type 'NSNumber *' which is incompatible with static type 'NSString *'}}
}

void testArgumentAfterCastToUnspecializedWithCovariantTypeParameter(NSArray<NSMutableString *> *allMutableStrings, NSNumber *number) {
  NSArray *allStrings = allMutableStrings; // no-warning
  id numberAsId = number;

  NSArray *moreStringsUnspecialized = [allStrings arrayByAddingObject:numberAsId]; // no-warning

  // Ideally the analyzer would warn here.
  NSArray<NSString *> *moreStringsSpecialized = [allStrings arrayByAddingObject:numberAsId];
}

void testCallToMethodWithCovariantParameterOnInstanceOfSubclassWithInvariantParameter(NSMutableArray<NSMutableString *> *mutableArrayOfMutableStrings, NSString *someString) {
  NSArray<NSString *> *arrayOfStrings = mutableArrayOfMutableStrings;
  [arrayOfStrings containsObject:someString]; // no-warning
}

