// RUN: %clang_cc1 -fblocks -fsyntax-only %s -verify -Wmethod-signatures

// Tests Objective-C 'kindof' types.

#if !__has_feature(objc_kindof)
#error does not support __kindof
#endif

@protocol NSObject
@end

@protocol NSCopying
- (id)copy;
+ (Class)classCopy;
@end

@protocol NSRandomProto
- (void)randomMethod;
+ (void)randomClassMethod;
@end

__attribute__((objc_root_class))
@interface NSObject <NSObject>
- (NSObject *)retain;
@end

@interface NSString : NSObject <NSCopying> // expected-note{{receiver is instance of class declared here}}
- (void)compare:(NSString *)string;
- (NSString *)stringByAppendingString:(NSString *)string;
+ (instancetype)string;
@end

@interface NSMutableString : NSString
- (void)appendString:(NSString *)string;
@end

@interface NSNumber : NSObject <NSCopying>
- (NSNumber *)numberByAddingNumber:(NSNumber *)number;
@end

// ---------------------------------------------------------------------------
// Parsing and semantic analysis for __kindof
// ---------------------------------------------------------------------------

// Test proper application of __kindof.
typedef __kindof NSObject *typedef1;
typedef NSObject __kindof *typedef2;
typedef __kindof NSObject<NSCopying> typedef3;
typedef NSObject<NSCopying> __kindof *typedef4;
typedef __kindof id<NSCopying> typedef5;
typedef __kindof Class<NSCopying> typedef6;

// Test redundancy of __kindof.
typedef __kindof id __kindof redundant_typedef1;
typedef __kindof NSObject __kindof *redundant_typedef2;

// Test application of __kindof to typedefs.
typedef NSObject *NSObject_ptr_typedef;
typedef NSObject NSObject_typedef;
typedef __kindof NSObject_ptr_typedef typedef_typedef1;
typedef __kindof NSObject_typedef typedef_typedef2;

// Test application of __kindof to non-object types.
typedef __kindof int nonobject_typedef1; // expected-error{{'__kindof' specifier cannot be applied to non-object type 'int'}}
typedef NSObject **NSObject_ptr_ptr;
typedef __kindof NSObject_ptr_ptr nonobject_typedef2; // expected-error{{'__kindof' specifier cannot be applied to non-object type 'NSObject_ptr_ptr' (aka 'NSObject **')}}

// Test application of __kindof outside of the decl-specifiers.
typedef NSObject * __kindof bad_specifier_location1; // expected-error{{'__kindof' type specifier must precede the declarator}}
typedef NSObject bad_specifier_location2 __kindof; // expected-error{{expected ';' after top level declarator}}
// expected-warning@-1{{declaration does not declare anything}}

// ---------------------------------------------------------------------------
// Pretty printing of __kindof
// ---------------------------------------------------------------------------
void test_pretty_print(int *ip) {
  __kindof NSObject *kindof_NSObject;
  ip = kindof_NSObject; // expected-warning{{from '__kindof NSObject *'}}
 
  __kindof NSObject_ptr_typedef kindof_NSObject_ptr;
  ip = kindof_NSObject_ptr; // expected-warning{{from '__kindof NSObject_ptr_typedef'}}

  __kindof id <NSCopying> *kindof_NSCopying;
  ip = kindof_NSCopying; // expected-warning{{from '__kindof id<NSCopying> *'}}

  __kindof NSObject_ptr_typedef *kindof_NSObject_ptr_typedef;
  ip = kindof_NSObject_ptr_typedef; // expected-warning{{from '__kindof NSObject_ptr_typedef *'}}
}

// ---------------------------------------------------------------------------
// Basic implicit conversions (dropping __kindof, upcasts, etc.)
// ---------------------------------------------------------------------------
void test_add_remove_kindof_conversions(void) {
  __kindof NSObject *kindof_NSObject_obj;
  NSObject *NSObject_obj;

  // Conversion back and forth
  kindof_NSObject_obj = NSObject_obj;
  NSObject_obj = kindof_NSObject_obj;

  // Qualified-id conversion back and forth.
  __kindof id <NSCopying> kindof_id_NSCopying_obj;
  id <NSCopying> id_NSCopying_obj;
  kindof_id_NSCopying_obj = id_NSCopying_obj;
  id_NSCopying_obj = kindof_id_NSCopying_obj;
}

void test_upcast_conversions(void) {
  __kindof NSObject *kindof_NSObject_obj;
  NSObject *NSObject_obj;

  // Upcasts
  __kindof NSString *kindof_NSString_obj;
  NSString *NSString_obj;
  kindof_NSObject_obj = kindof_NSString_obj;
  kindof_NSObject_obj = NSString_obj;
  NSObject_obj = kindof_NSString_obj;
  NSObject_obj = NSString_obj;

  // "Upcasts" with qualified-id.
  __kindof id <NSCopying> kindof_id_NSCopying_obj;
  id <NSCopying> id_NSCopying_obj;
  kindof_id_NSCopying_obj = kindof_NSString_obj;
  kindof_id_NSCopying_obj = NSString_obj;
  id_NSCopying_obj = kindof_NSString_obj;
  id_NSCopying_obj = NSString_obj;
}


void test_ptr_object_conversions(void) {
  __kindof NSObject **ptr_kindof_NSObject_obj;
  NSObject **ptr_NSObject_obj;

  // Conversions back and forth.
  ptr_kindof_NSObject_obj = ptr_NSObject_obj;
  ptr_NSObject_obj = ptr_kindof_NSObject_obj;

  // Conversions back and forth with qualified-id.
  __kindof id <NSCopying> *ptr_kindof_id_NSCopying_obj;
  id <NSCopying> *ptr_id_NSCopying_obj;
  ptr_kindof_id_NSCopying_obj = ptr_id_NSCopying_obj;
  ptr_id_NSCopying_obj = ptr_kindof_id_NSCopying_obj;

  // Upcasts.
  __kindof NSString **ptr_kindof_NSString_obj;
  NSString **ptr_NSString_obj;
  ptr_kindof_NSObject_obj = ptr_kindof_NSString_obj;
  ptr_kindof_NSObject_obj = ptr_NSString_obj;
  ptr_NSObject_obj = ptr_kindof_NSString_obj;
  ptr_NSObject_obj = ptr_NSString_obj;
}

// ---------------------------------------------------------------------------
// Implicit downcasting
// ---------------------------------------------------------------------------
void test_downcast_conversions(void) {
  __kindof NSObject *kindof_NSObject_obj;
  NSObject *NSObject_obj;
  __kindof NSString *kindof_NSString_obj;
  NSString *NSString_obj;

  // Implicit downcasting.
  kindof_NSString_obj = kindof_NSObject_obj;
  kindof_NSString_obj = NSObject_obj; // expected-warning{{assigning to '__kindof NSString *' from 'NSObject *'}}
  NSString_obj = kindof_NSObject_obj;
  NSString_obj = NSObject_obj; // expected-warning{{assigning to 'NSString *' from 'NSObject *'}}

  // Implicit downcasting with qualified id.
  __kindof id <NSCopying> kindof_NSCopying_obj;
  id <NSCopying> NSCopying_obj;
  kindof_NSString_obj = kindof_NSCopying_obj;
  kindof_NSString_obj = NSCopying_obj; // expected-warning{{from incompatible type 'id<NSCopying>'}}
  NSString_obj = kindof_NSCopying_obj;
  NSString_obj = NSCopying_obj; // expected-warning{{from incompatible type 'id<NSCopying>'}}
  kindof_NSObject_obj = kindof_NSCopying_obj;
  kindof_NSObject_obj = NSCopying_obj; // expected-warning{{from incompatible type 'id<NSCopying>'}}
  NSObject_obj = kindof_NSCopying_obj;
  NSObject_obj = NSCopying_obj; // expected-warning{{from incompatible type 'id<NSCopying>'}}
}

void test_crosscast_conversions(void) {
  __kindof NSString *kindof_NSString_obj;
  NSString *NSString_obj;
  __kindof NSNumber *kindof_NSNumber_obj;
  NSNumber *NSNumber_obj;

  NSString_obj = kindof_NSNumber_obj; // expected-warning{{from '__kindof NSNumber *'}}
}

@interface NSCell : NSObject
@end
@interface NSCellSub : NSCell
@end
@interface NSCellSub2 : NSCell
@end
@interface NSCellSubSub : NSCellSub
@end

typedef signed char BOOL;
void test_conditional(BOOL flag) {
  NSCellSubSub *result;
  __kindof NSCellSub *kindof_Sub;
  NSCell *cell;
  NSCellSub *sub;
  NSCellSub2 *sub2;
  NSCellSubSub *subsub;

  // LHS is kindof NSCellSub, RHS is NSCell --> kindof NSCell
  // LHS is kindof NSCellSub, RHS is NSCellSub --> kindof NSCellSub
  // LHS is kindof NSCellSub, RHS is NSCellSub2 --> kindof NSCell
  // LHS is kindof NSCellSub, RHS is NSCellSubSub --> kindof NSCellSub
  result = flag ? kindof_Sub : cell;
  result = flag ? kindof_Sub : sub;
  result = flag ? kindof_Sub : sub2;
  result = flag ? kindof_Sub : subsub;

  result = flag ? cell : kindof_Sub;
  result = flag ? sub : kindof_Sub;
  result = flag ? sub2 : kindof_Sub;
  result = flag ? subsub : kindof_Sub;
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------
void test_block_conversions(void) {
  // Adding/removing __kindof from return type.
  __kindof NSString *(^kindof_NSString_void_block)(void);
  NSString *(^NSString_void_block)(void);
  kindof_NSString_void_block = NSString_void_block;
  NSString_void_block = kindof_NSString_void_block;

  // Covariant return type.
  __kindof NSMutableString *(^kindof_NSMutableString_void_block)(void);
  NSMutableString *(^NSMutableString_void_block)(void);
  kindof_NSString_void_block = NSMutableString_void_block;
  NSString_void_block = kindof_NSMutableString_void_block;
  kindof_NSString_void_block = NSMutableString_void_block;
  NSString_void_block = kindof_NSMutableString_void_block;

  // "Covariant" return type via downcasting rule.
  kindof_NSMutableString_void_block = NSString_void_block; // expected-error{{from 'NSString *(^)(void)'}}
  NSMutableString_void_block = kindof_NSString_void_block;
  kindof_NSMutableString_void_block = NSString_void_block; // expected-error{{from 'NSString *(^)(void)'}}
  NSMutableString_void_block = kindof_NSString_void_block;

  // Cross-casted return type.
  __kindof NSNumber *(^kindof_NSNumber_void_block)(void);
  NSNumber *(^NSNumber_void_block)(void);
  kindof_NSString_void_block = NSNumber_void_block; // expected-error{{from 'NSNumber *(^)(void)'}}
  NSString_void_block = kindof_NSNumber_void_block; // expected-error{{'__kindof NSNumber *(^)(void)'}}
  kindof_NSString_void_block = NSNumber_void_block; // expected-error{{from 'NSNumber *(^)(void)'}}
  NSString_void_block = kindof_NSNumber_void_block; // expected-error{{'__kindof NSNumber *(^)(void)'}}

  // Adding/removing __kindof from argument type.
  void (^void_kindof_NSString_block)(__kindof NSString *);
  void (^void_NSString_block)(NSString *);
  void_kindof_NSString_block = void_NSString_block;
  void_NSString_block = void_kindof_NSString_block;

  // Contravariant argument type.
  void (^void_kindof_NSMutableString_block)(__kindof NSMutableString *);
  void (^void_NSMutableString_block)(NSMutableString *);
  void_kindof_NSMutableString_block = void_kindof_NSString_block;
  void_kindof_NSMutableString_block = void_NSString_block;
  void_NSMutableString_block = void_kindof_NSString_block;
  void_NSMutableString_block = void_NSString_block;

  // "Contravariant" argument type via downcasting rule.
  void_kindof_NSString_block = void_kindof_NSMutableString_block;
  void_kindof_NSString_block = void_NSMutableString_block;
  void_NSString_block = void_kindof_NSMutableString_block; // expected-error{{from 'void (^)(__kindof NSMutableString *)'}}
  void_NSString_block = void_NSMutableString_block; // expected-error{{from 'void (^)(NSMutableString *)'}}
}

// ---------------------------------------------------------------------------
// Messaging __kindof types.
// ---------------------------------------------------------------------------
void message_kindof_object(__kindof NSString *kindof_NSString) {
  [kindof_NSString retain]; // in superclass
  [kindof_NSString stringByAppendingString:0]; // in class
  [kindof_NSString appendString:0]; // in subclass
  [kindof_NSString numberByAddingNumber: 0]; // expected-warning{{instance method '-numberByAddingNumber:' not found (return type defaults to 'id')}}
  [kindof_NSString randomMethod]; // in protocol
}

void message_kindof_qualified_id(__kindof id <NSCopying> kindof_NSCopying) {
  [kindof_NSCopying copy]; // in protocol
  [kindof_NSCopying stringByAppendingString:0]; // in some class
  [kindof_NSCopying randomMethod]; // in unrelated protocol
}

void message_kindof_qualified_class(
       __kindof Class <NSCopying> kindof_NSCopying) {
  [kindof_NSCopying classCopy]; // in protocol
  [kindof_NSCopying string]; // in some class
  [kindof_NSCopying randomClassMethod]; // in unrelated protocol
}

// Make sure we don't emit warning about multiple methods found.
typedef int NSInteger;
@interface Foo : NSObject
- (NSString*)test;
@end
@interface Bar : NSObject
- (NSInteger)test;
@end
void test(__kindof Bar *kBar) {
    [kBar test];
}

// Make sure we don't emit warning about no method found.
@interface A : NSObject
@property (readonly, getter=isActive) BOOL active;
@end
@interface B : NSObject
@property (getter=isActive, readonly) BOOL active;
@end
void foo() {
  __kindof B *NSApp;
  if ([NSApp isActive]) {
  }
}

typedef const struct CGPath *CGPathRef;
@interface C : NSObject
@property (copy) NSString *path;
@end
@interface D : NSObject
@property CGPathRef path __attribute__((availability(macosx,unavailable)));
@end
// Make sure we choose "NSString *path" for [s1 path].
void bar(id s1, id s2) {
  return [[s1 path] compare:[s2 path]];
}

// ---------------------------------------------------------------------------
// __kindof within specialized types
// ---------------------------------------------------------------------------
@interface NSArray<T> : NSObject
@end

void implicit_convert_array(NSArray<__kindof NSString *> *kindofStringsArray,
                            NSArray<NSString *> *stringsArray,
                            NSArray<__kindof NSMutableString *>
                              *kindofMutStringsArray,
                            NSArray<NSMutableString *> *mutStringsArray) {
  // Adding/removing __kindof is okay.
  kindofStringsArray = stringsArray;
  stringsArray = kindofStringsArray;

  // Other covariant and contravariant conversions still not permitted.
  kindofStringsArray = mutStringsArray; // expected-warning{{incompatible pointer types}}
  stringsArray = kindofMutStringsArray; // expected-warning{{incompatible pointer types}}
  mutStringsArray = kindofStringsArray; // expected-warning{{incompatible pointer types}}

  // Adding/removing nested __kindof is okay.
  NSArray<NSArray<__kindof NSString *> *> *kindofStringsArrayArray;
  NSArray<NSArray<NSString *> *> *stringsArrayArray;
  kindofStringsArrayArray = stringsArrayArray;
  stringsArrayArray = kindofStringsArrayArray;
}

// ---------------------------------------------------------------------------
// __kindof + nullability
// ---------------------------------------------------------------------------

void testNullability() {
  // The base type being a pointer type tickles the bug.
  extern __kindof id <NSCopying> _Nonnull getSomeCopyable();
  NSString *string = getSomeCopyable(); // no-warning

  void processCopyable(__typeof(getSomeCopyable()) string);
  processCopyable(0); // expected-warning{{null passed to a callee that requires a non-null argument}}
}

// Make sure that we don't emit a warning about conflicting parameter types
// between __kindof id and id.
@interface A2 : NSObject
- (void)test:(__kindof id)T;
@end
@implementation A2
- (void)test:(id)T {
}
@end

@interface NSGeneric<ObjectType> : NSObject
- (void)test:(__kindof ObjectType)T;
- (void)mapUsingBlock:(id (^)(__kindof ObjectType))block;
@end
@implementation NSGeneric
- (void)test:(id)T {
}
- (void)mapUsingBlock:(id (^)(id))block {
}
@end

// Check that clang doesn't crash when a type parameter is illegal.
@interface Array1<T> : NSObject
@end

@interface I1 : NSObject
@end

@interface Array1<__kindof I1*>(extensions1) // expected-error{{expected type parameter name}}
@end

@interface Array2<T1, T2, T3> : NSObject
@end

@interface Array2<T, T, __kindof I1*>(extensions2) // expected-error{{expected type parameter name}} expected-error{{redeclaration of type parameter 'T'}}
@end
