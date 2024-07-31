// RUN: %clang_cc1 -fsyntax-only -verify -Wattributes -fpascal-strings %s

typedef long NSInteger;
typedef unsigned long NSUInteger;
typedef signed char BOOL;

@interface NSNumber
@end
@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
+ (NSNumber *)numberWithInteger:(NSInteger)value;
+ (NSNumber *)numberWithUnsignedInteger:(NSUInteger)value;
@end

typedef struct {
    int x, y, z;
} point;

void testStruct(void) {
    point p = { 0, 0, 0 };
    id boxed = @(p);    // expected-error {{illegal type 'point' used in a boxed expression}}
}

void testPointers(void) {
    void *null = 0;
    id boxed_null = @(null);        // expected-error {{illegal type 'void *' used in a boxed expression}}
    int numbers[] = { 0, 1, 2 };
    id boxed_numbers = @(numbers);  // expected-error {{illegal type 'int *' used in a boxed expression}}
}

void testInvalid(void) {
  @(not_defined); // expected-error {{use of undeclared identifier 'not_defined'}}
}

enum MyEnum {
  ME_foo
};

enum ForwE;

void testEnum(void *p) {
  enum MyEnum myen;
  id box = @(myen);
  box = @(ME_foo);
  box = @(*(enum ForwE*)p); // expected-error {{incomplete type 'enum ForwE' used in a boxed expression}}
}

@interface NSString
@end

void testStringLiteral(void) {
  NSString *s;
  s = @("abc");
  s = @(u8"abc");
  s = @(u"abc"); // expected-error {{illegal type 'unsigned short *' used in a boxed expression}}
  s = @(U"abc"); // expected-error {{illegal type 'unsigned int *' used in a boxed expression}}
  s = @(L"abc"); // expected-error-re {{illegal type {{.*}} used in a boxed expression}}
  s = @("\pabc"); // expected-error {{illegal type 'unsigned char *' used in a boxed expression}}
}

@class NSMutableDictionary;

@interface NSMutableArray
+ (NSMutableArray*) array;
@end

NSMutableDictionary* mBars;

__attribute((objc_root_class)) @interface rdar13333205 @end

@implementation rdar13333205
- (void) insertBar:(id)preset ofKind:(id) kind atIndex:(int)index {
  NSMutableArray* presetArray = mBars[kind] ?: [NSMutableArray array]; // expected-error {{expected method to read dictionary element not found on object of type 'NSMutableDictionary *'}}
}
@end
