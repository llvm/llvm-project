// RUN: %clang_cc1 -fsyntax-only -verify -Wattributes %s

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

void testStruct() {
    point p = { 0, 0, 0 };
    id boxed = @(p);    // expected-error {{illegal type 'point' used in a boxed expression}}
}

void testPointers() {
    void *null = 0;
    id boxed_null = @(null);        // expected-error {{illegal type 'void *' used in a boxed expression}}
    int numbers[] = { 0, 1, 2 };
    id boxed_numbers = @(numbers);  // expected-error {{illegal type 'int *' used in a boxed expression}}
}

void testInvalid() {
  @(not_defined); // expected-error {{use of undeclared identifier 'not_defined'}}
}

enum MyEnum {
  ME_foo
};

enum ForwE; // expected-error {{ISO C++ forbids forward references to 'enum' types}}

void testEnum(void *p) {
  enum MyEnum myen;
  id box = @(myen);
  box = @(ME_foo);
  box = @(*(enum ForwE*)p); // expected-error {{incomplete type 'enum ForwE' used in a boxed expression}}
}
