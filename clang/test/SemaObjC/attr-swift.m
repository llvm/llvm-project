// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

// --- swift_private ---

__attribute__((swift_private))
@protocol FooProto
@end

__attribute__((swift_private))
@interface Foo
@end

@interface Bar
@property id prop __attribute__((swift_private));
- (void)instMethod __attribute__((swift_private));
+ (instancetype)bar __attribute__((swift_private));
@end

void function(id) __attribute__((swift_private));

struct __attribute__((swift_private)) Point {
  int x;
  int y;
};

enum __attribute__((swift_private)) Colors {
  Red, Green, Blue
};

typedef struct {
  float x, y, z;
} Point3D __attribute__((swift_private));


// --- swift_name ---

__attribute__((swift_name("SNFooType")))
@protocol SNFoo
@end

__attribute__((swift_name("SNFooClass")))
@interface SNFoo <SNFoo>
- (instancetype)init __attribute__((swift_name("init()")));
- (instancetype)initWithValue:(int)value __attribute__((swift_name("fooWithValue(_:)")));

+ (void)refresh __attribute__((swift_name("refresh()")));

+ (instancetype)foo __attribute__((swift_name("foo()")));
+ (SNFoo *)fooWithValue:(int)value __attribute__((swift_name("foo(value:)")));
+ (SNFoo *)fooWithValue:(int)value value:(int)value2 __attribute__((swift_name("foo(value:extra:)")));
+ (SNFoo *)fooWithConvertingValue:(int)value value:(int)value2 __attribute__((swift_name("init(_:extra:)")));

+ (SNFoo *)fooWithOtherValue:(int)value __attribute__((swift_name("init"))); // expected-error {{parameter of 'swift_name' attribute must be a Swift function name string}}
+ (SNFoo *)fooWithAnotherValue:(int)value __attribute__((swift_name("foo()"))); // expected-warning {{too few parameters in 'swift_name' attribute (expected 1; got 0)}}
+ (SNFoo *)fooWithYetAnotherValue:(int)value __attribute__((swift_name("foo(value:extra:)"))); // expected-warning {{too many parameters in 'swift_name' attribute (expected 1; got 2)}}

+ (SNFoo *)fooAndReturnErrorCode:(int *)errorCode __attribute__((swift_name("foo()"))); // no-warning
+ (SNFoo *)fooWithValue:(int)value andReturnErrorCode:(int *)errorCode __attribute__((swift_name("foo(value:)"))); // no-warning
+ (SNFoo *)fooFromErrorCode:(const int *)errorCode __attribute__((swift_name("foo()"))); // expected-warning {{too few parameters in 'swift_name' attribute (expected 1; got 0)}}
+ (SNFoo *)fooWithValue:(int)value fromErrorCode:(const int *)errorCode __attribute__((swift_name("foo(value:)"))); // expected-warning {{too few parameters in 'swift_name' attribute (expected 2; got 1)}}
+ (SNFoo *)fooWithPointerA:(int *)value andReturnErrorCode:(int *)errorCode __attribute__((swift_name("foo()"))); // no-warning
+ (SNFoo *)fooWithPointerB:(int *)value andReturnErrorCode:(int *)errorCode __attribute__((swift_name("foo(pointer:)"))); // no-warning
+ (SNFoo *)fooWithPointerC:(int *)value andReturnErrorCode:(int *)errorCode __attribute__((swift_name("foo(pointer:errorCode:)"))); // no-warning
+ (SNFoo *)fooWithOtherFoo:(SNFoo *)other __attribute__((swift_name("foo()"))); // expected-warning {{too few parameters in 'swift_name' attribute (expected 1; got 0)}}

+ (instancetype)specialFoo __attribute__((swift_name("init(options:)")));
+ (instancetype)specialBar __attribute__((swift_name("init(options:extra:)"))); // expected-warning {{too many parameters in 'swift_name' attribute (expected 0; got 2)}}
+ (instancetype)specialBaz __attribute__((swift_name("init(_:)"))); // expected-warning {{too many parameters in 'swift_name' attribute (expected 0; got 1)}}
+ (instancetype)specialGarply __attribute__((swift_name("foo(options:)"))); // expected-warning {{too many parameters in 'swift_name' attribute (expected 0; got 1)}}

+ (instancetype)trailingParen __attribute__((swift_name("foo("))); // expected-error {{parameter of 'swift_name' attribute must be a Swift function name string}}
+ (instancetype)trailingColon:(int)value __attribute__((swift_name("foo(value)"))); // expected-error {{parameter of 'swift_name' attribute must be a Swift function name string}}
+ (instancetype)initialIgnore:(int)value __attribute__((swift_name("_(value:)"))); // expected-error {{parameter of 'swift_name' attribute must be a Swift function name string}}
+ (instancetype)middleOmitted:(int)value __attribute__((swift_name("foo(:)"))); // expected-error {{parameter of 'swift_name' attribute must be a Swift function name string}}

@property(strong) id someProp __attribute__((swift_name("prop")));
@end

enum __attribute__((swift_name("MoreColors"))) MoreColors {
  Cyan,
  Magenta,
  Yellow __attribute__((swift_name("RoseGold"))),
  Black __attribute__((swift_name("SpaceGrey()"))) // expected-error {{parameter of 'swift_name' attribute must be an ASCII identifier string}}
};

struct __attribute__((swift_name("FooStruct"))) BarStruct {
  int x, y, z __attribute__((swift_name("zed")));
};

int global_int __attribute__((swift_name("GlobalInt")));

void foo1(int i) __attribute__((swift_name("foo"))); // expected-error{{parameter of 'swift_name' attribute must be a Swift function name string}}
void foo2(int i) __attribute__((swift_name("foo()"))); // expected-warning{{too few parameters in 'swift_name' attribute (expected 1; got 0)}}
void foo2(int i) __attribute__((swift_name("foo(a:b:)"))); // expected-warning{{too many parameters in 'swift_name' attribute (expected 1; got 2)}}
void foo3(int i, int j) __attribute__((swift_name("fooWithX(_:y:)"))); // okay
void foo4(int i, int *error) __attribute__((swift_name("fooWithA(_:)"))); // okay

typedef int some_int_type __attribute__((swift_name("SomeInt")));

struct Point3D createPoint3D(float x, float y, float z) __attribute__((swift_name("Point3D.init(x:y:z:)")));
struct Point3D rotatePoint3D(Point3D point, float radians) __attribute__((swift_name("Point3D.rotate(self:radius:)")));
struct Point3D badRotatePoint3D(Point3D point, float radians) __attribute__((swift_name("Point3D.rotate(radius:)"))); // expected-warning {{too few parameters in 'swift_name' attribute (expected 2; got 1)}}

extern struct Point3D identityPoint __attribute__((swift_name("Point3D.identity")));

// Getters and setters.
float Point3DGetMagnitude(Point3D point) __attribute__((swift_name("getter:Point3D.magnitude(self:)")));

float Point3DGetRadius(Point3D point) __attribute__((swift_name("getter:Point3D.radius(self:)")));
void Point3DSetRadius(Point3D point, float radius) __attribute__((swift_name("setter:Point3D.radius(self:_:)")));

Point3D getCurrentPoint3D(void) __attribute__((swift_name("getter:currentPoint3D()")));

void setCurrentPoint3D(Point3D point) __attribute__((swift_name("setter:currentPoint3D(_:)")));

Point3D getLastPoint3D(void) __attribute__((swift_name("getter:lastPoint3D()")));

void setLastPoint3D(Point3D point) __attribute__((swift_name("setter:lastPoint3D(_:)")));

Point3D getZeroPoint() __attribute__((swift_name("getter:Point3D()")));
void setZeroPoint(Point3D point) __attribute__((swift_name("setter:Point3D(_:)")));

Point3D badGetter1(int x) __attribute__((swift_name("getter:bad1(_:))"))); // expected-error{{parameter of 'swift_name' attribute must be a Swift function name string}}
void badSetter1() __attribute__((swift_name("getter:bad1())"))); // expected-error{{parameter of 'swift_name' attribute must be a Swift function name string}}

Point3D badGetter2(Point3D point) __attribute__((swift_name("getter:bad2(_:))"))); // expected-error{{parameter of 'swift_name' attribute must be a Swift function name string}}

void badSetter2(Point3D point) __attribute__((swift_name("setter:bad2(self:))"))); // expected-error{{parameter of 'swift_name' attribute must be a Swift function name string}}

// --- swift_error ---

@class NSError;

typedef struct __attribute__((objc_bridge(NSError))) __CFError *CFErrorRef;

@interface Erroneous
- (_Bool) tom0: (NSError**) err __attribute__((swift_error(none)));
- (_Bool) tom1: (NSError**) err __attribute__((swift_error(nonnull_error)));
- (_Bool) tom2: (NSError**) err __attribute__((swift_error(null_result))); // expected-error {{'swift_error' attribute with 'null_result' convention can only be applied to a method returning a pointer}}
- (_Bool) tom3: (NSError**) err __attribute__((swift_error(nonzero_result)));
- (_Bool) tom4: (NSError**) err __attribute__((swift_error(zero_result)));

- (Undeclared) richard0: (NSError**) err __attribute__((swift_error(none))); // expected-error {{expected a type}}
- (Undeclared) richard1: (NSError**) err __attribute__((swift_error(nonnull_error))); // expected-error {{expected a type}}
- (Undeclared) richard2: (NSError**) err __attribute__((swift_error(null_result))); // expected-error {{expected a type}}
// FIXME: the follow-on warnings should really be suppressed, but apparently having an ill-formed return type doesn't mark anything as invalid
- (Undeclared) richard3: (NSError**) err __attribute__((swift_error(nonzero_result))); // expected-error {{expected a type}} expected-error {{can only be applied}}
- (Undeclared) richard4: (NSError**) err __attribute__((swift_error(zero_result))); // expected-error {{expected a type}} expected-error {{can only be applied}}

- (instancetype) harry0: (NSError**) err __attribute__((swift_error(none)));
- (instancetype) harry1: (NSError**) err __attribute__((swift_error(nonnull_error)));
- (instancetype) harry2: (NSError**) err __attribute__((swift_error(null_result)));
- (instancetype) harry3: (NSError**) err __attribute__((swift_error(nonzero_result))); // expected-error {{'swift_error' attribute with 'nonzero_result' convention can only be applied to a method returning an integral type}}
- (instancetype) harry4: (NSError**) err __attribute__((swift_error(zero_result))); // expected-error {{'swift_error' attribute with 'zero_result' convention can only be applied to a method returning an integral type}}

- (instancetype) harry0 __attribute__((swift_error(none)));
- (instancetype) harry1 __attribute__((swift_error(nonnull_error))); // expected-error {{'swift_error' attribute can only be applied to a method with an error parameter}}
- (instancetype) harry2 __attribute__((swift_error(null_result))); // expected-error {{'swift_error' attribute can only be applied to a method with an error parameter}}
- (instancetype) harry3 __attribute__((swift_error(nonzero_result))); // expected-error {{'swift_error' attribute can only be applied to a method with an error parameter}}
- (instancetype) harry4 __attribute__((swift_error(zero_result))); // expected-error {{'swift_error' attribute can only be applied to a method with an error parameter}}
@end

extern _Bool tom0(CFErrorRef *) __attribute__((swift_error(none)));
extern _Bool tom1(CFErrorRef *) __attribute__((swift_error(nonnull_error)));
extern _Bool tom2(CFErrorRef *) __attribute__((swift_error(null_result))); // expected-error {{'swift_error' attribute with 'null_result' convention can only be applied to a function returning a pointer}}
extern _Bool tom3(CFErrorRef *) __attribute__((swift_error(nonzero_result)));
extern _Bool tom4(CFErrorRef *) __attribute__((swift_error(zero_result)));

extern Undeclared richard0(CFErrorRef *) __attribute__((swift_error(none))); // expected-error {{unknown type name 'Undeclared'}}
extern Undeclared richard1(CFErrorRef *) __attribute__((swift_error(nonnull_error))); // expected-error {{unknown type name 'Undeclared'}}
extern Undeclared richard2(CFErrorRef *) __attribute__((swift_error(null_result))); // expected-error {{unknown type name 'Undeclared'}}
extern Undeclared richard3(CFErrorRef *) __attribute__((swift_error(nonzero_result))); // expected-error {{unknown type name 'Undeclared'}}
extern Undeclared richard4(CFErrorRef *) __attribute__((swift_error(zero_result))); // expected-error {{unknown type name 'Undeclared'}}

extern void *harry0(CFErrorRef *) __attribute__((swift_error(none)));
extern void *harry1(CFErrorRef *) __attribute__((swift_error(nonnull_error)));
extern void *harry2(CFErrorRef *) __attribute__((swift_error(null_result)));
extern void *harry3(CFErrorRef *) __attribute__((swift_error(nonzero_result))); // expected-error {{'swift_error' attribute with 'nonzero_result' convention can only be applied to a function returning an integral type}}
extern void *harry4(CFErrorRef *) __attribute__((swift_error(zero_result))); // expected-error {{'swift_error' attribute with 'zero_result' convention can only be applied to a function returning an integral type}}

extern void *wilma0(void) __attribute__((swift_error(none)));
extern void *wilma1(void) __attribute__((swift_error(nonnull_error))); // expected-error {{'swift_error' attribute can only be applied to a function with an error parameter}}
extern void *wilma2(void) __attribute__((swift_error(null_result))); // expected-error {{'swift_error' attribute can only be applied to a function with an error parameter}}
extern void *wilma3(void) __attribute__((swift_error(nonzero_result))); // expected-error {{'swift_error' attribute can only be applied to a function with an error parameter}}
extern void *wilma4(void) __attribute__((swift_error(zero_result))); // expected-error {{'swift_error' attribute can only be applied to a function with an error parameter}}


extern _Bool suzanne __attribute__((swift_error(none))); // expected-error {{'swift_error' attribute only applies to functions and methods}}
