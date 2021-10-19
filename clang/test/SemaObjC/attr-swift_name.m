// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

#define SWIFT_NAME(name) __attribute__((__swift_name__(name)))
#define SWIFT_ASYNC_NAME(name) __attribute__((__swift_async_name__(name)))

typedef struct {
  float x, y, z;
} Point3D;

__attribute__((__swift_name__("PType")))
@protocol P
@end

__attribute__((__swift_name__("IClass")))
@interface I<P>
- (instancetype)init SWIFT_NAME("init()");
- (instancetype)initWithValue:(int)value SWIFT_NAME("iWithValue(_:)");

+ (void)refresh SWIFT_NAME("refresh()");

- (instancetype)i SWIFT_NAME("i()");

- (I *)iWithValue:(int)value SWIFT_NAME("i(value:)");
- (I *)iWithValue:(int)value value:(int)value2 SWIFT_NAME("i(value:extra:)");
- (I *)iWithValueConvertingValue:(int)value value:(int)value2 SWIFT_NAME("i(_:extra:)");

+ (I *)iWithOtheValue:(int)value SWIFT_NAME("init");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}

+ (I *)iWithAnotherValue:(int)value SWIFT_NAME("i()");
// expected-warning@-1 {{too few parameters in the signature specified by the '__swift_name__' attribute (expected 1; got 0)}}

+ (I *)iWithYetAnotherValue:(int)value SWIFT_NAME("i(value:extra:)");
// expected-warning@-1 {{too many parameters in the signature specified by the '__swift_name__' attribute (expected 1; got 2}}

+ (I *)iAndReturnErrorCode:(int *)errorCode SWIFT_NAME("i()"); // no-warning
+ (I *)iWithValue:(int)value andReturnErrorCode:(int *)errorCode SWIFT_NAME("i(value:)"); // no-warning

+ (I *)iFromErrorCode:(const int *)errorCode SWIFT_NAME("i()");
// expected-warning@-1 {{too few parameters in the signature specified by the '__swift_name__' attribute (expected 1; got 0)}}

+ (I *)iWithPointerA:(int *)value andReturnErrorCode:(int *)errorCode SWIFT_NAME("i()"); // no-warning
+ (I *)iWithPointerB:(int *)value andReturnErrorCode:(int *)errorCode SWIFT_NAME("i(pointer:)"); // no-warning
+ (I *)iWithPointerC:(int *)value andReturnErrorCode:(int *)errorCode SWIFT_NAME("i(pointer:errorCode:)"); // no-warning

+ (I *)iWithOtherI:(I *)other SWIFT_NAME("i()");
// expected-warning@-1 {{too few parameters in the signature specified by the '__swift_name__' attribute (expected 1; got 0)}}

+ (instancetype)specialI SWIFT_NAME("init(options:)");
+ (instancetype)specialJ SWIFT_NAME("init(options:extra:)");
// expected-warning@-1 {{too many parameters in the signature specified by the '__swift_name__' attribute (expected 0; got 2)}}
+ (instancetype)specialK SWIFT_NAME("init(_:)");
// expected-warning@-1 {{too many parameters in the signature specified by the '__swift_name__' attribute (expected 0; got 1)}}
+ (instancetype)specialL SWIFT_NAME("i(options:)");
// expected-warning@-1 {{too many parameters in the signature specified by the '__swift_name__' attribute (expected 0; got 1)}}

+ (instancetype)trailingParen SWIFT_NAME("foo(");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}
+ (instancetype)trailingColon SWIFT_NAME("foo:");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}
+ (instancetype)initialIgnore:(int)value SWIFT_NAME("_(value:)");
// expected-warning@-1 {{'__swift_name__' attribute has invalid identifier for the base name}}
+ (instancetype)middleOmitted:(int)value SWIFT_NAME("i(:)");
// expected-warning@-1 {{'__swift_name__' attribute has invalid identifier for the parameter name}}

@property(strong) id someProp SWIFT_NAME("prop");
@end

enum SWIFT_NAME("E") E {
  value1,
  value2,
  value3 SWIFT_NAME("three"),
  value4 SWIFT_NAME("four()"), // expected-warning {{'__swift_name__' attribute has invalid identifier for the base name}}
};

struct SWIFT_NAME("TStruct") SStruct {
  int i, j, k SWIFT_NAME("kay");
};

int i SWIFT_NAME("g_i");

void f0(int i) SWIFT_NAME("f_0");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}

void f1(int i) SWIFT_NAME("f_1()");
// expected-warning@-1 {{too few parameters in the signature specified by the '__swift_name__' attribute (expected 1; got 0)}}

void f2(int i) SWIFT_NAME("f_2(a:b:)");
// expected-warning@-1 {{too many parameters in the signature specified by the '__swift_name__' attribute (expected 1; got 2)}}

void f3(int x, int y) SWIFT_NAME("fWithX(_:y:)");
void f4(int x, int *error) SWIFT_NAME("fWithX(_:)");

typedef int int_t SWIFT_NAME("IntType");

struct Point3D createPoint3D(float x, float y, float z) SWIFT_NAME("Point3D.init(x:y:z:)");
struct Point3D rotatePoint3D(Point3D point, float radians) SWIFT_NAME("Point3D.rotate(self:radians:)");
struct Point3D badRotatePoint3D(Point3D point, float radians) SWIFT_NAME("Point3D.rotate(radians:)");
// expected-warning@-1 {{too few parameters in the signature specified by the '__swift_name__' attribute (expected 2; got 1)}}

extern struct Point3D identityPoint SWIFT_NAME("Point3D.identity");

float Point3DGetMagnitude(Point3D point) SWIFT_NAME("getter:Point3D.magnitude(self:)");
float Point3DGetMagnitudeAndSomethingElse(Point3D point, float f) SWIFT_NAME("getter:Point3D.magnitude(self:f:)");
// expected-warning@-1 {{'__swift_name__' attribute for getter must not have any parameters besides 'self:'}}

float Point3DGetRadius(Point3D point) SWIFT_NAME("getter:Point3D.radius(self:)");
void Point3DSetRadius(Point3D point, float radius) SWIFT_NAME("setter:Point3D.radius(self:newValue:)");

float Point3DPreGetRadius(Point3D point) SWIFT_NAME("getter:Point3D.preRadius(self:)");
void Point3DPreSetRadius(float radius, Point3D point) SWIFT_NAME("setter:Point3D.preRadius(newValue:self:)");

void Point3DSetRadiusAndSomethingElse(Point3D point, float radius, float f) SWIFT_NAME("setter:Point3D.radius(self:newValue:f:)");
// expected-warning@-1 {{'__swift_name__' attribute for setter must have one parameter for new value}}

float Point3DGetComponent(Point3D point, unsigned index) SWIFT_NAME("getter:Point3D.subscript(self:_:)");
float Point3DSetComponent(Point3D point, unsigned index, float value) SWIFT_NAME("setter:Point3D.subscript(self:_:newValue:)");

float Point3DGetMatrixComponent(Point3D point, unsigned x, unsigned y) SWIFT_NAME("getter:Point3D.subscript(self:x:y:)");
void Point3DSetMatrixComponent(Point3D point, unsigned x, float value, unsigned y) SWIFT_NAME("setter:Point3D.subscript(self:x:newValue:y:)");

float Point3DSetWithoutNewValue(Point3D point, unsigned x, unsigned y) SWIFT_NAME("setter:Point3D.subscript(self:x:y:)");
// expected-warning@-1 {{'__swift_name__' attribute for 'subscript' setter must have a 'newValue:' parameter}}

float Point3DSubscriptButNotGetterSetter(Point3D point, unsigned x) SWIFT_NAME("Point3D.subscript(self:_:)");
// expected-warning@-1 {{'__swift_name__' attribute for 'subscript' must be a getter or setter}}

void Point3DSubscriptSetterTwoNewValues(Point3D point, unsigned x, float a, float b) SWIFT_NAME("setter:Point3D.subscript(self:_:newValue:newValue:)");
// expected-warning@-1 {{'__swift_name__' attribute for 'subscript' setter cannot have multiple 'newValue:' parameters}}

float Point3DSubscriptGetterNewValue(Point3D point, unsigned x, float a, float b) SWIFT_NAME("getter:Point3D.subscript(self:_:newValue:newValue:)");
// expected-warning@-1 {{'__swift_name__' attribute for 'subscript' getter cannot have a 'newValue:' parameter}}

void Point3DMethodWithNewValue(Point3D point, float newValue) SWIFT_NAME("Point3D.method(self:newValue:)");
void Point3DMethodWithNewValues(Point3D point, float newValue, float newValueB) SWIFT_NAME("Point3D.method(self:newValue:newValue:)");

float Point3DStaticSubscript(unsigned x) SWIFT_NAME("getter:Point3D.subscript(_:)");
// expected-warning@-1 {{'__swift_name__' attribute for 'subscript' must have a 'self:' parameter}}

float Point3DStaticSubscriptNoArgs(void) SWIFT_NAME("getter:Point3D.subscript()");
// expected-warning@-1 {{'__swift_name__' attribute for 'subscript' must have at least one parameter}}

float Point3DPreGetComponent(Point3D point, unsigned index) SWIFT_NAME("getter:Point3D.subscript(self:_:)");

Point3D getCurrentPoint3D(void) SWIFT_NAME("getter:currentPoint3D()");

void setCurrentPoint3D(Point3D point) SWIFT_NAME("setter:currentPoint3D(newValue:)");

Point3D getLastPoint3D(void) SWIFT_NAME("getter:lastPoint3D()");

void setLastPoint3D(Point3D point) SWIFT_NAME("setter:lastPoint3D(newValue:)");

Point3D getZeroPoint(void) SWIFT_NAME("getter:Point3D.zero()");
void setZeroPoint(Point3D point) SWIFT_NAME("setter:Point3D.zero(newValue:)");
Point3D getZeroPointNoPrototype() SWIFT_NAME("getter:Point3D.zeroNoPrototype()");
// expected-warning@-1 {{'__swift_name__' attribute only applies to non-K&R-style functions}}

Point3D badGetter1(int x) SWIFT_NAME("getter:bad1(_:)");
// expected-warning@-1 {{'__swift_name__' attribute for getter must not have any parameters besides 'self:'}}

void badSetter1(void) SWIFT_NAME("getter:bad1())");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}

Point3D badGetter2(Point3D point) SWIFT_NAME("getter:bad2(_:))");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}

void badSetter2(Point3D point) SWIFT_NAME("setter:bad2(self:))");
// expected-warning@-1 {{'__swift_name__' attribute argument must be a string literal specifying a Swift function name}}

void g(int i) SWIFT_NAME("function(int:)");
// expected-note@-1 {{conflicting attribute is here}}

// expected-error@+1 {{'swift_name' and 'swift_name' attributes are not compatible}}
void g(int i) SWIFT_NAME("function(_:)") {
}

typedef int (^CallbackTy)(void);

@interface AsyncI<P>

- (void)doSomethingWithCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething()");
- (void)doSomethingX:(int)x withCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething(x:)");

// expected-warning@+1 {{too many parameters in the signature specified by the '__swift_async_name__' attribute (expected 1; got 2)}}
- (void)doSomethingY:(int)x withCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething(x:y:)");

// expected-warning@+1 {{too few parameters in the signature specified by the '__swift_async_name__' attribute (expected 1; got 0)}}
- (void)doSomethingZ:(int)x withCallback:(CallbackTy)callback SWIFT_ASYNC_NAME("doSomething()");

// expected-warning@+1 {{'__swift_async_name__' attribute cannot be applied to a method with no parameters}}
- (void)doSomethingNone SWIFT_ASYNC_NAME("doSomething()");

// expected-error@+1 {{'__swift_async_name__' attribute takes one argument}}
- (void)brokenAttr __attribute__((__swift_async_name__("brokenAttr", 2)));

@end

void asyncFunc(CallbackTy callback) SWIFT_ASYNC_NAME("asyncFunc()");

// expected-warning@+1 {{'__swift_async_name__' attribute cannot be applied to a function with no parameters}}
void asyncNoParams(void) SWIFT_ASYNC_NAME("asyncNoParams()");

// expected-error@+1 {{'__swift_async_name__' attribute only applies to Objective-C methods and functions}}
SWIFT_ASYNC_NAME("NoAsync")
@protocol NoAsync @end
