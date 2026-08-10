// RUN: %clang_cc1 -fobjc-gc -emit-llvm -o %t %s
// RUN: grep objc_assign_ivar %t | count 6

// PR13820
// REQUIRES: LP64

@interface I @end

typedef I TI;
typedef I* TPI;

typedef id ID;

@interface MyClass {
}

@property id property;
@property I* propertyI;

@property TI* propertyTI;

@property TPI propertyTPI;

@property ID propertyID;
@end

@implementation MyClass
	@synthesize property=_property;
        @synthesize propertyI;
        @synthesize propertyTI=_propertyTI;
        @synthesize propertyTPI=_propertyTPI;
         @synthesize propertyID = _propertyID;
@end

int main () {
    MyClass *myObj;
    myObj.property = 0;
    myObj.propertyI = 0;
    myObj.propertyTI = 0;
    myObj.propertyTPI = 0;
    myObj.propertyID = 0;
    return 0;
}
