// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s
// rdar://8308053

@class NSObject;

@interface I {
    __attribute__((iboutletcollection(I))) id ivar1;
    __attribute__((iboutletcollection(id))) id ivar2;
    __attribute__((iboutletcollection())) id ivar3;
    __attribute__((iboutletcollection)) id ivar4;
}
@property (nonatomic, retain) __attribute__((iboutletcollection(I))) id prop1;
@property (nonatomic, retain) __attribute__((iboutletcollection(id))) id prop2;
@property (nonatomic, retain) __attribute__((iboutletcollection())) id prop3;
@property (nonatomic, retain) __attribute__((iboutletcollection)) id prop4;
@end

typedef void *PV;
@interface BAD {
    __attribute__((iboutletcollection(I, 1))) id ivar1; // expected-error {{attribute takes one argument}}
    __attribute__((iboutletcollection(B))) id ivar2; // expected-error {{invalid type 'B' as argument of iboutletcollection attribute}}
    __attribute__((iboutletcollection(PV))) id ivar3; // expected-error {{invalid type 'PV' as argument of iboutletcollection attribute}}
    __attribute__((iboutletcollection(PV))) void *ivar4; // expected-warning {{instance variable with 'iboutletcollection' attribute must be an object type (invalid 'void *')}}
    __attribute__((iboutletcollection(int))) id ivar5; // expected-error {{type argument of iboutletcollection attribute cannot be a builtin type}}
    __attribute__((iboutlet)) int ivar6;  // expected-warning {{instance variable with 'iboutlet' attribute must be an object type}}
}
@property (nonatomic, retain) __attribute__((iboutletcollection(I,2,3))) id prop1; // expected-error {{attribute takes one argument}}
@property (nonatomic, retain) __attribute__((iboutletcollection(B))) id prop2; // expected-error {{invalid type 'B' as argument of iboutletcollection attribute}}

@property __attribute__((iboutletcollection(BAD))) int prop3; // expected-warning {{property with 'iboutletcollection' attribute must be an object type (invalid 'int')}}
@end

// rdar://10296078
@interface ParentRDar10296078 @end
@class NSArray;
@protocol RDar10296078_Protocol;
@class RDar10296078_OtherClass;

@interface RDar10296078  : ParentRDar10296078
@property (nonatomic, strong) 
  __attribute__((iboutletcollection(RDar10296078_OtherClass<RDar10296078_Protocol>))) NSArray *stuff; 
@end
