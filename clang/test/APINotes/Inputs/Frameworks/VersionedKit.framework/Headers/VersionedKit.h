void moveToPoint(double x, double y) __attribute__((swift_name("moveTo(x:y:)")));

void acceptClosure(void (^ __attribute__((noescape)) block)(void));

@class NSString;

extern NSString *MyErrorDomain;

enum __attribute__((ns_error_domain(MyErrorDomain))) MyErrorCode {
  MyErrorCodeFailed = 1
};

__attribute__((swift_bridge("MyValueType")))
@interface MyReferenceType
@end

void privateFunc(void) __attribute__((swift_private));

typedef double MyDoubleWrapper __attribute__((swift_wrapper(struct)));

@interface TestProperties
@property (nonatomic, readwrite, retain) id accessorsOnly;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClass;

@property (nonatomic, readwrite, retain) id accessorsOnlyInVersion3;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClassInVersion3;

@property (nonatomic, readwrite, retain) id accessorsOnlyExceptInVersion3;
@property (nonatomic, readwrite, retain, class) id accessorsOnlyForClassExceptInVersion3;
@end
