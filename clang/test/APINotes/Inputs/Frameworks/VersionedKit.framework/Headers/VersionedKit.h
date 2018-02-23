void moveToPointDUMP(double x, double y) __attribute__((swift_name("moveTo(x:y:)")));

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

@interface Base
@end

@interface TestGenericDUMP<Element> : Base
- (Element)element;
@end

@interface Swift3RenamedOnlyDUMP
@end

__attribute__((swift_name("Swift4Name")))
@interface Swift3RenamedAlsoDUMP
@end


enum __attribute__((flag_enum)) FlagEnum {
  FlagEnumA = 1,
  FlagEnumB = 2
};

enum __attribute__((flag_enum)) NewlyFlagEnum {
  NewlyFlagEnumA = 1,
  NewlyFlagEnumB = 2
};

enum APINotedFlagEnum {
  APINotedFlagEnumA = 1,
  APINotedFlagEnumB = 2
};


enum __attribute__((enum_extensibility(open))) OpenEnum {
  OpenEnumA = 1,
};

enum __attribute__((enum_extensibility(open))) NewlyOpenEnum {
  NewlyOpenEnumA = 1,
};

enum __attribute__((enum_extensibility(closed))) NewlyClosedEnum {
  NewlyClosedEnumA = 1,
};

enum __attribute__((enum_extensibility(open))) ClosedToOpenEnum {
  ClosedToOpenEnumA = 1,
};

enum __attribute__((enum_extensibility(closed))) OpenToClosedEnum {
  OpenToClosedEnumA = 1,
};

enum APINotedOpenEnum {
  APINotedOpenEnumA = 1,
};

enum APINotedClosedEnum {
  APINotedClosedEnumA = 1,
};


enum SoonToBeCFEnum {
  SoonToBeCFEnumA = 1
};
enum SoonToBeNSEnum {
  SoonToBeNSEnumA = 1
};
enum SoonToBeCFOptions {
  SoonToBeCFOptionsA = 1
};
enum SoonToBeNSOptions {
  SoonToBeNSOptionsA = 1
};
enum SoonToBeCFClosedEnum {
  SoonToBeCFClosedEnumA = 1
};
enum SoonToBeNSClosedEnum {
  SoonToBeNSClosedEnumA = 1
};
enum UndoAllThatHasBeenDoneToMe {
  UndoAllThatHasBeenDoneToMeA = 1
} __attribute__((flag_enum)) __attribute__((enum_extensibility(closed)));
