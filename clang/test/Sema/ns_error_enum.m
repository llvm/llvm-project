// RUN: %clang_cc1 -verify %s -x objective-c
// RUN: %clang_cc1 -verify %s -x objective-c++


#define CF_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_ENUM(_type, _name) CF_ENUM(_type, _name)

#define NS_ERROR_ENUM(_type, _name, _domain)  \
  enum _name : _type _name; enum __attribute__((ns_error_domain(_domain))) _name : _type

typedef NS_ENUM(unsigned, MyEnum) {
  MyFirst,
  MySecond,
};

typedef NS_ENUM(invalidType, MyInvalidEnum) {
// expected-error@-1{{unknown type name 'invalidType'}}
// expected-error@-2{{unknown type name 'invalidType'}}
  MyFirstInvalid,
  MySecondInvalid,
};

@interface NSString
@end

extern NSString *const MyErrorDomain;
typedef NS_ERROR_ENUM(unsigned char, MyErrorEnum, MyErrorDomain) {
	MyErrFirst,
	MyErrSecond,
};

typedef NSString *const NsErrorDomain;
extern NsErrorDomain MyTypedefErrorDomain;
typedef NS_ERROR_ENUM(unsigned char, MyTypedefErrorEnum, MyTypedefErrorDomain) {
  MyTypedefErrFirst,
  MyTypedefErrSecond,
};

typedef const struct __CFString * CFStringRef;

extern CFStringRef const MyCFErrorDomain;
typedef NS_ERROR_ENUM(unsigned char, MyCFErrorEnum, MyCFErrorDomain) {
  MyCFErrFirst,
  MyCFErrSecond,
};

typedef CFStringRef CFErrorDomain;
extern CFErrorDomain const MyCFTypedefErrorDomain;
typedef NS_ERROR_ENUM(unsigned char, MyCFTypedefErrorEnum, MyCFTypedefErrorDomain) {
  MyCFTypedefErrFirst,
  MyCFTypedefErrSecond,
};

extern char *const WrongErrorDomainType;
enum __attribute__((ns_error_domain(WrongErrorDomainType))) MyWrongErrorDomainType { MyWrongErrorDomain };
// expected-error@-1{{domain argument 'WrongErrorDomainType' does not point to an NSString or CFString constant}}

struct __attribute__((ns_error_domain(MyErrorDomain))) MyStructWithErrorDomain {};
// expected-error@-1{{'ns_error_domain' attribute only applies to enums}}

int __attribute__((ns_error_domain(MyErrorDomain))) NotTagDecl;
  // expected-error@-1{{'ns_error_domain' attribute only applies to enums}}

enum __attribute__((ns_error_domain())) NoArg { NoArgError };
// expected-error@-1{{'ns_error_domain' attribute takes one argument}}

enum __attribute__((ns_error_domain(MyErrorDomain, MyErrorDomain))) TwoArgs { TwoArgsError };
// expected-error@-1{{'ns_error_domain' attribute takes one argument}}

typedef NS_ERROR_ENUM(unsigned char, MyErrorEnumInvalid, InvalidDomain) {
	// expected-error@-1{{use of undeclared identifier 'InvalidDomain'}}
	MyErrFirstInvalid,
	MyErrSecondInvalid,
};

typedef NS_ERROR_ENUM(unsigned char, MyErrorEnumInvalid, "domain-string");
  // expected-error@-1{{domain argument does not refer to global constant}}

void foo() {}
typedef NS_ERROR_ENUM(unsigned char, MyErrorEnumInvalidFunction, foo);
  // expected-error@-1{{domain argument 'foo' does not refer to global constant}}
