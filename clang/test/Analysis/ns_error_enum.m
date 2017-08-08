// RUN: %clang_cc1 -verify %s

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

const char *MyErrorDomain;
typedef NS_ERROR_ENUM(unsigned char, MyErrorEnum, MyErrorDomain) {
	MyErrFirst,
	MyErrSecond,
};
struct __attribute__((ns_error_domain(MyErrorDomain))) MyStructErrorDomain {};

typedef NS_ERROR_ENUM(unsigned char, MyErrorEnumInvalid, InvalidDomain) {
	// expected-error@-1{{domain argument 'InvalidDomain' does not refer to global constant}}
	MyErrFirstInvalid,
	MyErrSecondInvalid,
};

typedef NS_ERROR_ENUM(unsigned char, MyErrorEnumInvalid, "domain-string");
  // expected-error@-1{{domain argument must be an identifier}}

int __attribute__((ns_error_domain(MyErrorDomain))) NotTagDecl;
  // expected-error@-1{{ns_error_domain attribute only valid on enums, structs, and unions}}

void foo() {}
typedef NS_ERROR_ENUM(unsigned char, MyErrorEnumInvalidFunction, foo);
  // expected-error@-1{{domain argument 'foo' does not refer to global constant}}
