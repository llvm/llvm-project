// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify -fms-compatibility

enum ENUM1; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}
enum ENUM1 var1 = 3;
enum ENUM1* var2 = 0;


enum ENUM2 {
  ENUM2_a = (enum ENUM2) 4,
  ENUM2_b = 0x9FFFFFFF, // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
  ENUM2_c = 0x100000000 // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
};

__declspec(noreturn) void f6( void ) {
	return;  // expected-warning {{function 'f6' declared 'noreturn' should not return}}
}

__declspec(align(32768)) struct S1 { int a; } s;	/* expected-error {{requested alignment must be 8192 bytes or smaller}} */
struct __declspec(aligned) S2 {}; /* expected-warning {{unknown __declspec attribute 'aligned' ignored}} */

struct __declspec(appdomain) S3 {}; /* expected-warning {{__declspec attribute 'appdomain' is not supported}} */
