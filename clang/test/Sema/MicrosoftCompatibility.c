// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify=expected,compat -fms-compatibility -DMSVCCOMPAT -triple i686-pc-win32
// RUN: %clang_cc1 %s -fsyntax-only -Wno-unused-value -Wmicrosoft -verify=expected,ext -fms-extensions -triple i686-pc-win32

#ifdef MSVCCOMPAT
enum ENUM1; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}
enum ENUM1 var1 = 3;
enum ENUM1* var2 = 0;
#else
enum ENUM1; // expected-note {{forward declaration of}}
enum ENUM1 var1 = 3; // expected-error {{variable has incomplete type 'enum ENUM1'}}
enum ENUM1* var2 = 0;
#endif


// FIXME: The rest of this seems to be controlled by -fms-extensions. Move it.
enum ENUM2 {
  ENUM2_a = (enum ENUM2) 4,
  ENUM2_b = 0x9FFFFFFF, // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
  ENUM2_c = 0x100000000 // expected-warning {{enumerator value is not representable in the underlying type 'int'}}
};

__declspec(noreturn) void f6( void ) {
	return;  // expected-warning {{function 'f6' declared 'noreturn' should not return}}
}

__declspec(align(32768)) struct S1 { int a; } s;	/* expected-error {{requested alignment must be 8192 bytes or smaller}} */
struct __declspec(aligned) S2 {}; /* expected-warning {{__declspec attribute 'aligned' is not supported}} */

struct __declspec(appdomain) S3 {}; /* expected-warning {{__declspec attribute 'appdomain' is not supported}} */

__declspec(__noreturn__) void f7(void); /* expected-warning {{__declspec attribute '__noreturn__' is not supported}} */

#ifdef MSVCCOMPAT
size_t x;
#else
size_t x; // expected-error {{unknown type name 'size_t'}}
#endif

/* Microsoft allows inline, __inline, and __forceinline to appear on a typedef
   of a function type; this is used in their system headers such as ufxclient.h
   See GitHub #124869 for more details.
 */
typedef int inline Foo1(int);       // compat-warning {{'inline' can only appear on functions}} \
                                       ext-error {{'inline' can only appear on functions}}
typedef int __inline Foo2(int);     // compat-warning {{'inline' can only appear on functions}} \
                                       ext-error {{'inline' can only appear on functions}}
typedef int __forceinline Foo(int); // compat-warning {{'inline' can only appear on functions}} \
                                       ext-error {{'inline' can only appear on functions}} \
                                       expected-warning {{'__forceinline' attribute only applies to functions and statements}}
