// RUN: %clang_cc1 -triple arm64-apple-ios -std=c23 -fsyntax-only -verify -fptrauth-intrinsics %s
// RUN: %clang_cc1 -triple aarch64-linux-gnu -std=c23 -fsyntax-only -verify -fptrauth-intrinsics %s

#ifndef __PTRAUTH__
#error "the ptrauth qualifier should be available"
#endif

#if __aarch64__
#define VALID_CODE_KEY 0
#define VALID_DATA_KEY 2
#define INVALID_KEY 200
#else
#error Provide these constants if you port this test
#endif


typedef int *intp;

int *__ptrauth(VALID_DATA_KEY, 1, 65535, "Foo") invalid13_1;
// expected-error@-1 {{unknown '__ptrauth' authentication option 'Foo'}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip", 41) invalid14_1;
// expected-error@-1 {{'__ptrauth' qualifier must take between 1 and 4 arguments}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip,sign-and-strip") invalid15;
// expected-error@-1 {{repeated '__ptrauth' authentication mode, prior mode was 'strip'}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "isa-pointer,isa-pointer") invalid16;
// expected-error@-1 {{repeated '__ptrauth' authentication option}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip, , isa-pointer") invalid18;
// expected-error@-1 {{'__ptrauth' options parameter contains an empty option}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip,") invalid19;
// expected-error@-1 {{'__ptrauth' options parameter has a trailing comma}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, ",") invalid20;
// expected-error@-1 {{'__ptrauth' options parameter contains an empty option}}
// expected-error@-2 {{'__ptrauth' options parameter has a trailing comma}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, ",,") invalid21;
// expected-error@-1 2 {{'__ptrauth' options parameter contains an empty option}}
// expected-error@-2 {{'__ptrauth' options parameter has a trailing comma}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip isa-pointer") invalid22;
// expected-error@-1 {{missing comma after 'strip' option in '__ptrauth' qualifier}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip\nisa-pointer") invalid23;
// expected-error@-1 {{missing comma after 'strip' option in '__ptrauth' qualifier}}
int *__ptrauth(VALID_DATA_KEY, 1, 65535, "strip"
                                         " isa-pointer") invalid24;
// expected-error@-2{{missing comma after 'strip' option in '__ptrauth' qualifier}}
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip,\n,isa-pointer") invalid25; // expected-error{{'__ptrauth' options parameter contains an empty option}}
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip,\t,isa-pointer") invalid26; // expected-error{{'__ptrauth' options parameter contains an empty option}}

int *__ptrauth(VALID_DATA_KEY, 1, 0, "strip") valid12;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip") valid13;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-auth") valid14;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "isa-pointer") valid15;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-auth,isa-pointer") valid15;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip,isa-pointer") valid16;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "strip,isa-pointer") valid17;
int *__ptrauth(VALID_DATA_KEY, 1, 0, " strip,isa-pointer") valid18;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "strip ,isa-pointer") valid19;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "strip, isa-pointer") valid20;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "strip,isa-pointer ") valid21;
int *__ptrauth(VALID_DATA_KEY, 1, 0, " strip") valid22;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "strip ") valid23;
int *__ptrauth(VALID_DATA_KEY, 1, 0, " strip ") valid24;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip,"
                                     "isa-pointer") valid25;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip"
                                     ",isa-pointer") valid26;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip\n,isa-pointer") valid27;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "sign-and-strip\t,isa-pointer") valid28;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "") valid29;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "authenticates-null-values") valid30;
int *__ptrauth(VALID_DATA_KEY, 1, 0, "isa-pointer, authenticates-null-values") valid31;

int *global_ptr;

// Check qualifier option printing
int *__ptrauth(1, 1, 0, "strip") * strip = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0,"strip") *}}
int *__ptrauth(1, 1, 0, "sign-and-strip") * signAndStrip = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0,"sign-and-strip") *}}
int *__ptrauth(1, 1, 0, "sign-and-auth") * signAndAuth = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0) *}}
int *__ptrauth(1, 1, 0, "sign-and-auth, isa-pointer") * signAndAuthIsa = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0,"isa-pointer") *}}
int *__ptrauth(1, 1, 0, "sign-and-strip, isa-pointer") * signAndStripIsa = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0,"sign-and-strip,isa-pointer") *'}}
int *__ptrauth(1, 1, 0, "sign-and-strip, authenticates-null-values") * signAndAuthAuthenticatesNullValues = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0,"sign-and-strip,authenticates-null-values") *}}
int *__ptrauth(1, 1, 0, "sign-and-strip, authenticates-null-values, isa-pointer") * signAndAuthIsaPointerAuthenticatesNullValues = &global_ptr;
// expected-error@-1 {{int *__ptrauth(1,1,0,"sign-and-strip,isa-pointer,authenticates-null-values") *}}

// Check mismatching options are identified
void f() {
    // default auth matches explicit use of sign-and-auth
    int *__ptrauth(1, 1, 0) * signAndAuthDefault = signAndAuth;

    int *__ptrauth(1, 1, 0, "sign-and-strip") * signAndStrip2 = strip;
    // expected-error@-1 {{initializing 'int *__ptrauth(1,1,0,"sign-and-strip") *' with an expression of type 'int *__ptrauth(1,1,0,"strip") *' changes pointer authentication of pointee type}}
    int *__ptrauth(1, 1, 0) * signAndAuth2 = signAndStrip;
    // expected-error@-1 {{'int *__ptrauth(1,1,0) *' with an expression of type 'int *__ptrauth(1,1,0,"sign-and-strip") *' changes pointer authentication of pointee type}}
    int *__ptrauth(1, 1, 0, "authenticates-null-values") * signAndAuthAuthenticatesNullValues2 = signAndAuth;
    // expected-error@-1 {{initializing 'int *__ptrauth(1,1,0,"authenticates-null-values") *' with an expression of type 'int *__ptrauth(1,1,0) *' changes pointer authentication of pointee type}}
    int *__ptrauth(0, 1, 0) * signAndAuth3 = signAndAuth; // different key
    // expected-error@-1 {{initializing 'int *__ptrauth(0,1,0) *' with an expression of type 'int *__ptrauth(1,1,0) *' changes pointer authentication of pointee type}}
    int *__ptrauth(1, 0, 0) * signAndAuth4 = signAndAuth; // different address discrimination
    // expected-error@-1 {{initializing 'int *__ptrauth(1,0,0) *' with an expression of type 'int *__ptrauth(1,1,0) *' changes pointer authentication of pointee type}}
    int *__ptrauth(1, 1, 1) * signAndAuth5 = signAndAuth; // different discriminator
    // expected-error@-1 {{initializing 'int *__ptrauth(1,1,1) *' with an expression of type 'int *__ptrauth(1,1,0) *' changes pointer authentication of pointee type}}
}




