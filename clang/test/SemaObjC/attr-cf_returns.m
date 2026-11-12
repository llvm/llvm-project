// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

#if __has_feature(attribute_cf_returns_on_parameters)
# error "okay!"
// expected-error@-1 {{okay!}}
#else
# error "uh-oh"
#endif

#define CF_RETURNS_RETAINED __attribute__((cf_returns_retained))
#define CF_RETURNS_NOT_RETAINED __attribute__((cf_returns_not_retained))

int x CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to functions, methods, and parameters}}
int y CF_RETURNS_NOT_RETAINED; // expected-warning{{'cf_returns_not_retained' attribute only applies to functions, methods, and parameters}}

typedef struct __CFFoo *CFFooRef;

int invalid1(void) CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to functions that return a pointer}}
void invalid2(void) CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to functions that return a pointer}}

CFFooRef valid1(void) CF_RETURNS_RETAINED;
id valid2(void) CF_RETURNS_RETAINED;

@interface Test
- (int)invalid1 CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to methods that return a pointer}}
- (void)invalid2 CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to methods that return a pointer}}

- (CFFooRef)valid1 CF_RETURNS_RETAINED;
- (id)valid2 CF_RETURNS_RETAINED;

@property int invalidProp1 CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to properties that return a pointer}}
@property void invalidProp2 CF_RETURNS_RETAINED; // expected-warning{{'cf_returns_retained' attribute only applies to properties that return a pointer}}

@property CFFooRef valid1 CF_RETURNS_RETAINED;
@property id valid2 CF_RETURNS_RETAINED;
@end

void invalidParam(int a CF_RETURNS_RETAINED, // expected-warning{{'cf_returns_retained' attribute only applies to pointer-to-CF-pointer parameters}}
                  int *b CF_RETURNS_RETAINED, // expected-warning{{'cf_returns_retained' attribute only applies to pointer-to-CF-pointer parameters}}
                  id c CF_RETURNS_RETAINED, // expected-warning{{'cf_returns_retained' attribute only applies to pointer-to-CF-pointer parameters}}
                  void *d CF_RETURNS_RETAINED, // expected-warning{{'cf_returns_retained' attribute only applies to pointer-to-CF-pointer parameters}}
                  CFFooRef e CF_RETURNS_RETAINED); // expected-warning{{'cf_returns_retained' attribute only applies to pointer-to-CF-pointer parameters}}

void validParam(id *a CF_RETURNS_RETAINED,
                CFFooRef *b CF_RETURNS_RETAINED,
                void **c CF_RETURNS_RETAINED);
void validParam2(id *a CF_RETURNS_NOT_RETAINED,
                 CFFooRef *b CF_RETURNS_NOT_RETAINED,
                 void **c CF_RETURNS_NOT_RETAINED);
