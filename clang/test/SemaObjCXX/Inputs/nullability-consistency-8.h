typedef int* _Nonnull mynonnull;

__attribute__((objc_root_class))
@interface typedefClass
- (void) func1:(mynonnull)i;
@end

void func2(mynonnull i);

void func3(int *); // expected-warning{{pointer is missing a nullability type specifier}}

#define CF_RETURNS_NOT_RETAINED __attribute__((cf_returns_not_retained))
typedef void *CFTypeRef;
void cf1(CFTypeRef * p CF_RETURNS_NOT_RETAINED); // expected-warning {{pointer is missing a nullability type specifier}}

void cf2(CFTypeRef * _Nullable p CF_RETURNS_NOT_RETAINED);
void cf3(CFTypeRef * _Nonnull p CF_RETURNS_NOT_RETAINED);

void cf4(CFTypeRef _Nullable * _Nullable p CF_RETURNS_NOT_RETAINED);
void cf5(CFTypeRef _Nonnull * _Nullable p CF_RETURNS_NOT_RETAINED);

void cf6(CFTypeRef * _Nullable CF_RETURNS_NOT_RETAINED p);
void cf7(CF_RETURNS_NOT_RETAINED CFTypeRef * _Nonnull p);

typedef CFTypeRef _Nullable *CFTypeRefPtr;
void cfp1(CFTypeRefPtr p CF_RETURNS_NOT_RETAINED); // expected-warning {{pointer is missing a nullability type specifier}}
void cfp2(CFTypeRefPtr _Nonnull p CF_RETURNS_NOT_RETAINED);
