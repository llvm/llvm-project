int *ptr; // expected-warning {{missing a nullability type specifier}}
// expected-note@-1{{insert '_Nullable' if the pointer may be null}}
// expected-note@-2{{insert '_Nonnull' if the pointer should never be null}}
#pragma clang assume_nonnull begin

extern void **blah; // expected-warning 2{{missing a nullability type specifier}}
// expected-note@-1 2 {{insert '_Nullable' if the pointer may be null}}
// expected-note@-2 2 {{insert '_Nonnull' if the pointer should never be null}}

__attribute__((objc_root_class))
@interface ClassWithWeakProperties
@property (readonly, weak) ClassWithWeakProperties *prop1;
@property (readonly, weak, null_unspecified) ClassWithWeakProperties *prop2;
@end

@interface ClassWithWeakProperties ()
@property (readonly, weak) ClassWithWeakProperties *prop3;
@end

#pragma clang assume_nonnull end

