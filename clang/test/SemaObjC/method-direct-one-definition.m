// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-protocol-method-implementation %s

__attribute__((objc_root_class))
@interface A
@end

@interface A (Cat)
- (void)A_Cat __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
@end

@implementation A
- (void)A_Cat { // expected-error {{direct method was declared in a category but is implemented in the primary interface}}
}
@end

__attribute__((objc_root_class))
@interface B
- (void)B_primary __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
@end

@interface B ()
- (void)B_extension __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
@end

@interface B (Cat)
- (void)B_Cat __attribute__((objc_direct));
@end

@interface B (OtherCat)
- (void)B_OtherCat __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
@end

@implementation B (Cat)
- (void)B_primary { // expected-error {{direct method was declared in the primary interface but is implemented in a category}}
}
- (void)B_extension { // expected-error {{direct method was declared in an extension but is implemented in a different category}}
}
- (void)B_Cat {
}
- (void)B_OtherCat { // expected-error {{direct method was declared in a category but is implemented in a different category}}
}
@end

__attribute__((objc_root_class))
@interface C
- (void)C1 __attribute__((objc_direct)); // expected-note {{previous declaration is here}}
- (void)C2;                              // expected-note {{previous declaration is here}}
@end

@interface C (Cat)
- (void)C1;                              // expected-error {{method declaration conflicts with previous direct declaration of method 'C1'}}
- (void)C2 __attribute__((objc_direct)); // expected-error {{direct method declaration conflicts with previous declaration of method 'C2'}}
@end
