// RUN: %clang_cc1 -fsyntax-only -verify %s

void __assert_rtn(const char *, const char *, int, const char *) __attribute__((__noreturn__));
static __inline__ int __inline_isfinitef (float ) __attribute__ ((always_inline));

@interface NSATSTypesetter (NSPantherCompatibility) // expected-error {{cannot find interface declaration for 'NSATSTypesetter'}}
- (id)lineFragmentRectForProposedRect:(id)proposedRect remainingRect:(id)remainingRect __attribute__((deprecated));
@end
