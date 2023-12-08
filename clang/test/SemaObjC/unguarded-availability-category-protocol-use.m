// RUN: %clang_cc1 -triple arm64-apple-ios10 -Wunguarded-availability -fblocks -fsyntax-only -verify %s

__attribute__((availability(ios,unavailable)))
@protocol Prot // expected-note {{here}}

@end

@interface A
@end

__attribute__((availability(ios,unavailable)))
@interface A (Cat) <Prot> // No error.
@end

__attribute__((availability(tvos,unavailable)))
@interface B @end
@interface B (Cat) <Prot> // expected-error {{'Prot' is unavailable: not available on iOS}}
@end
