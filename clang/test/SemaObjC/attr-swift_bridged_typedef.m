// RUN: %clang_cc1 -verify -fsyntax-only %s

@interface NSString
@end

typedef NSString *NSStringAlias __attribute__((__swift_bridged_typedef__));

typedef int IntAlias __attribute__((__swift_bridged_typedef__));

struct __attribute__((swift_bridged_typedef)) S {};
// expected-error@-1 {{'swift_bridged_typedef' attribute only applies to typedefs}}

typedef unsigned char UnsignedChar __attribute__((__swift_bridged_typedef__("UnsignedChar")));
// expected-error@-1 {{'__swift_bridged_typedef__' attribute takes no arguments}}
