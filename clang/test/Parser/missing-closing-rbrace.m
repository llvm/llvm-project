// RUN: %clang_cc1 -fsyntax-only -verify %s
@interface A {@end // expected-error {{'@end' appears where closing brace '}' is expected}}
