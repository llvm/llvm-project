// RUN: %clang_cc1 %s -fsyntax-only -verify
// expected-no-diagnostics

// id is now builtin. There should be no errors. 
id obj; 

@interface Foo

- defaultToId; 

@end
