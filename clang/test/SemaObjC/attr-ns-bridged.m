// RUN: %clang_cc1 -fsyntax-only -verify %s

typedef struct __attribute__((ns_bridged)) test0s *test0ref;

void test0func(void) __attribute__((ns_bridged)); // expected-error {{'ns_bridged' attribute only applies to structs}}

union __attribute__((ns_bridged)) test0u; // expected-error {{'ns_bridged' attribute only applies to structs}}

struct __attribute__((ns_bridged(Test1))) test1s;

@class Test2;
struct __attribute__((ns_bridged(Test2))) test2s;

void Test3(void); // expected-note {{declared here}}
struct __attribute__((ns_bridged(Test3))) test3s; // expected-error {{parameter of 'ns_bridged' attribute does not name an Objective-C class}}
