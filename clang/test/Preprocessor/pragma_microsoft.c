// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions

// rdar://6495941

#define FOO 1
#define BAR "2"

#pragma comment(linker,"foo=" FOO) // expected-error {{pragma comment requires parenthesized identifier and optional string}}
#pragma comment(linker," bar=" BAR)

#pragma comment( user, "Compiled on " __DATE__ " at " __TIME__ ) 

#pragma comment(foo)    // expected-error {{unknown kind of pragma comment}}
#pragma comment(compiler,)     // expected-error {{pragma comment requires}}
#define foo compiler
#pragma comment(foo)   // macro expand kind.
#pragma comment(foo) x // expected-error {{pragma comment requires}}

#pragma comment(user, "foo\abar\nbaz\tsome	thing")


// __pragma

__pragma(comment(linker," bar=" BAR))                                            

#define MACRO_WITH__PRAGMA { \
  __pragma(warning(push)); \
  __pragma(warning(disable: 10000)); \
  2+2; \
  __pragma(warning(pop)); \
}

void f()
{
  __pragma()

  // If we ever actually *support* __pragma(warning(disable: x)),
  // this warning should go away.
  MACRO_WITH__PRAGMA // expected-warning {{expression result unused}}
}


// This should include macro_arg_directive even though the include
// is looking for test.h  This allows us to assign to "n"
#pragma include_alias("test.h", "macro_arg_directive.h" )
#include "test.h"
void test( void ) {
  n = 12;
}

#pragma include_alias(<bar.h>, "bar.h") // expected-warning {{angle-bracketed include <bar.h> cannot be aliased to double-quoted include "bar.h"}}
#pragma include_alias("foo.h", <bar.h>) // expected-warning {{double-quoted include "foo.h" cannot be aliased to angle-bracketed include <bar.h>}}
#pragma include_alias("test.h") // expected-warning {{pragma include_alias expected ','}}

// Make sure that the names match exactly for a replacement, including path information.  If
// this were to fail, we would get a file not found error
#pragma include_alias(".\pp-record.h", "does_not_exist.h")
#include "pp-record.h"

#pragma include_alias(12) // expected-warning {{pragma include_alias expected include filename}}

// It's expected that we can map "bar" and <bar> separately
#define test
// We can't actually look up stdio.h because we're using cc1 without header paths, but this will ensure
// that we get the right bar.h, because the "bar.h" will undef test for us, where <bar.h> won't
#pragma include_alias(<bar.h>, <stdio.h>)
#pragma include_alias("bar.h", "pr2086.h")  // This should #undef test

#include "bar.h"
#if defined(test)
// This should not warn because test should not be defined
#pragma include_alias("test.h")
#endif

// Test to make sure there are no use-after-free problems
#define B "pp-record.h"
#pragma include_alias("quux.h", B)
void g() {}
#include "quux.h"

// Make sure that empty includes don't work
#pragma include_alias("", "foo.h")  // expected-error {{empty filename}}
#pragma include_alias(<foo.h>, <>)  // expected-error {{empty filename}}
