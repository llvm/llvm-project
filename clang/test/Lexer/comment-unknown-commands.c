// RUN: %clang_cc1 -verify %s -fsyntax-only -Wdocumentation -Wdocumentation-unknown-command

/// A test function with an `@command` in backticks
void test(void) {}
// ok: the command name is in backticks

/// A test function with an "@command" in quotes
void test2(void) {}
// ok: the command name is in quotes

/// A test function with an @command outside of quotes or backticks
void test3(void) {}
// expected-warning@-2 {{unknown command tag name}}
