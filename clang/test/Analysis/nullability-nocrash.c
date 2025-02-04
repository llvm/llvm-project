// RUN: %clang_analyze_cc1 -w -analyzer-checker=nullability \
// RUN:                       -analyzer-output=text -verify %s
//
// expected-no-diagnostics
//
// Previously there was an assertion requiring that if an Event is handled by
// some enabled checker, then there must be at least one enabled checker which
// can emit that kind of Event.
// This assertion failed when NullabilityChecker (which is a subclass of
// check::Event<ImplicitNullDerefEvent>) was enabled, but the checkers
// inheriting from EventDispatcher<ImplicitNullDerefEvent> were all disabled.
// This test file validates that enabling the nullability checkers (without any
// other checkers) no longer causes a crash.
