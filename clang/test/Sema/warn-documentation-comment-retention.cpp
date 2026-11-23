// RUN: %clang_cc1 -fsyntax-only -verify %s

// The comment-retention optimization (Sema::shouldRetainCommentsInAST) must
// still parse a documentation comment when -Wdocumentation is enabled at the
// comment's location -- including when it is turned on by a #pragma clang
// diagnostic rather than on the command line. The check is done at the
// comment's location precisely so pragma regions are honored.

/// \returns Aaa
void outside();
// -Wdocumentation is off at this location, so the comment is not checked and
// no diagnostic is produced. -verify fails on any unexpected diagnostic, so
// this line asserts the comment is *not* diagnosed here.

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wdocumentation"
/// \returns Aaa
void inside();
// expected-warning@-2 {{'\returns' command used in a comment that is attached to a function returning void}}
#pragma clang diagnostic pop

// Any warning in the -Wdocumentation group must keep the comment, not just a
// hard-coded subset: -Wdocumentation-html is a subgroup of -Wdocumentation.
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wdocumentation-html"
/// Aaa <br></br>
void html_inside();
// expected-warning@-2 {{HTML end tag 'br' is forbidden}}
#pragma clang diagnostic pop

// -Wdocumentation-unknown-command is under -Wdocumentation-pedantic, which is
// not a subgroup of -Wdocumentation and must be checked separately.
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wdocumentation-unknown-command"
/// \unknowncommand Aaa
void unknown_inside();
// expected-warning@-2 {{unknown command tag name}}
#pragma clang diagnostic pop
