// RUN: %clang_cc1 %s -verify -fsyntax-only
// vim: set binary noeol:

// This file intentionally ends without a \n on the last line.  Make sure your
// editor doesn't add one.

>>>> ORIGINAL
// expected-error@-1 {{version control conflict marker in file}}
<<<<
// expected-error@-1 {{expected identifier or '('}}
<<<<