// RUN: not %clang_cc1 -std=c++2a -E -I%S/Inputs %s -o - | FileCheck %s --strict-whitespace --implicit-check-not=ERROR

// Check for context-sensitive header-name token formation.
// CHECK: import <foo  bar>;
import <foo  bar>;

// Not at the top level: these are each 8 tokens rather than 5.
// CHECK: { import <foo bar>; }
{ import <foo  bar>; }
// CHECK: ( import <foo bar>; :>
( import <foo  bar>; :>
// CHECK: [ import <foo bar>; %>
[ import <foo  bar>; %>

// CHECK: import <foo  bar>;
import <foo  bar>;

// CHECK: foo; import <foo  bar>;
foo; import <foo  bar>;

// CHECK: foo import <foo bar>;
foo import <foo  bar>;

// CHECK: import <foo  bar> {{\[\[ ]]}};
import <foo  bar> [[ ]];

// CHECK: import <foo  bar> import <foo bar>;
import <foo  bar> import <foo  bar>;

// FIXME: We do not form header-name tokens in the pp-import-suffix of a
// pp-import. Conforming programs can't tell the difference.
// CHECK: import <foo  bar> {} import <foo bar>;
// FIXME: import <foo  bar> {} import <foo  bar>;
import <foo  bar> {} import <foo  bar>;


// CHECK: export import <foo  bar>;
export import <foo  bar>;

// CHECK: export export import <foo bar>;
export export import <foo  bar>;

#define UNBALANCED_PAREN (
// CHECK: import <foo  bar>;
import <foo  bar>;

UNBALANCED_PAREN
// CHECK: import <foo bar>;
import <foo  bar>;
)

_Pragma("clang no_such_pragma (");
// CHECK: import <foo  bar>;
import <foo  bar>;

#define HEADER <foo  bar>
// CHECK: import <foo bar>;
import HEADER;

// CHECK: import <foo bar>;
import <
foo
  bar
>;

// CHECK: import{{$}}
// CHECK: {{^}}<foo bar>;
import
<
foo
  bar
>;

// CHECK: import{{$}}
// CHECK: {{^}}<foo  bar>;
import
<foo  bar>;

#define IMPORT import <foo  bar>
// CHECK: import <foo bar>;
IMPORT;
