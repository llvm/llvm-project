// RUN: %clang_cc1 -std=c23 %s -E -verify

// Test the parsing behavior for #embed and all of its parameters to ensure we
// recover from failures gracefully.
char buffer[] = {
#embed
// expected-error@-1 {{expected "FILENAME" or <FILENAME>}}

#embed <
// expected-error@-1 {{expected '>'}} \
   expected-note@-1 {{to match this '<'}}

#embed "
// expected-error@-1 {{expected "FILENAME" or <FILENAME>}} \
   expected-warning@-1 {{missing terminating '"' character}}

#embed file.txt
// expected-error@-1{{expected "FILENAME" or <FILENAME>}}

#embed "embed_parsing_errors.c" xxx
// expected-error@-1 {{unknown embed preprocessor parameter 'xxx'}}

#embed "embed_parsing_errors.c" xxx::
// expected-error@-1 {{expected identifier}}

#embed "embed_parsing_errors.c" xxx::xxx
// expected-error@-1 {{unknown embed preprocessor parameter 'xxx::xxx'}}

#embed "embed_parsing_errors.c" xxx::42
// expected-error@-1 {{expected identifier}}

#embed "embed_parsing_errors.c" limit
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" limit(
// expected-error@-1 {{expected value in expression}}

#embed "embed_parsing_errors.c" limit(xxx
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" limit(42
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" limit([
// expected-error@-1 {{invalid token at start of a preprocessor expression}}

#embed "embed_parsing_errors.c" limit([)
// expected-error@-1 {{invalid token at start of a preprocessor expression}}

#embed "embed_parsing_errors.c" limit(1/0)
// expected-error@-1 {{division by zero in preprocessor expression}}

#embed "embed_parsing_errors.c" clang::offset
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" clang::offset(
// expected-error@-1 {{expected value in expression}}

#embed "embed_parsing_errors.c" clang::offset(xxx
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" clang::offset(42
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" clang::offset([
// expected-error@-1 {{invalid token at start of a preprocessor expression}}

#embed "embed_parsing_errors.c" clang::offset([)
// expected-error@-1 {{invalid token at start of a preprocessor expression}}

#embed "embed_parsing_errors.c" clang::offset(1/0)
// expected-error@-1 {{division by zero in preprocessor expression}}

#embed "embed_parsing_errors.c" clang::offset 42
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" prefix
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" prefix(
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" prefix(xxx
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" prefix(1/0) // OK: emitted as tokens, not evaluated yet.
#embed "embed_parsing_errors.c" prefix(([{}])) // OK: delimiters balanced
#embed "embed_parsing_errors.c" prefix(([{)]})
// expected-error@-1 {{expected '}'}} expected-note@-1 {{to match this '{'}}
#embed "embed_parsing_errors.c" prefix(([{})})
// expected-error@-1 {{expected ']'}} expected-note@-1 {{to match this '['}}
#embed "embed_parsing_errors.c" prefix(([{}]})
// expected-error@-1 {{expected ')'}} expected-note@-1 {{to match this '('}}
#embed "embed_parsing_errors.c" prefix() // OK: tokens within parens are optional
#embed "embed_parsing_errors.c" prefix)
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" suffix
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" suffix(
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" suffix(xxx
// expected-error@-1 {{expected ')'}}

#embed "embed_parsing_errors.c" suffix(1/0) // OK: emitted as tokens, not evaluated yet.
#embed "embed_parsing_errors.c" suffix(([{}])) // OK: delimiters balanced
#embed "embed_parsing_errors.c" suffix(([{)]})
// expected-error@-1 {{expected '}'}} expected-note@-1 {{to match this '{'}}
#embed "embed_parsing_errors.c" suffix(([{})})
// expected-error@-1 {{expected ']'}} expected-note@-1 {{to match this '['}}
#embed "embed_parsing_errors.c" suffix(([{}]})
// expected-error@-1 {{expected ')'}} expected-note@-1 {{to match this '('}}
#embed "embed_parsing_errors.c" suffix() // OK: tokens within parens are optional
#embed "embed_parsing_errors.c" suffix)
// expected-error@-1 {{expected '('}}

#embed "embed_parsing_errors.c" if_empty(1/0) // OK: emitted as tokens, not evaluated yet.
#embed "embed_parsing_errors.c" if_empty(([{}])) // OK: delimiters balanced
#embed "embed_parsing_errors.c" if_empty(([{)]})
// expected-error@-1 {{expected '}'}} expected-note@-1 {{to match this '{'}}
#embed "embed_parsing_errors.c" if_empty(([{})})
// expected-error@-1 {{expected ']'}} expected-note@-1 {{to match this '['}}
#embed "embed_parsing_errors.c" if_empty(([{}]})
// expected-error@-1 {{expected ')'}} expected-note@-1 {{to match this '('}}
#embed "embed_parsing_errors.c" if_empty() // OK: tokens within parens are optional
#embed "embed_parsing_errors.c" if_empty)
// expected-error@-1 {{expected '('}}
};
