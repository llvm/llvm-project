// RUN: %clang_cc1 -std=c23 %s -E -verify

// Test the parsing behavior for __has_embed and all of its parameters to ensure we
// recover from failures gracefully.

// expected-error@+2 {{missing '(' after '__has_embed'}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed
#endif

// expected-error@+3 {{expected '>'}} \
   expected-note@+3 {{to match this '<'}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed(<)
#endif

// expected-error@+3 {{expected "FILENAME" or <FILENAME>}} \
   expected-warning@+3 {{missing terminating '"' character}} \
   expected-error@+3 {{invalid token at start of a preprocessor expression}}
#if __has_embed(")
#endif

// expected-error@+2 {{missing '(' after '__has_embed'}} \
   expected-error@+2 {{token is not a valid binary operator in a preprocessor subexpression}}
#if __has_embed file.txt
#endif

// OK, no diagnostic for an unknown embed parameter.
#if __has_embed("media/empty" xxx)
#endif

// expected-error@+2 {{expected identifier}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" xxx::)
#endif

// OK, no diagnostic for an unknown embed parameter.
#if __has_embed("media/empty" xxx::xxx)
#endif

// expected-error@+2 {{expected identifier}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" xxx::42)
#endif

// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" limit)
#endif

// We get the same diagnostic twice intentionally. The first one is because of
// the missing value within limit() and the second one is because the #if does
// not resolve to a value due to the earlier error.
// expected-error@+1 2 {{expected value in expression}}
#if __has_embed("media/empty" limit()
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" limit(xxx)
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" limit(42)
#endif

// expected-error@+2 {{invalid token at start of a preprocessor expression}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" limit([)
#endif

// expected-error@+2 {{invalid token at start of a preprocessor expression}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" limit([))
#endif

// expected-error@+2 {{division by zero in preprocessor expression}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" limit(1/0))
#endif

// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" clang::offset)
#endif

// We get the same diagnostic twice intentionally. The first one is because of
// the missing value within clang::offset() and the second one is because the
// #if does not resolve to a value due to the earlier error.
// expected-error@+1 2 {{expected value in expression}}
#if __has_embed("media/empty" clang::offset()
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" clang::offset(xxx)
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" clang::offset(42)
#endif

// expected-error@+2 {{invalid token at start of a preprocessor expression}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" clang::offset([)
#endif

// expected-error@+2 {{invalid token at start of a preprocessor expression}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" clang::offset([))
#endif

// expected-error@+2 {{division by zero in preprocessor expression}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" clang::offset(1/0))
#endif

// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" clang::offset 42)
#endif

// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" prefix)
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" prefix()
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" prefix(xxx)
#endif

#if __has_embed("media/empty" prefix(1/0)) // OK: emitted as tokens, not evaluated yet.
#endif
#if __has_embed("media/empty" prefix(([{}]))) // OK: delimiters balanced
#endif
// expected-error@+3 {{expected '}'}} \
   expected-note@+3 {{to match this '{'}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/empty" prefix(([{)]}))
#endif
// expected-error@+3 {{expected ']'}} \
   expected-note@+3 {{to match this '['}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/empty" prefix(([{})}))
#endif
// expected-error@+3 {{expected ')'}} \
   expected-note@+3 {{to match this '('}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/empty" prefix(([{}]}))
#endif
#if __has_embed("media/empty" prefix()) // OK: tokens within parens are optional
#endif
// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" prefix))
#endif

// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" suffix)
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" suffix()
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed("media/empty" suffix(xxx)
#endif

#if __has_embed("media/empty" suffix(1/0)) // OK: emitted as tokens, not evaluated yet.
#endif
#if __has_embed("media/empty" suffix(([{}]))) // OK: delimiters balanced
#endif
// expected-error@+3 {{expected '}'}} \
   expected-note@+3 {{to match this '{'}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/empty" suffix(([{)]}))
#endif
// expected-error@+3 {{expected ']'}} \
   expected-note@+3 {{to match this '['}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/empty" suffix(([{})}))
#endif
// expected-error@+3 {{expected ')'}} \
   expected-note@+3 {{to match this '('}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/empty" suffix(([{}]}))
#endif
#if __has_embed("media/empty" suffix()) // OK: tokens within parens are optional
#endif
// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/empty" suffix))
#endif

#if __has_embed("media/art.txt" if_empty(1/0)) // OK: emitted as tokens, not evaluated yet.
#endif
#if __has_embed("media/art.txt" if_empty(([{}]))) // OK: delimiters balanced
#endif
// expected-error@+3 {{expected '}'}} \
   expected-note@+3 {{to match this '{'}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/art.txt" if_empty(([{)]}))
#endif
// expected-error@+3 {{expected ']'}} \
   expected-note@+3 {{to match this '['}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/art.txt" if_empty(([{})}))
#endif
// expected-error@+3 {{expected ')'}} \
   expected-note@+3 {{to match this '('}} \
   expected-error@+3 {{expected value in expression}}
#if __has_embed("media/art.txt" if_empty(([{}]}))
#endif
#if __has_embed("media/art.txt" if_empty()) // OK: tokens within parens are optional
#endif
// expected-error@+2 {{expected '('}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed("media/art.txt" if_empty))
#endif

// expected-error@+2 {{invalid value '-1'; must be positive}} \
   expected-error@+2 {{expected value in expression}}
#if __has_embed (__FILE__ limit(-1))
#endif

// expected-error@+2 {{invalid value '-100000000000000000'; must be positive}}\
   expected-error@+2 {{expected value in expression}}
#if __has_embed (__FILE__ limit(-100000000000000000)) != __STDC_EMBED_NOT_FOUND__
#endif

#if __has_embed("") // expected-error {{empty filename}}
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed (__FILE__  foo limit(1)
#endif

//--- test3.c
// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed (__FILE__  foo
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed ("a" foo()
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed ("a" bar() foo
#endif

// expected-error@+3 {{missing ')' after '__has_embed'}} \
   expected-error@+3 {{expected value in expression}} \
   expected-note@+3 {{to match this '('}}
#if __has_embed (__FILE__ limit(1) foo
int a = __has_embed (__FILE__);
#endif
