// RUN: %clang_cc1 -fsyntax-only -verify=expected,garbage -DGARBAGE %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

// The undeclared template-name 'foo' starts a tentative parse that caches the
// rest of the file, so the second '__make_unsigned' token keeps its keyword
// kind after the first one has already been reverted to an identifier.
#ifdef GARBAGE
foo < bar()
// garbage-error@-1 {{no template named 'foo'}}
// garbage-error@-2 {{use of undeclared identifier 'bar'}}
// garbage-error@-3 {{expected '>'}}
// garbage-note@-4 {{to match this '<'}}
#endif

namespace N { // garbage-error {{expected unqualified-id}}
template <typename _Tp> struct __make_unsigned { typedef _Tp __type; };
// expected-warning@-1 {{keyword '__make_unsigned' will be made available as an identifier for the remainder of the translation unit}}
struct __make_unsigned<char> { typedef char __type; };
// expected-error@-1 {{template specialization requires 'template<>'}}
} // namespace N
