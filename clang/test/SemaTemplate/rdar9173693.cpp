// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/9173693>
template< bool C > struct assert { };
// FIXME: We diagnose the same problem multiple times here because we have no
// way to indicate in the token stream that we already tried to annotate a
// template-id and we failed.
template< bool > struct assert_arg_pred_impl { }; // expected-note 4 {{declared here}}
template< typename Pred > assert<false> assert_not_arg( void (*)(Pred), typename assert_arg_pred<Pred>::type ); // expected-error 6 {{}}
