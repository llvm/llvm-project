// RUN: %clang_cc1 -fsyntax-only -verify %s

// <rdar://problem/9173693>
template< bool C > struct assert { };
template< bool > struct assert_arg_pred_impl { }; // expected-note 3 {{declared here}}
template< typename Pred > assert<false> assert_not_arg( void (*)(Pred), typename assert_arg_pred<Pred>::type ); // expected-error 5 {{}}
