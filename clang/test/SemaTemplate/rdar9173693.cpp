// RUN: %clang_cc1 -fsyntax-only -verify %s

template< bool C > struct assert { };
template< bool > struct assert_arg_pred_impl { }; // expected-note 2 {{declared here}}
template< typename Pred > assert<false> assert_not_arg( void (*)(Pred), typename assert_arg_pred<Pred>::type );
// expected-error@-1 {{did you mean 'assert_arg_pred_impl'}}
// expected-error@-2 {{template argument for non-type template parameter must be an expression}}
