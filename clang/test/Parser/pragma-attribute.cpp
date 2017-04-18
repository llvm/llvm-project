// RUN: %clang_cc1 -Wno-pragma-clang-attribute -verify -std=c++11 %s

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = function)

void function();

#pragma clang attribute pop

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any(variable(is_parameter), function))
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = variable(unless(is_parameter)))
#pragma clang attribute pop

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any(variable(unless(is_parameter))))
#pragma clang attribute pop

#pragma clang attribute push (__attribute__((abi_tag("a")))) // expected-error {{expected ','}}
#pragma clang attribute push (__attribute__((abi_tag("a"))) apply_to=function) // expected-error {{expected ','}}
#pragma clang attribute push (__attribute__((abi_tag("a"))) = function) // expected-error {{expected ','}}
#pragma clang attribute push (__attribute__((abi_tag("a"))) any(function)) // expected-error {{expected ','}}

#pragma clang attribute push (__attribute__((abi_tag("a"))) 22) // expected-error {{expected ','}}
#pragma clang attribute push (__attribute__((abi_tag("a"))) function) // expected-error {{expected ','}}
#pragma clang attribute push (__attribute__((abi_tag("a"))) (function)) // expected-error {{expected ','}}

#pragma clang attribute push(__attribute__((annotate("test"))), ) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), = any(function)) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), = function) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), any(function)) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), function) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply = any(function )) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), to = function) // expected-error {{expected attribute subject set specifier 'apply_to'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_only_to = function) // expected-error {{expected attribute subject set specifier 'apply_to'}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to any(function)) // expected-error {{expected '='}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to function) // expected-error {{expected '='}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to) // expected-error {{expected '='}}
#pragma clang attribute push (__attribute__((abi_tag("a"))), apply_to 41 (22)) // expected-error {{expected '='}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any) // expected-error {{expected '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any {) // expected-error {{expected '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any function) // expected-error {{expected '('}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = { function, enum }) // expected-error {{expected an identifier that corresponds to an attribute subject rule}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any(function ) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any(function, )) // expected-error {{expected an identifier that corresponds to an attribute subject rule}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, enum ) // expected-error {{expected ')'}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = () ) // expected-error {{expected an identifier that corresponds to an attribute subject rule}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( + ) ) // expected-error {{expected an identifier that corresponds to an attribute subject rule}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any()) // expected-error {{expected an identifier that corresponds to an attribute subject rule}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, 42 )) // expected-error {{expected an identifier that corresponds to an attribute subject rule}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( diag )) // expected-error {{unknown attribute subject rule 'diag'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( a )) // expected-error {{unknown attribute subject rule 'a'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, for)) // expected-error {{unknown attribute subject rule 'for'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function42, for )) // expected-error {{unknown attribute subject rule 'function42'}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any(hasType)) // expected-error {{expected '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = hasType) // expected-error {{expected '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = hasType(functionType)) // OK

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable( )) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable( ) )) // expected-error {{expected an identifier that corresponds to an attribute subject matcher sub-rule; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(is ) )) // expected-error {{unknown attribute subject matcher sub-rule 'is'; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(is_parameter, not) )) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable is_parameter )) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable ( ) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = variable (  // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, variable (is ()) )) // expected-error {{unknown attribute subject matcher sub-rule 'is'; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, variable (42) )) // expected-error {{expected an identifier that corresponds to an attribute subject matcher sub-rule; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, namespace("test") )) // expected-error {{expected an identifier that corresponds to an attribute subject matcher sub-rule; 'namespace' matcher does not support sub-rules}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, variable ("test" )) // expected-error {{expected ')'}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = enum(is_parameter)) // expected-error {{invalid use of attribute subject matcher sub-rule 'is_parameter'; 'enum' matcher does not support sub-rules}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any(enum(is_parameter))) // expected-error {{invalid use of attribute subject matcher sub-rule 'is_parameter'; 'enum' matcher does not support sub-rules}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any (function, variable (unless) )) // expected-error {{expected '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any (function, variable (unless() )) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any ( function, variable (unless(is)) )) // expected-error {{unknown attribute subject matcher sub-rule 'unless(is)'; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(unless(is_parameter, not)) )) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(unless(is_parameter), not) ) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable unless is_parameter )) // expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(unless is_parameter) )) // expected-error {{expected '('}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, variable (unless(42)) )) // expected-error {{expected an identifier that corresponds to an attribute subject matcher sub-rule; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, enum(unless("test")) )) // expected-error {{expected an identifier that corresponds to an attribute subject matcher sub-rule; 'enum' matcher does not support sub-rules}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, variable (unless(is_global)) )) // expected-error {{unknown attribute subject matcher sub-rule 'unless(is_global)'; 'variable' matcher supports the following sub-rules: 'is_thread_local', 'is_global', 'is_parameter', 'unless(is_parameter)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( enum(unless(is_parameter)) )) // expected-error {{invalid use of attribute subject matcher sub-rule 'unless(is_parameter)'; 'enum' matcher does not support sub-rules}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, function )) // expected-error {{duplicate attribute subject matcher 'function'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, function, function )) // expected-error 2 {{duplicate attribute subject matcher 'function'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( function, enum, function )) // expected-error {{duplicate attribute subject matcher 'function'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( enum, enum, function )) // expected-error {{duplicate attribute subject matcher 'enum'}}

#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(is_global), variable(is_global) )) // expected-error {{duplicate attribute subject matcher 'variable(is_global)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(is_global), function, variable(is_global), variable(is_global) )) // expected-error 2 {{duplicate attribute subject matcher 'variable(is_global)'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(unless(is_parameter)), variable(unless(is_parameter)) )) // expected-error {{duplicate attribute subject matcher 'variable(unless(is_parameter))'}}
#pragma clang attribute push(__attribute__((annotate("test"))), apply_to = any( variable(unless(is_parameter)), variable(unless(is_parameter)), enum, variable(unless(is_parameter)) )) // expected-error 2 {{duplicate attribute subject matcher 'variable(unless(is_parameter))'}}

#pragma clang attribute // expected-error {{expected 'push' or 'pop' after '#pragma clang attribute'}}
#pragma clang attribute 42 // expected-error {{expected 'push' or 'pop' after '#pragma clang attribute'}}
#pragma clang attribute pushpop // expected-error {{unexpected argument 'pushpop' to '#pragma clang attribute'; expected 'push' or 'pop'}}

#pragma clang attribute push // expected-error {{expected '('}}
#pragma clang attribute push ( // expected-error {{expected an attribute after '('}}
#pragma clang attribute push (__attribute__((annotate)) // expected-error {{expected ')'}}
#pragma clang attribute push () // expected-error {{expected an attribute after '('}}

#pragma clang attribute push (__attribute__((annotate("test"))), apply_to = function) () // expected-warning {{extra tokens at end of '#pragma clang attribute'}}
// expected-error@-1 {{expected unqualified-id}}
// expected-error@-2 {{unterminated '#pragma clang attribute push' at end of file}}

#pragma clang attribute pop () // expected-warning {{extra tokens at end of '#pragma clang attribute'}}

;

#pragma clang attribute push (__attribute__((42))) // expected-error {{expected identifier that represents an attribute name}}

#pragma clang attribute push (__attribute__((annotate)) foo) // expected-error {{expected ','}}
#pragma clang attribute push (__attribute__((annotate)), apply_to=function foo) // expected-error {{extra tokens after attribute in a '#pragma clang attribute push'}}

#pragma clang attribute push (__attribute__((availability(macos, foo=1))), apply_to=function) // expected-error {{'foo' is not an availability stage; use 'introduced', 'deprecated', or 'obsoleted'}}
// expected-error@-1 {{attribute 'availability' is not supported by '#pragma clang attribute'}}
#pragma clang attribute push (__attribute__((availability(macos, 1))), apply_to=function) // expected-error {{expected 'introduced', 'deprecated', or 'obsoleted'}}

#pragma clang attribute push (__attribute__((used)), apply_to=function) // expected-error {{attribute 'used' is not supported by '#pragma clang attribute'}}

void statementPragmasAndPragmaExpression() {
#pragma clang attribute push (__attribute__((annotate("hello"))), apply_to=variable)
#pragma clang attribute pop
int x = 0;
_Pragma("clang attribute push (__attribute__((annotate(\"hi\"))), apply_to = function)");

_Pragma("clang attribute push (__attribute__((annotate(\"hi\"))), apply_to = any(function(is_method ))"); // expected-error {{expected ')'}}
}

_Pragma("clang attribute pop");

#pragma clang attribute push (__attribute__((address_space(0))), apply_to=variable) // expected-error {{attribute 'address_space' is not supported by '#pragma clang attribute'}}

// Check support for CXX11 style attributes
#pragma clang attribute push ([[noreturn]], apply_to = any(function))
#pragma clang attribute pop

#pragma clang attribute push ([[clang::disable_tail_calls]], apply_to = function)
#pragma clang attribute pop

#pragma clang attribute push ([[gnu::abi_tag]], apply_to=any(function))
#pragma clang attribute pop

#pragma clang attribute push ([[clang::disable_tail_calls, noreturn]], apply_to = function) // expected-error {{more than one attribute specified in '#pragma clang attribute push'}}
#pragma clang attribute push ([[clang::disable_tail_calls, noreturn]]) // expected-error {{more than one attribute specified in '#pragma clang attribute push'}}

#pragma clang attribute push ([[gnu::abi_tag]], apply_to=namespace)
#pragma clang attribute pop

#pragma clang attribute push ([[fallthrough]], apply_to=function) // expected-error {{attribute 'fallthrough' is not supported by '#pragma clang attribute'}}
#pragma clang attribute push ([[clang::fallthrough]], apply_to=function) // expected-error {{attribute 'fallthrough' is not supported by '#pragma clang attribute'}}

#pragma clang attribute push ([[]], apply_to = function) // A noop

#pragma clang attribute push ([[noreturn ""]], apply_to=function) // expected-error {{expected ']'}}
#pragma clang attribute pop
#pragma clang attribute push ([[noreturn 42]]) // expected-error {{expected ']'}} expected-error {{expected ','}}

#pragma clang attribute push(__attribute__, apply_to=function) // expected-error {{expected '(' after 'attribute'}}
#pragma clang attribute push(__attribute__(), apply_to=function) // expected-error {{expected '(' after '('}}
#pragma clang attribute push(__attribute__(()), apply_to=function) // expected-error {{expected identifier that represents an attribute name}}
#pragma clang attribute push(__attribute__((annotate, apply_to=function))) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate("test"), apply_to=function))) // expected-error {{expected ')'}}
#pragma clang attribute push(__attribute__((annotate), apply_to=function)) // expected-error {{expected ')'}}

#pragma clang attribute push (42) // expected-error {{expected an attribute that is specified using the GNU, C++11 or '__declspec' syntax}}
#pragma clang attribute push (test) // expected-error {{expected an attribute that is specified using the GNU, C++11 or '__declspec' syntax}}
#pragma clang attribute push (annotate) // expected-error {{expected an attribute that is specified using the GNU, C++11 or '__declspec' syntax}}
// expected-note@-1 {{use the GNU '__attribute__' syntax}}
#pragma clang attribute push (annotate("test")) // expected-error {{expected an attribute that is specified using the GNU, C++11 or '__declspec' syntax}}
// expected-note@-1 {{use the GNU '__attribute__' syntax}}
