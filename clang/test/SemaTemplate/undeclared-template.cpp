// RUN: %clang_cc1 -std=c++2a -verify %s

template<typename T> notdefined<T::any> PR45207; // expected-error {{no template named 'notdefined'}}

// FIXME: We don't disambiguate this as an undeclared template-id even though there's nothing else it could be.
template<typename T> int var_template(notdefined<T::any>); // expected-error {{undeclared identifier 'notdefined'}} expected-error {{expected expression}}

// FIXME: We don't disambiguate this as a function template even though it can't be a variable template due to the ', int'.
template<typename T> int fn_template(notdefined<T::any>, int); // expected-error {{undeclared identifier 'notdefined'}} expected-error {{expected expression}} expected-error {{expected '('}}
