// RUN: %clang_cc1 -fsyntax-only -verify %s 

// Don't crash.

template<typename,typename=int,typename=int>struct basic_string;

typedef basic_string<char> string;



template<typename aT,typename,typename oc>
struct basic_string
{
int us;
basic_string(const aT*,const oc&a=int());

int _S_construct();

int _S_construct(int);

_S_construct(); // expected-error {{requires}}
};

template<typename _CharT,typename _Traits,typename _Alloc>
basic_string<_CharT,_Traits,_Alloc>::basic_string(const _CharT*,const _Alloc&)
:us(_S_construct)
{string a;}

struct runtime_error{runtime_error(string);};

struct system_error:runtime_error{ // expected-note {{to match}}
system_error():time_error("" // expected-error 3 {{expected}} expected-note {{to match}}
