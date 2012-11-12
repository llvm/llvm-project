// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

class BaseEx {};
class Ex1: public BaseEx {};
typedef Ex1 Ex2;

void f();

void test()
try {}
catch (BaseEx &e) { f(); }
catch (Ex1 &e) { f(); } // expected-note {{for type class Ex1 &}}
catch (Ex2 &e) { f(); } // expected-warning {{exception of type Ex2 & will be caught by earlier handler}}

