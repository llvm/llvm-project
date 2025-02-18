// RUN: %clang_cc1 -std=c++2b -verify -fsyntax-only %s


class A trivially_relocatable_if_eligible {};
class E final trivially_relocatable_if_eligible {};
class G trivially_relocatable_if_eligible final{};
class I trivially_relocatable_if_eligible trivially_relocatable_if_eligible final {}; // expected-error {{class already marked 'trivially_relocatable_if_eligible'}}
class trivially_relocatable_if_eligible trivially_relocatable_if_eligible {};
