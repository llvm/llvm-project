// RUN: %clang_cc1 -std=c++2c -verify=expected -fsyntax-only %s
// RUN: %clang_cc1 -std=c++11 -verify=expected,cxx11 -fsyntax-only %s



class A trivially_relocatable_if_eligible {};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
class E final trivially_relocatable_if_eligible {};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
class G trivially_relocatable_if_eligible final{};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
class I trivially_relocatable_if_eligible trivially_relocatable_if_eligible final {}; // expected-error {{class already marked 'trivially_relocatable_if_eligible'}}
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
class trivially_relocatable_if_eligible trivially_relocatable_if_eligible {};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}

class J replaceable_if_eligible{};
// cxx11-warning@-1 {{'replaceable_if_eligible' keyword is a C++2c extension}}
class K replaceable_if_eligible replaceable_if_eligible {}; // expected-error {{class already marked 'replaceable_if_eligible'}}
// cxx11-warning@-1 {{'replaceable_if_eligible' keyword is a C++2c extension}}
class replaceable_if_eligible replaceable_if_eligible {};
// cxx11-warning@-1 {{'replaceable_if_eligible' keyword is a C++2c extension}}
class L replaceable_if_eligible trivially_relocatable_if_eligible final {};
// cxx11-warning@-1 {{'replaceable_if_eligible' keyword is a C++2c extension}}
// cxx11-warning@-2 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
class M replaceable_if_eligible final trivially_relocatable_if_eligible {};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
// cxx11-warning@-2 {{'replaceable_if_eligible' keyword is a C++2c extension}}
class N final trivially_relocatable_if_eligible replaceable_if_eligible {};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
// cxx11-warning@-2 {{'replaceable_if_eligible' keyword is a C++2c extension}}
class O trivially_relocatable_if_eligible replaceable_if_eligible final {};
// cxx11-warning@-1 {{'trivially_relocatable_if_eligible' keyword is a C++2c extension}}
// cxx11-warning@-2 {{'replaceable_if_eligible' keyword is a C++2c extension}}
