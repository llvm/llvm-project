// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template <typename T, typename Base> class EngineClassTypeInfo; // expected-note {{template is declared here}}
template <typename T> struct _ConcreteClassBase {};

struct _GLOBALSCOPE {};
template <typename T = _GLOBALSCOPE> struct _SCOPE {};

class Zone {
private:
  typedef _ConcreteClassBase<Zone> _ClassBase;
  static EngineClassTypeInfo<Zone, _ClassBase> _smTypeInfo;
  static EngineExportScope &__engineExportScope(); // expected-error {{unknown type name 'EngineExportScope'}}
};

EngineClassTypeInfo<Zone, Zone::_ClassBase>
    Zone::_smTypeInfo("Zone", &_SCOPE<__DeclScope>()(), 0); /* expected-error {{use of undeclared identifier '__DeclScope'}} \
                                                              expected-error {{implicit instantiation of undefined template 'EngineClassTypeInfo<Zone, _ConcreteClassBase<Zone>>'}} */
EngineExportScope &Zone::__engineExportScope() { return Zone::_smTypeInfo; } // expected-error {{unknown type name 'EngineExportScope'}}
