// RUN: %clang_cc1 -triple i686-win32             -fsyntax-only -fms-extensions -verify -std=c++11 -Wunsupported-dll-base-class-template -DMS %s
// RUN: %clang_cc1 -triple x86_64-win32           -fsyntax-only -fms-extensions -verify -std=c++1y -Wunsupported-dll-base-class-template -DMS %s
// RUN: %clang_cc1 -triple i686-mingw32           -fsyntax-only -fms-extensions -verify -std=c++1y -Wunsupported-dll-base-class-template -DGNU %s
// RUN: %clang_cc1 -triple x86_64-mingw32         -fsyntax-only -fms-extensions -verify -std=c++11 -Wunsupported-dll-base-class-template -DGNU %s
// RUN: %clang_cc1 -triple x86_64-mingw32         -fsyntax-only -fms-extensions -verify -std=c++17 -Wunsupported-dll-base-class-template -DGNU %s
// RUN: %clang_cc1 -triple i686-windows-itanium   -fsyntax-only -fms-extensions -verify -std=c++11 -Wunsupported-dll-base-class-template -DWI %s
// RUN: %clang_cc1 -triple x86_64-windows-itanium -fsyntax-only -fms-extensions -verify -std=c++17 -Wunsupported-dll-base-class-template -DWI %s
// RUN: %clang_cc1 -triple x86_64-scei-ps4        -fsyntax-only -fdeclspec      -verify -std=c++11 -Wunsupported-dll-base-class-template -DWI %s
// RUN: %clang_cc1 -triple x86_64-scei-ps4        -fsyntax-only -fdeclspec      -verify -std=c++17 -Wunsupported-dll-base-class-template -DWI %s
// RUN: %clang_cc1 -triple x86_64-sie-ps5         -fsyntax-only -fdeclspec      -verify -std=c++17 -Wunsupported-dll-base-class-template -DWI %s

// Helper structs to make templates more expressive.
struct ImplicitInst_Imported {};
struct ExplicitDecl_Imported {};
struct ExplicitInst_Imported {};
struct ExplicitSpec_Imported {};
struct ExplicitSpec_Def_Imported {};
struct ExplicitSpec_InlineDef_Imported {};
struct ExplicitSpec_NotImported {};
namespace { struct Internal {}; }


// Invalid usage.
__declspec(dllimport) typedef int typedef1;
// expected-warning@-1{{'dllimport' attribute only applies to functions, variables, classes, and Objective-C interfaces}}
typedef __declspec(dllimport) int typedef2;
// expected-warning@-1{{'dllimport' attribute only applies to}}
typedef int __declspec(dllimport) typedef3;
// expected-warning@-1{{'dllimport' attribute only applies to}}
typedef __declspec(dllimport) void (*FunTy)();
// expected-warning@-1{{'dllimport' attribute only applies to}}
enum __declspec(dllimport) Enum {};
// expected-warning@-1{{'dllimport' attribute only applies to}}
#if __has_feature(cxx_strong_enums)
enum class __declspec(dllimport) EnumClass {};
// expected-warning@-1{{'dllimport' attribute only applies to}}
#endif



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
__declspec(dllimport) extern int ExternGlobalDecl;

// dllimport implies a declaration.
__declspec(dllimport) int GlobalDecl;
int **__attribute__((dllimport))* GlobalDeclChunkAttr;
int GlobalDeclAttr __attribute__((dllimport));

// Not allowed on definitions.
__declspec(dllimport) extern int ExternGlobalInit = 1; // expected-error{{definition of dllimport data}}
__declspec(dllimport) int GlobalInit1 = 1; // expected-error{{definition of dllimport data}}
int __declspec(dllimport) GlobalInit2 = 1; // expected-error{{definition of dllimport data}}

// Declare, then reject definition.
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
__declspec(dllimport) extern int ExternGlobalDeclInit; // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-warning@+4{{'ExternGlobalDeclInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'ExternGlobalDeclInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int ExternGlobalDeclInit = 1;

#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
__declspec(dllimport) int GlobalDeclInit; // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-warning@+4{{'GlobalDeclInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'GlobalDeclInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int GlobalDeclInit = 1;

#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
int *__attribute__((dllimport)) GlobalDeclChunkAttrInit; // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-warning@+4{{'GlobalDeclChunkAttrInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'GlobalDeclChunkAttrInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int *GlobalDeclChunkAttrInit = 0;

#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
int GlobalDeclAttrInit __attribute__((dllimport)); // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-warning@+4{{'GlobalDeclAttrInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'GlobalDeclAttrInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int GlobalDeclAttrInit = 1;

// Redeclarations
__declspec(dllimport) extern int GlobalRedecl1;
__declspec(dllimport) extern int GlobalRedecl1;

__declspec(dllimport) int GlobalRedecl2a;
__declspec(dllimport) int GlobalRedecl2a;

int *__attribute__((dllimport)) GlobalRedecl2b;
int *__attribute__((dllimport)) GlobalRedecl2b;

int GlobalRedecl2c __attribute__((dllimport));
int GlobalRedecl2c __attribute__((dllimport));

__declspec(dllimport) extern int GlobalRedecl3; // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
                      extern int GlobalRedecl3; // expected-warning{{'GlobalRedecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

                      extern int GlobalRedecl4; // expected-note{{previous declaration is here}}
__declspec(dllimport) extern int GlobalRedecl4; // expected-warning{{redeclaration of 'GlobalRedecl4' should not add 'dllimport' attribute}}

extern "C" {
                      extern int GlobalRedecl5; // expected-note{{previous declaration is here}}
__declspec(dllimport) extern int GlobalRedecl5; // expected-warning{{redeclaration of 'GlobalRedecl5' should not add 'dllimport' attribute}}
}

// External linkage is required.
__declspec(dllimport) static int StaticGlobal; // expected-error{{'StaticGlobal' must have external linkage when declared 'dllimport'}}
__declspec(dllimport) Internal InternalTypeGlobal; // expected-error{{'InternalTypeGlobal' must have external linkage when declared 'dllimport'}}
#ifndef MS
namespace    { __declspec(dllimport) int InternalGlobal; } // expected-error{{'(anonymous namespace)::InternalGlobal' must have external linkage when declared 'dllimport'}}
#endif
namespace ns { __declspec(dllimport) int ExternalGlobal; }

__declspec(dllimport) auto InternalAutoTypeGlobal = Internal(); // expected-error{{'InternalAutoTypeGlobal' must have external linkage when declared 'dllimport'}}
                                                                // expected-error@-1{{definition of dllimport data}}

// Thread local variables are invalid.
__declspec(dllimport) __thread int ThreadLocalGlobal; // expected-error{{'ThreadLocalGlobal' cannot be thread local when declared 'dllimport'}}
// This doesn't work on MinGW, because there, dllimport on the inline function is ignored.
#ifndef GNU
inline void __declspec(dllimport) ImportedInlineWithThreadLocal() {
  static __thread int OK; // no-error
}
#endif

// Import in local scope.
__declspec(dllimport) float LocalRedecl1; // expected-note{{previous declaration is here}}
__declspec(dllimport) float LocalRedecl2; // expected-note{{previous declaration is here}}
__declspec(dllimport) float LocalRedecl3; // expected-note{{previous declaration is here}}
void functionScope() {
  __declspec(dllimport) int LocalRedecl1; // expected-error{{redeclaration of 'LocalRedecl1' with a different type: 'int' vs 'float'}}
  int *__attribute__((dllimport)) LocalRedecl2; // expected-error{{redeclaration of 'LocalRedecl2' with a different type: 'int *' vs 'float'}}
  int LocalRedecl3 __attribute__((dllimport)); // expected-error{{redeclaration of 'LocalRedecl3' with a different type: 'int' vs 'float'}}

  __declspec(dllimport)        int LocalVarDecl;
  __declspec(dllimport)        int LocalVarDef = 1; // expected-error{{definition of dllimport data}}
  __declspec(dllimport) extern int ExternLocalVarDecl;
  __declspec(dllimport) extern int ExternLocalVarDef = 1; // expected-error{{definition of dllimport data}}
  __declspec(dllimport) static int StaticLocalVar; // expected-error{{'StaticLocalVar' must have external linkage when declared 'dllimport'}}
}



//===----------------------------------------------------------------------===//
// Variable templates
//===----------------------------------------------------------------------===//
#if __has_feature(cxx_variable_templates)

// Import declaration.
template<typename T> __declspec(dllimport) extern int ExternVarTmplDecl;

// dllimport implies a declaration.
template<typename T> __declspec(dllimport) int VarTmplDecl;

// Not allowed on definitions.
template<typename T> __declspec(dllimport) extern int ExternVarTmplInit = 1; // expected-error{{definition of dllimport data}}
template<typename T> __declspec(dllimport) int VarTmplInit1 = 1; // expected-error{{definition of dllimport data}}
template<typename T> int __declspec(dllimport) VarTmplInit2 = 1; // expected-error{{definition of dllimport data}}

// Declare, then reject definition.
#ifdef GNU
// expected-note@+3{{previous attribute is here}}
#endif
template <typename T>
__declspec(dllimport) extern int ExternVarTmplDeclInit; // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-warning@+5{{'ExternVarTmplDeclInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+3{{'ExternVarTmplDeclInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
template <typename T>
int ExternVarTmplDeclInit = 1;

#ifdef GNU
// expected-note@+3{{previous attribute is here}}
#endif
template <typename T>
__declspec(dllimport) int VarTmplDeclInit; // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-warning@+5{{'VarTmplDeclInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+3{{'VarTmplDeclInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
template <typename T>
int VarTmplDeclInit = 1;

// Redeclarations
template<typename T> __declspec(dllimport) extern int VarTmplRedecl1;
template<typename T> __declspec(dllimport) extern int VarTmplRedecl1;

template<typename T> __declspec(dllimport) int VarTmplRedecl2;
template<typename T> __declspec(dllimport) int VarTmplRedecl2;

template<typename T> __declspec(dllimport) extern int VarTmplRedecl3; // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
template<typename T>                       extern int VarTmplRedecl3; // expected-warning{{'VarTmplRedecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

template<typename T>                       extern int VarTmplRedecl4; // expected-note{{previous declaration is here}}
template<typename T> __declspec(dllimport) extern int VarTmplRedecl4; // expected-error{{redeclaration of 'VarTmplRedecl4' cannot add 'dllimport' attribute}}

// External linkage is required.
template<typename T> __declspec(dllimport) static int StaticVarTmpl; // expected-error{{'StaticVarTmpl' must have external linkage when declared 'dllimport'}}
template<typename T> __declspec(dllimport) Internal InternalTypeVarTmpl; // expected-error{{'InternalTypeVarTmpl' must have external linkage when declared 'dllimport'}}
#ifndef MS
namespace    { template<typename T> __declspec(dllimport) int InternalVarTmpl; } // expected-error{{'(anonymous namespace)::InternalVarTmpl' must have external linkage when declared 'dllimport'}}
#endif
namespace ns { template<typename T> __declspec(dllimport) int ExternalVarTmpl; }

template<typename T> __declspec(dllimport) auto InternalAutoTypeVarTmpl = Internal(); // expected-error{{definition of dllimport data}} // expected-error{{'InternalAutoTypeVarTmpl' must have external linkage when declared 'dllimport'}}


template<typename T> int VarTmpl;
template<typename T> __declspec(dllimport) int ImportedVarTmpl;

// Import implicit instantiation of an imported variable template.
int useVarTmpl() { return ImportedVarTmpl<ImplicitInst_Imported>; }

// Import explicit instantiation declaration of an imported variable template.
extern template int ImportedVarTmpl<ExplicitDecl_Imported>;

// An explicit instantiation definition of an imported variable template cannot
// be imported because the template must be defined which is illegal.

// Import specialization of an imported variable template.
template<> __declspec(dllimport) int ImportedVarTmpl<ExplicitSpec_Imported>;
template<> __declspec(dllimport) int ImportedVarTmpl<ExplicitSpec_Def_Imported> = 1; // expected-error{{definition of dllimport data}}

// Not importing specialization of an imported variable template without
// explicit dllimport.
template<> int ImportedVarTmpl<ExplicitSpec_NotImported>;


// Import explicit instantiation declaration of a non-imported variable template.
extern template __declspec(dllimport) int VarTmpl<ExplicitDecl_Imported>;

// Import explicit instantiation definition of a non-imported variable template.
template __declspec(dllimport) int VarTmpl<ExplicitInst_Imported>;

// Import specialization of a non-imported variable template.
template<> __declspec(dllimport) int VarTmpl<ExplicitSpec_Imported>;
template<> __declspec(dllimport) int VarTmpl<ExplicitSpec_Def_Imported> = 1; // expected-error{{definition of dllimport data}}

#endif // __has_feature(cxx_variable_templates)


//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration. Check different placements.
__attribute__((dllimport)) void decl1A(); // Validation check with __attribute__
__declspec(dllimport)      void decl1B();

void __attribute__((dllimport)) decl2A();
void __declspec(dllimport)      decl2B();

// Not allowed on function definitions.
__declspec(dllimport) void def() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}

// extern  "C"
extern "C" __declspec(dllimport) void externC();

// Import inline function.
#ifdef GNU
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
#endif
__declspec(dllimport) inline void inlineFunc1() {}
inline void __attribute__((dllimport)) inlineFunc2() {}

#ifdef GNU
// expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
__declspec(dllimport) inline void inlineDecl();
                             void inlineDecl() {}

__declspec(dllimport) void inlineDef();
#ifdef GNU
// expected-warning@+2{{'inlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
               inline void inlineDef() {}

// Redeclarations
__declspec(dllimport) void redecl1();
__declspec(dllimport) void redecl1();

__declspec(dllimport) void redecl2(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
                      void redecl2(); // expected-warning{{'redecl2' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

#ifdef GNU
                      // expected-note@+2{{previous attribute is here}}
#endif
                      __declspec(dllimport) void redecl3(); // expected-note{{previous declaration is here}}
                      // NB: Both MSVC and Clang issue a warning and make redecl3 dllexport.
#if defined(MS) || defined(WI)
                      // expected-warning@+4{{'redecl3' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                      // expected-warning@+2{{'redecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
                      void redecl3() {}

                      void redecl4(); // expected-note{{previous declaration is here}}
__declspec(dllimport) void redecl4(); // expected-warning{{redeclaration of 'redecl4' should not add 'dllimport' attribute}}

extern "C" {
                      void redecl5(); // expected-note{{previous declaration is here}}
__declspec(dllimport) void redecl5(); // expected-warning{{redeclaration of 'redecl5' should not add 'dllimport' attribute}}
}

#if defined(MS) || defined(WI)
                      void redecl6(); // expected-note{{previous declaration is here}}
__declspec(dllimport) inline void redecl6() {} // expected-warning{{redeclaration of 'redecl6' should not add 'dllimport' attribute}}
#else
                      void redecl6();
__declspec(dllimport) inline void redecl6() {} // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif

// Friend functions
struct FuncFriend {
  friend __declspec(dllimport) void friend1();
  friend __declspec(dllimport) void friend2(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
  friend __declspec(dllimport) void friend3(); // expected-note{{previous declaration is here}}
  friend                       void friend4(); // expected-note{{previous declaration is here}}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}}
#endif
  friend                       void friend5();
};
__declspec(dllimport) void friend1();
                      void friend2(); // expected-warning{{'friend2' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#if defined(MS) || defined(WI)
                      // expected-warning@+4{{'friend3' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                      // expected-warning@+2{{'friend3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
                      void friend3() {}
__declspec(dllimport) void friend4(); // expected-warning{{redeclaration of 'friend4' should not add 'dllimport' attribute}}
#if defined(MS) || defined(WI)
__declspec(dllimport) inline void friend5() {} // expected-warning{{redeclaration of 'friend5' should not add 'dllimport' attribute}}
#else
__declspec(dllimport) inline void friend5() {} // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif


void __declspec(dllimport) friend6(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
void __declspec(dllimport) friend7();
struct FuncFriend2 {
  friend void friend6(); // expected-warning{{'friend6' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
  friend void ::friend7();
};

// Implicit declarations can be redeclared with dllimport.
__declspec(dllimport) void* operator new(__SIZE_TYPE__ n);

// External linkage is required.
__declspec(dllimport) static int staticFunc(); // expected-error{{'staticFunc' must have external linkage when declared 'dllimport'}}
__declspec(dllimport) Internal internalRetFunc(); // expected-error{{'internalRetFunc' must have external linkage when declared 'dllimport'}}
namespace    { __declspec(dllimport) void internalFunc(); } // expected-error{{'(anonymous namespace)::internalFunc' must have external linkage when declared 'dllimport'}}
namespace ns { __declspec(dllimport) void externalFunc(); }

// Import deleted functions.
// FIXME: Deleted functions are definitions so a missing inline is diagnosed
// here which is irrelevant. But because the delete keyword is parsed later
// there is currently no straight-forward way to avoid this diagnostic.
__declspec(dllimport) void deletedFunc() = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}} expected-error{{dllimport cannot be applied to non-inline function definition}}
#if defined(MS) || defined(WI)
__declspec(dllimport) inline void deletedInlineFunc() = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
#else
__declspec(dllimport) inline void deletedInlineFunc() = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif



//===----------------------------------------------------------------------===//
// Function templates
//===----------------------------------------------------------------------===//

// Import function template declaration. Check different placements.
template<typename T> __declspec(dllimport) void funcTmplDecl1();
template<typename T> void __declspec(dllimport) funcTmplDecl2();

// Import function template definition.
template<typename T> __declspec(dllimport) void funcTmplDef() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}

// Import inline function template.
#ifdef GNU // MinGW always ignores dllimport on inline functions.

template<typename T> __declspec(dllimport) inline void inlineFuncTmpl1() {} // expected-warning{{'dllimport' attribute ignored on inline function}}
template<typename T> inline void __attribute__((dllimport)) inlineFuncTmpl2() {} // expected-warning{{'dllimport' attribute ignored on inline function}}

template<typename T> __declspec(dllimport) inline void inlineFuncTmplDecl(); // expected-warning{{'dllimport' attribute ignored on inline function}}
template<typename T>                              void inlineFuncTmplDecl() {}

template<typename T> __declspec(dllimport) void inlineFuncTmplDef();
template<typename T>                inline void inlineFuncTmplDef() {} // expected-warning{{'inlineFuncTmplDef' redeclared inline; 'dllimport' attribute ignored}}

#else // MSVC drops dllimport when the function template is redeclared without it. (It doesn't warn, but we do.)

template<typename T> __declspec(dllimport) inline void inlineFuncTmpl1() {}
template<typename T> inline void __attribute__((dllimport)) inlineFuncTmpl2() {}

template<typename T> __declspec(dllimport) inline void inlineFuncTmplDecl(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
template<typename T>                              void inlineFuncTmplDecl() {} // expected-warning{{'inlineFuncTmplDecl' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

template<typename T> __declspec(dllimport) void inlineFuncTmplDef(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
template<typename T>                inline void inlineFuncTmplDef() {} // expected-warning{{'inlineFuncTmplDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif

// Redeclarations
template<typename T> __declspec(dllimport) void funcTmplRedecl1();
template<typename T> __declspec(dllimport) void funcTmplRedecl1();

template<typename T> __declspec(dllimport) void funcTmplRedecl2(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
template<typename T>                       void funcTmplRedecl2(); // expected-warning{{'funcTmplRedecl2' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

template<typename T> __declspec(dllimport) void funcTmplRedecl3(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
template<typename T>                       void funcTmplRedecl3() {} // expected-warning{{'funcTmplRedecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

template<typename T>                       void funcTmplRedecl4(); // expected-note{{previous declaration is here}}
template<typename T> __declspec(dllimport) void funcTmplRedecl4(); // expected-error{{redeclaration of 'funcTmplRedecl4' cannot add 'dllimport' attribute}}

#ifdef MS
template<typename T>                       void funcTmplRedecl5(); // expected-note{{previous declaration is here}}
template<typename T> __declspec(dllimport) inline void funcTmplRedecl5() {} // expected-error{{redeclaration of 'funcTmplRedecl5' cannot add 'dllimport' attribute}}
#endif

// Function template friends
struct FuncTmplFriend {
  template<typename T> friend __declspec(dllimport) void funcTmplFriend1();
  template<typename T> friend __declspec(dllimport) void funcTmplFriend2(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
  template<typename T> friend __declspec(dllimport) void funcTmplFriend3(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
  template<typename T> friend                       void funcTmplFriend4(); // expected-note{{previous declaration is here}}
#ifdef GNU
// expected-warning@+4{{'dllimport' attribute ignored on inline function}}
#else
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename T> friend __declspec(dllimport) inline void funcTmplFriend5();
};
template<typename T> __declspec(dllimport) void funcTmplFriend1();
template<typename T>                       void funcTmplFriend2(); // expected-warning{{'funcTmplFriend2' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T>                       void funcTmplFriend3() {} // expected-warning{{'funcTmplFriend3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T> __declspec(dllimport) void funcTmplFriend4(); // expected-error{{redeclaration of 'funcTmplFriend4' cannot add 'dllimport' attribute}}
#if defined(MS) || defined(WI)
// expected-warning@+2{{'funcTmplFriend5' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
template<typename T>                       inline void funcTmplFriend5() {}

// External linkage is required.
template<typename T> __declspec(dllimport) static int staticFuncTmpl(); // expected-error{{'staticFuncTmpl' must have external linkage when declared 'dllimport'}}
template<typename T> __declspec(dllimport) Internal internalRetFuncTmpl(); // expected-error{{'internalRetFuncTmpl' must have external linkage when declared 'dllimport'}}
namespace    { template<typename T> __declspec(dllimport) void internalFuncTmpl(); } // expected-error{{'(anonymous namespace)::internalFuncTmpl' must have external linkage when declared 'dllimport'}}
namespace ns { template<typename T> __declspec(dllimport) void externalFuncTmpl(); }


template<typename T> void funcTmpl() {}
template<typename T> inline void inlineFuncTmpl() {}
template<typename T> __declspec(dllimport) void importedFuncTmplDecl();
#ifdef GNU
// expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template<typename T> __declspec(dllimport) inline void importedFuncTmpl() {}

// Import implicit instantiation of an imported function template.
void useFunTmplDecl() { importedFuncTmplDecl<ImplicitInst_Imported>(); }
void useFunTmplDef() { importedFuncTmpl<ImplicitInst_Imported>(); }

// Import explicit instantiation declaration of an imported function template.
extern template void importedFuncTmpl<ExplicitDecl_Imported>();

// Import explicit instantiation definition of an imported function template.
// NB: MSVC fails this instantiation without explicit dllimport which is most
// likely a bug because an implicit instantiation is accepted.
template void importedFuncTmpl<ExplicitInst_Imported>();

// Import specialization of an imported function template. A definition must be
// declared inline.
template<> __declspec(dllimport) void importedFuncTmpl<ExplicitSpec_Imported>();
template<> __declspec(dllimport) void importedFuncTmpl<ExplicitSpec_Def_Imported>() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}
#ifdef MS
template<> __declspec(dllimport) inline void importedFuncTmpl<ExplicitSpec_InlineDef_Imported>() {}
#endif

// Not importing specialization of an imported function template without
// explicit dllimport.
template<> void importedFuncTmpl<ExplicitSpec_NotImported>() {}


// Import explicit instantiation declaration of a non-imported function template.
extern template __declspec(dllimport) void funcTmpl<ExplicitDecl_Imported>();
#ifdef GNU
// expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
extern template __declspec(dllimport) void inlineFuncTmpl<ExplicitDecl_Imported>();

// Import explicit instantiation definition of a non-imported function template.
template __declspec(dllimport) void funcTmpl<ExplicitInst_Imported>();
#ifdef GNU
// expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template __declspec(dllimport) void inlineFuncTmpl<ExplicitInst_Imported>();

// Import specialization of a non-imported function template. A definition must
// be declared inline.
template<> __declspec(dllimport) void funcTmpl<ExplicitSpec_Imported>();
template<> __declspec(dllimport) void funcTmpl<ExplicitSpec_Def_Imported>() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}
#ifdef GNU
// expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template<> __declspec(dllimport) inline void funcTmpl<ExplicitSpec_InlineDef_Imported>() {}


//===----------------------------------------------------------------------===//
// Class members
//===----------------------------------------------------------------------===//

// Import individual members of a class.
struct ImportMembers {
  struct Nested {
    __declspec(dllimport) void normalDecl();
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
    __declspec(dllimport) void normalDef(); // expected-note{{previous declaration is here}}
  };

#ifdef GNU
// expected-note@+5{{previous attribute is here}}
// expected-warning@+5{{'dllimport' attribute ignored on inline function}}
// expected-warning@+6{{'dllimport' attribute ignored on inline function}}
#endif
  __declspec(dllimport)                void normalDecl();
  __declspec(dllimport) void normalDef(); // expected-note{{previous declaration is here}}
  __declspec(dllimport)                void normalInclass() {}
  __declspec(dllimport)                void normalInlineDef();
  __declspec(dllimport)         inline void normalInlineDecl();
#ifdef GNU
// expected-note@+5{{previous attribute is here}}
// expected-warning@+5{{'dllimport' attribute ignored on inline function}}
// expected-warning@+6{{'dllimport' attribute ignored on inline function}}
#endif
  __declspec(dllimport) virtual        void virtualDecl();
  __declspec(dllimport) virtual void virtualDef(); // expected-note{{previous declaration is here}}
  __declspec(dllimport) virtual        void virtualInclass() {}
  __declspec(dllimport) virtual        void virtualInlineDef();
  __declspec(dllimport) virtual inline void virtualInlineDecl();
#ifdef GNU
// expected-note@+5{{previous attribute is here}}
// expected-warning@+5{{'dllimport' attribute ignored on inline function}}
// expected-warning@+6{{'dllimport' attribute ignored on inline function}}
#endif
  __declspec(dllimport) static         void staticDecl();
  __declspec(dllimport) static void staticDef(); // expected-note{{previous declaration is here}}
  __declspec(dllimport) static         void staticInclass() {}
  __declspec(dllimport) static         void staticInlineDef();
  __declspec(dllimport) static  inline void staticInlineDecl();

protected:
  __declspec(dllimport)                void protectedDecl();
private:
  __declspec(dllimport)                void privateDecl();
public:

  __declspec(dllimport)                int  Field; // expected-warning{{'dllimport' attribute only applies to}}
  __declspec(dllimport) static         int  StaticField;
  __declspec(dllimport) static         int  StaticFieldDef; // expected-note{{attribute is here}}
  __declspec(dllimport) static  const  int  StaticConstField;
  __declspec(dllimport) static  const  int  StaticConstFieldDef; // expected-note{{attribute is here}}
  __declspec(dllimport) static  const  int  StaticConstFieldEqualInit = 1;
  __declspec(dllimport) static  const  int  StaticConstFieldBraceInit{1};
  __declspec(dllimport) constexpr static int ConstexprField = 1;
#if __cplusplus < 201703L && !defined(MS)
  // expected-note@+2{{attribute is here}}
#endif
  __declspec(dllimport) constexpr static int ConstexprFieldDef = 1;
};

#if defined(MS) || defined(WI)
// expected-warning@+4{{'ImportMembers::Nested::normalDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                                                                                 // expected-warning@+2{{'ImportMembers::Nested::normalDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
void ImportMembers::Nested::normalDef() {}
#if defined(MS) || defined(WI)
// expected-warning@+4{{'ImportMembers::normalDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                                                                                 // expected-warning@+2{{'ImportMembers::normalDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
void ImportMembers::normalDef() {}
#ifdef GNU
// expected-warning@+2{{'ImportMembers::normalInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
inline void ImportMembers::normalInlineDef() {}
       void ImportMembers::normalInlineDecl() {}
#if defined(MS) || defined(WI)
       // expected-warning@+4{{'ImportMembers::virtualDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                                                                                 // expected-warning@+2{{'ImportMembers::virtualDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
       void ImportMembers::virtualDef() {}
#ifdef GNU
// expected-warning@+2{{'ImportMembers::virtualInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
inline void ImportMembers::virtualInlineDef() {}
       void ImportMembers::virtualInlineDecl() {}
#if defined(MS) || defined(WI)
       // expected-warning@+4{{'ImportMembers::staticDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                                                                                 // expected-warning@+2{{'ImportMembers::staticDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
       void ImportMembers::staticDef() {}
#ifdef GNU
// expected-warning@+2{{'ImportMembers::staticInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
inline void ImportMembers::staticInlineDef() {}
       void ImportMembers::staticInlineDecl() {}

       int  ImportMembers::StaticFieldDef; // expected-error{{definition of dllimport static field not allowed}}
const  int  ImportMembers::StaticConstFieldDef = 1; // expected-error{{definition of dllimport static field not allowed}}
#if __cplusplus < 201703L && !defined(MS)
// expected-error@+2{{definition of dllimport static field not allowed}}
#endif
constexpr int ImportMembers::ConstexprFieldDef;


// Import on member definitions.
struct ImportMemberDefs {
  __declspec(dllimport)                void normalDef();
  __declspec(dllimport)                void normalInlineDef();
  __declspec(dllimport) virtual        void virtualDef();
  __declspec(dllimport) virtual        void virtualInlineDef();
  __declspec(dllimport) static         void staticDef();
  __declspec(dllimport) static         void staticInlineDef();
#ifdef MS
  __declspec(dllimport)         inline void normalInlineDecl();
  __declspec(dllimport) virtual inline void virtualInlineDecl();
  __declspec(dllimport) static  inline void staticInlineDecl();
#endif

  __declspec(dllimport) static         int  StaticField;
  __declspec(dllimport) static  const  int  StaticConstField;
  __declspec(dllimport) constexpr static int ConstexprField = 1;
};

__declspec(dllimport)        void ImportMemberDefs::normalDef() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}
__declspec(dllimport)        void ImportMemberDefs::virtualDef() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}
__declspec(dllimport)        void ImportMemberDefs::staticDef() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}
#ifdef MS
__declspec(dllimport) inline void ImportMemberDefs::normalInlineDef() {}
__declspec(dllimport)        void ImportMemberDefs::normalInlineDecl() {}
__declspec(dllimport) inline void ImportMemberDefs::virtualInlineDef() {}
__declspec(dllimport)        void ImportMemberDefs::virtualInlineDecl() {}
__declspec(dllimport) inline void ImportMemberDefs::staticInlineDef() {}
__declspec(dllimport)        void ImportMemberDefs::staticInlineDecl() {}
#endif

__declspec(dllimport)        int  ImportMemberDefs::StaticField; // expected-error{{definition of dllimport static field not allowed}} expected-note{{attribute is here}}
__declspec(dllimport) const  int  ImportMemberDefs::StaticConstField = 1; // expected-error{{definition of dllimport static field not allowed}} expected-note{{attribute is here}}
#if __cplusplus < 201703L && !defined(MS)
// expected-error@+3{{definition of dllimport static field not allowed}}
// expected-note@+2{{attribute is here}}
#endif
__declspec(dllimport) constexpr int ImportMemberDefs::ConstexprField;


// Import special member functions.
struct ImportSpecials {
  __declspec(dllimport) ImportSpecials();
  __declspec(dllimport) ~ImportSpecials();
  __declspec(dllimport) ImportSpecials(const ImportSpecials&);
  __declspec(dllimport) ImportSpecials& operator=(const ImportSpecials&);
  __declspec(dllimport) ImportSpecials(ImportSpecials&&);
  __declspec(dllimport) ImportSpecials& operator=(ImportSpecials&&);
};


// Import deleted member functions.
struct ImportDeleted {
#if defined(MS) || defined(WI)
  __declspec(dllimport) ImportDeleted() = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
  __declspec(dllimport) ~ImportDeleted() = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
  __declspec(dllimport) ImportDeleted(const ImportDeleted&) = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
  __declspec(dllimport) ImportDeleted& operator=(const ImportDeleted&) = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
  __declspec(dllimport) ImportDeleted(ImportDeleted&&) = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
  __declspec(dllimport) ImportDeleted& operator=(ImportDeleted&&) = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
  __declspec(dllimport) void deleted() = delete; // expected-error{{attribute 'dllimport' cannot be applied to a deleted function}}
#else
  __declspec(dllimport) ImportDeleted() = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
  __declspec(dllimport) ~ImportDeleted() = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
  __declspec(dllimport) ImportDeleted(const ImportDeleted&) = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
  __declspec(dllimport) ImportDeleted& operator=(const ImportDeleted&) = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
  __declspec(dllimport) ImportDeleted(ImportDeleted&&) = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
  __declspec(dllimport) ImportDeleted& operator=(ImportDeleted&&) = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
  __declspec(dllimport) void deleted() = delete; // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif
};


// Import allocation functions.
struct ImportAlloc {
  __declspec(dllimport) void* operator new(__SIZE_TYPE__);
  __declspec(dllimport) void* operator new[](__SIZE_TYPE__);
  __declspec(dllimport) void operator delete(void*);
  __declspec(dllimport) void operator delete[](void*);
};


// Import defaulted member functions.
struct ImportDefaulted {
#ifdef GNU
  // expected-warning@+7{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+7{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+7{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+7{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+7{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+7{{'dllimport' attribute ignored on inline function}}
#endif
  __declspec(dllimport) ImportDefaulted() = default;
  __declspec(dllimport) ~ImportDefaulted() = default;
  __declspec(dllimport) ImportDefaulted(const ImportDefaulted&) = default;
  __declspec(dllimport) ImportDefaulted& operator=(const ImportDefaulted&) = default;
  __declspec(dllimport) ImportDefaulted(ImportDefaulted&&) = default;
  __declspec(dllimport) ImportDefaulted& operator=(ImportDefaulted&&) = default;
};


// Import defaulted member function definitions.
struct ImportDefaultedDefs {
  __declspec(dllimport) ImportDefaultedDefs();
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
  __declspec(dllimport) ~ImportDefaultedDefs(); // expected-note{{previous declaration is here}}

#ifdef GNU
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
// expected-note@+2{{previous declaration is here}}
#endif
  __declspec(dllimport) inline ImportDefaultedDefs(const ImportDefaultedDefs&);
  __declspec(dllimport) ImportDefaultedDefs& operator=(const ImportDefaultedDefs&);

  __declspec(dllimport) ImportDefaultedDefs(ImportDefaultedDefs&&);
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
  __declspec(dllimport) ImportDefaultedDefs &operator=(ImportDefaultedDefs &&); // expected-note{{previous declaration is here}}
};

// Not allowed on definitions.
__declspec(dllimport) ImportDefaultedDefs::ImportDefaultedDefs() = default; // expected-error{{dllimport cannot be applied to non-inline function definition}}

#if defined(MS) || defined(WI)
// expected-warning@+5{{'ImportDefaultedDefs::~ImportDefaultedDefs' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+3{{'ImportDefaultedDefs::~ImportDefaultedDefs' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
// dllimport cannot be dropped.
ImportDefaultedDefs::~ImportDefaultedDefs() = default;

// Import inline declaration and definition.
#ifdef GNU
// expected-error@+3{{redeclaration of 'ImportDefaultedDefs::ImportDefaultedDefs' cannot add 'dllimport' attribute}}
// expected-warning@+3{{'ImportDefaultedDefs::operator=' redeclared inline; 'dllimport' attribute ignored}}
#endif
__declspec(dllimport) ImportDefaultedDefs::ImportDefaultedDefs(const ImportDefaultedDefs&) = default;
inline ImportDefaultedDefs& ImportDefaultedDefs::operator=(const ImportDefaultedDefs&) = default;

__declspec(dllimport) ImportDefaultedDefs::ImportDefaultedDefs(ImportDefaultedDefs&&) = default; // expected-error{{dllimport cannot be applied to non-inline function definition}}
#if defined(MS) || defined(WI)
// expected-warning@+4{{'ImportDefaultedDefs::operator=' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'ImportDefaultedDefs::operator=' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
ImportDefaultedDefs &ImportDefaultedDefs::operator=(ImportDefaultedDefs &&) = default;

// Redeclarations cannot add dllimport.
struct MemberRedecl {
                 void normalDef();         // expected-note{{previous declaration is here}}
          inline void normalInlineDecl();  // expected-note{{previous declaration is here}}
  virtual        void virtualDef();        // expected-note{{previous declaration is here}}
  virtual inline void virtualInlineDecl(); // expected-note{{previous declaration is here}}
  static         void staticDef();         // expected-note{{previous declaration is here}}
  static  inline void staticInlineDecl();  // expected-note{{previous declaration is here}}

#if defined(MS) || defined(WI)
  // expected-note@+4{{previous declaration is here}}
  // expected-note@+4{{previous declaration is here}}
  // expected-note@+4{{previous declaration is here}}
#endif
                 void normalInlineDef();
  virtual        void virtualInlineDef();
  static         void staticInlineDef();

  static         int  StaticField;         // expected-note{{previous declaration is here}}
  static  const  int  StaticConstField;    // expected-note{{previous declaration is here}}
  constexpr static int ConstexprField = 1; // expected-note-re{{previous {{(declaration|definition)}} is here}}
};

__declspec(dllimport)        void MemberRedecl::normalDef() {}         // expected-error{{redeclaration of 'MemberRedecl::normalDef' cannot add 'dllimport' attribute}}
                                                                       // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
__declspec(dllimport)        void MemberRedecl::normalInlineDecl() {}  // expected-error{{redeclaration of 'MemberRedecl::normalInlineDecl' cannot add 'dllimport' attribute}}
__declspec(dllimport)        void MemberRedecl::virtualDef() {}        // expected-error{{redeclaration of 'MemberRedecl::virtualDef' cannot add 'dllimport' attribute}}
                                                                       // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
__declspec(dllimport)        void MemberRedecl::virtualInlineDecl() {} // expected-error{{redeclaration of 'MemberRedecl::virtualInlineDecl' cannot add 'dllimport' attribute}}
__declspec(dllimport)        void MemberRedecl::staticDef() {}         // expected-error{{redeclaration of 'MemberRedecl::staticDef' cannot add 'dllimport' attribute}}
                                                                       // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
__declspec(dllimport)        void MemberRedecl::staticInlineDecl() {}  // expected-error{{redeclaration of 'MemberRedecl::staticInlineDecl' cannot add 'dllimport' attribute}}

#if defined(MS) || defined(WI)
__declspec(dllimport) inline void MemberRedecl::normalInlineDef() {}   // expected-error{{redeclaration of 'MemberRedecl::normalInlineDef' cannot add 'dllimport' attribute}}
__declspec(dllimport) inline void MemberRedecl::virtualInlineDef() {}  // expected-error{{redeclaration of 'MemberRedecl::virtualInlineDef' cannot add 'dllimport' attribute}}
__declspec(dllimport) inline void MemberRedecl::staticInlineDef() {}   // expected-error{{redeclaration of 'MemberRedecl::staticInlineDef' cannot add 'dllimport' attribute}}
#else
__declspec(dllimport) inline void MemberRedecl::normalInlineDef() {}   // expected-warning{{'dllimport' attribute ignored on inline function}}
__declspec(dllimport) inline void MemberRedecl::virtualInlineDef() {}  // expected-warning{{'dllimport' attribute ignored on inline function}}
__declspec(dllimport) inline void MemberRedecl::staticInlineDef() {}   // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif



__declspec(dllimport)        int  MemberRedecl::StaticField = 1;       // expected-error{{redeclaration of 'MemberRedecl::StaticField' cannot add 'dllimport' attribute}}
                                                                       // expected-error@-1{{definition of dllimport static field not allowed}}
                                                                       // expected-note@-2{{attribute is here}}
__declspec(dllimport) const  int  MemberRedecl::StaticConstField = 1;  // expected-error{{redeclaration of 'MemberRedecl::StaticConstField' cannot add 'dllimport' attribute}}
                                                                       // expected-error@-1{{definition of dllimport static field not allowed}}
                                                                       // expected-note@-2{{attribute is here}}

#if __cplusplus < 201703L && !defined(MS)
// expected-error@+6{{redeclaration of 'MemberRedecl::ConstexprField' cannot add 'dllimport' attribute}}
// expected-error@+5{{definition of dllimport static field not allowed}}
// expected-note@+4{{attribute is here}}
#else
// expected-warning@+2{{attribute declaration must precede definition}}
#endif
__declspec(dllimport) constexpr int MemberRedecl::ConstexprField;



//===----------------------------------------------------------------------===//
// Class member templates
//===----------------------------------------------------------------------===//

struct ImportMemberTmpl {
  template<typename T> __declspec(dllimport)               void normalDecl();
  template<typename T> __declspec(dllimport)               void normalDef(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename T> __declspec(dllimport)               void normalInlineDef();
  template<typename T> __declspec(dllimport) static        void staticDecl();
  template<typename T> __declspec(dllimport) static        void staticDef(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename T> __declspec(dllimport) static        void staticInlineDef();

#ifdef GNU
  template<typename T> __declspec(dllimport)               void normalInclass() {} // expected-warning{{'dllimport' attribute ignored on inline function}}
  template<typename T> __declspec(dllimport)        inline void normalInlineDecl(); // expected-warning{{'dllimport' attribute ignored on inline function}}
  template<typename T> __declspec(dllimport) static        void staticInclass() {} // expected-warning{{'dllimport' attribute ignored on inline function}}
  template<typename T> __declspec(dllimport) static inline void staticInlineDecl(); // expected-warning{{'dllimport' attribute ignored on inline function}}
#else
  template<typename T> __declspec(dllimport)               void normalInclass() {}
  template<typename T> __declspec(dllimport)        inline void normalInlineDecl(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
  template<typename T> __declspec(dllimport) static        void staticInclass() {}
  template<typename T> __declspec(dllimport) static inline void staticInlineDecl(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
#endif

#if __has_feature(cxx_variable_templates)
  template<typename T> __declspec(dllimport) static        int  StaticField;
  template<typename T> __declspec(dllimport) static        int  StaticFieldDef; // expected-note{{attribute is here}}
  template<typename T> __declspec(dllimport) static const  int  StaticConstField;
  template<typename T> __declspec(dllimport) static const  int  StaticConstFieldDef; // expected-note{{attribute is here}}
  template<typename T> __declspec(dllimport) static const  int  StaticConstFieldEqualInit = 1;
  template<typename T> __declspec(dllimport) static const  int  StaticConstFieldBraceInit{1};
  template<typename T> __declspec(dllimport) constexpr static int ConstexprField = 1;
  template<typename T> __declspec(dllimport) constexpr static int ConstexprFieldDef = 1;
#endif // __has_feature(cxx_variable_templates)
};

template<typename T>        void ImportMemberTmpl::normalDef() {} // expected-warning{{'ImportMemberTmpl::normalDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T>        void ImportMemberTmpl::staticDef() {} // expected-warning{{'ImportMemberTmpl::staticDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#ifdef GNU // dllimport was ignored above
template<typename T>        void ImportMemberTmpl::normalInlineDecl() {}
template<typename T>        void ImportMemberTmpl::staticInlineDecl() {}
#else // dllimport dropped here
template<typename T>        void ImportMemberTmpl::normalInlineDecl() {} // expected-warning{{'ImportMemberTmpl::normalInlineDecl' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T>        void ImportMemberTmpl::staticInlineDecl() {} // expected-warning{{'ImportMemberTmpl::staticInlineDecl' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif

#ifdef GNU
template<typename T> inline void ImportMemberTmpl::normalInlineDef() {} // expected-warning{{ImportMemberTmpl::normalInlineDef' redeclared inline; 'dllimport' attribute ignored}}
template<typename T> inline void ImportMemberTmpl::staticInlineDef() {} // expected-warning{{ImportMemberTmpl::staticInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#else
template<typename T> inline void ImportMemberTmpl::normalInlineDef() {} // expected-warning{{ImportMemberTmpl::normalInlineDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T> inline void ImportMemberTmpl::staticInlineDef() {} // expected-warning{{ImportMemberTmpl::staticInlineDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif

#if __has_feature(cxx_variable_templates)
template<typename T>        int  ImportMemberTmpl::StaticFieldDef; // expected-error{{definition of dllimport static field not allowed}}
template<typename T> const  int  ImportMemberTmpl::StaticConstFieldDef = 1; // expected-error{{definition of dllimport static field not allowed}}
#ifdef MS
template<typename T> constexpr int ImportMemberTmpl::ConstexprFieldDef;
#endif
#endif // __has_feature(cxx_variable_templates)


// Redeclarations cannot add dllimport.
struct MemTmplRedecl {
  template<typename T>               void normalDef();         // expected-note{{previous declaration is here}}
  template<typename T>        inline void normalInlineDecl();  // expected-note{{previous declaration is here}}
  template<typename T> static        void staticDef();         // expected-note{{previous declaration is here}}
  template<typename T> static inline void staticInlineDecl();  // expected-note{{previous declaration is here}}

#if defined(MS) || defined(WI)
// expected-note@+3{{previous declaration is here}}
// expected-note@+3{{previous declaration is here}}
#endif
  template<typename T>               void normalInlineDef();
  template<typename T> static        void staticInlineDef();

#if __has_feature(cxx_variable_templates)
  template<typename T> static        int  StaticField;         // expected-note{{previous declaration is here}}
  template<typename T> static const  int  StaticConstField;    // expected-note{{previous declaration is here}}
#ifdef MS
  template<typename T> constexpr static int ConstexprField = 1; // expected-note{{previous definition is here}}
#endif
#endif // __has_feature(cxx_variable_templates)
};

template<typename T> __declspec(dllimport)        void MemTmplRedecl::normalDef() {}        // expected-error{{redeclaration of 'MemTmplRedecl::normalDef' cannot add 'dllimport' attribute}}
                                                                                            // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
#if defined(MS) || defined(WI)
template<typename T> __declspec(dllimport) inline void MemTmplRedecl::normalInlineDef() {}  // expected-error{{redeclaration of 'MemTmplRedecl::normalInlineDef' cannot add 'dllimport' attribute}}
#else
template<typename T> __declspec(dllimport) inline void MemTmplRedecl::normalInlineDef() {}  // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif
template<typename T> __declspec(dllimport)        void MemTmplRedecl::normalInlineDecl() {} // expected-error{{redeclaration of 'MemTmplRedecl::normalInlineDecl' cannot add 'dllimport' attribute}}
template<typename T> __declspec(dllimport)        void MemTmplRedecl::staticDef() {}        // expected-error{{redeclaration of 'MemTmplRedecl::staticDef' cannot add 'dllimport' attribute}}
                                                                                            // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
#if defined(MS) || defined(WI)
template<typename T> __declspec(dllimport) inline void MemTmplRedecl::staticInlineDef() {}  // expected-error{{redeclaration of 'MemTmplRedecl::staticInlineDef' cannot add 'dllimport' attribute}}
#else
template<typename T> __declspec(dllimport) inline void MemTmplRedecl::staticInlineDef() {}  // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif
template<typename T> __declspec(dllimport)        void MemTmplRedecl::staticInlineDecl() {} // expected-error{{redeclaration of 'MemTmplRedecl::staticInlineDecl' cannot add 'dllimport' attribute}}

#if __has_feature(cxx_variable_templates)
template<typename T> __declspec(dllimport)        int  MemTmplRedecl::StaticField = 1;      // expected-error{{redeclaration of 'MemTmplRedecl::StaticField' cannot add 'dllimport' attribute}}
                                                                                            // expected-error@-1{{definition of dllimport static field not allowed}}
                                                                                            // expected-note@-2{{attribute is here}}
template<typename T> __declspec(dllimport) const  int  MemTmplRedecl::StaticConstField = 1; // expected-error{{redeclaration of 'MemTmplRedecl::StaticConstField' cannot add 'dllimport' attribute}}
                                                                                            // expected-error@-1{{definition of dllimport static field not allowed}}
                                                                                            // expected-note@-2{{attribute is here}}
#ifdef MS
// expected-warning@+1{{attribute declaration must precede definition}}
template<typename T> __declspec(dllimport) constexpr int MemTmplRedecl::ConstexprField;
#endif
#endif // __has_feature(cxx_variable_templates)



struct MemFunTmpl {
  template<typename T>                              void normalDef() {}
#ifdef GNU
  // expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
  template<typename T> __declspec(dllimport)        void importedNormal() {}
  template<typename T>                       static void staticDef() {}
#ifdef GNU
  // expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
  template<typename T> __declspec(dllimport) static void importedStatic() {}
};

// Import implicit instantiation of an imported member function template.
void useMemFunTmpl() {
  MemFunTmpl().importedNormal<ImplicitInst_Imported>();
  MemFunTmpl().importedStatic<ImplicitInst_Imported>();
}

// Import explicit instantiation declaration of an imported member function
// template.
extern template void MemFunTmpl::importedNormal<ExplicitDecl_Imported>();
extern template void MemFunTmpl::importedStatic<ExplicitDecl_Imported>();

// Import explicit instantiation definition of an imported member function
// template.
// NB: MSVC fails this instantiation without explicit dllimport.
template void MemFunTmpl::importedNormal<ExplicitInst_Imported>();
template void MemFunTmpl::importedStatic<ExplicitInst_Imported>();

// Import specialization of an imported member function template.
template<> __declspec(dllimport) void MemFunTmpl::importedNormal<ExplicitSpec_Imported>();
template<> __declspec(dllimport) void MemFunTmpl::importedNormal<ExplicitSpec_Def_Imported>() {} // error on mingw
#ifdef GNU
  // expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template<> __declspec(dllimport) inline void MemFunTmpl::importedNormal<ExplicitSpec_InlineDef_Imported>() {}
#if 1
// FIXME: This should not be an error when targeting MSVC. (PR21406)
// expected-error@-7{{dllimport cannot be applied to non-inline function definition}}
#endif

template<> __declspec(dllimport) void MemFunTmpl::importedStatic<ExplicitSpec_Imported>();
template<> __declspec(dllimport) void MemFunTmpl::importedStatic<ExplicitSpec_Def_Imported>() {} // error on mingw
#ifdef GNU
  // expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template<> __declspec(dllimport) inline void MemFunTmpl::importedStatic<ExplicitSpec_InlineDef_Imported>() {}
#if 1
// FIXME: This should not be an error when targeting MSVC. (PR21406)
// expected-error@-7{{dllimport cannot be applied to non-inline function definition}}
#endif

// Not importing specialization of an imported member function template without
// explicit dllimport.
template<> void MemFunTmpl::importedNormal<ExplicitSpec_NotImported>() {}
template<> void MemFunTmpl::importedStatic<ExplicitSpec_NotImported>() {}


// Import explicit instantiation declaration of a non-imported member function
// template.
#ifdef GNU
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
#endif
extern template __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitDecl_Imported>();
extern template __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitDecl_Imported>();

// Import explicit instantiation definition of a non-imported member function
// template.
#ifdef GNU
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
#endif
template __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitInst_Imported>();
template __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitInst_Imported>();

// Import specialization of a non-imported member function template.
template<> __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitSpec_Imported>();
template<> __declspec(dllimport) void MemFunTmpl::normalDef<ExplicitSpec_Def_Imported>() {} // error on mingw
#ifdef GNU
  // expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template<> __declspec(dllimport) inline void MemFunTmpl::normalDef<ExplicitSpec_InlineDef_Imported>() {}
#if 1
// FIXME: This should not be an error when targeting MSVC. (PR21406)
// expected-error@-7{{dllimport cannot be applied to non-inline function definition}}
#endif

template<> __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitSpec_Imported>();
template<> __declspec(dllimport) void MemFunTmpl::staticDef<ExplicitSpec_Def_Imported>() {} // error on mingw
#ifdef GNU
  // expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
template<> __declspec(dllimport) inline void MemFunTmpl::staticDef<ExplicitSpec_InlineDef_Imported>() {}
#if 1
// FIXME: This should not be an error when targeting MSVC. (PR21406)
// expected-error@-7{{dllimport cannot be applied to non-inline function definition}}
#endif



#if __has_feature(cxx_variable_templates)
struct MemVarTmpl {
  template<typename T>                       static const int StaticVar = 1;
  template<typename T> __declspec(dllimport) static const int ImportedStaticVar = 1;
};

// Import implicit instantiation of an imported member variable template.
int useMemVarTmpl() { return MemVarTmpl::ImportedStaticVar<ImplicitInst_Imported>; }

// Import explicit instantiation declaration of an imported member variable
// template.
extern template const int MemVarTmpl::ImportedStaticVar<ExplicitDecl_Imported>;

// An explicit instantiation definition of an imported member variable template
// cannot be imported because the template must be defined which is illegal. The
// in-class initializer does not count.

// Import specialization of an imported member variable template.
template<> __declspec(dllimport) const int MemVarTmpl::ImportedStaticVar<ExplicitSpec_Imported>;
template<> __declspec(dllimport) const int MemVarTmpl::ImportedStaticVar<ExplicitSpec_Def_Imported> = 1;
                                                                                // expected-error@-1{{definition of dllimport static field not allowed}}
                                                                                // expected-note@-2{{attribute is here}}

// Not importing specialization of a member variable template without explicit
// dllimport.
template<> const int MemVarTmpl::ImportedStaticVar<ExplicitSpec_NotImported>;


// Import explicit instantiation declaration of a non-imported member variable
// template.
extern template __declspec(dllimport) const int MemVarTmpl::StaticVar<ExplicitDecl_Imported>;

// An explicit instantiation definition of a non-imported member variable template
// cannot be imported because the template must be defined which is illegal. The
// in-class initializer does not count.

// Import specialization of a non-imported member variable template.
template<> __declspec(dllimport) const int MemVarTmpl::StaticVar<ExplicitSpec_Imported>;
template<> __declspec(dllimport) const int MemVarTmpl::StaticVar<ExplicitSpec_Def_Imported> = 1;
                                                                                // expected-error@-1{{definition of dllimport static field not allowed}}
                                                                                // expected-note@-2{{attribute is here}}

#endif // __has_feature(cxx_variable_templates)



//===----------------------------------------------------------------------===//
// Class template members
//===----------------------------------------------------------------------===//

// Import individual members of a class template.
template<typename T>
struct ImportClassTmplMembers {
  __declspec(dllimport)                void normalDecl();
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
  __declspec(dllimport) void normalDef(); // expected-note{{previous declaration is here}}
  __declspec(dllimport)                void normalInlineDef();
  __declspec(dllimport) virtual        void virtualDecl();
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
  __declspec(dllimport) virtual void virtualDef(); // expected-note{{previous declaration is here}}
  __declspec(dllimport) virtual        void virtualInlineDef();
  __declspec(dllimport) static         void staticDecl();
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
  __declspec(dllimport) static void staticDef(); // expected-note{{previous declaration is here}}
  __declspec(dllimport) static         void staticInlineDef();

#ifdef GNU
// expected-warning@+7{{'dllimport' attribute ignored on inline function}}
// expected-warning@+7{{'dllimport' attribute ignored on inline function}}
// expected-warning@+7{{'dllimport' attribute ignored on inline function}}
// expected-warning@+7{{'dllimport' attribute ignored on inline function}}
// expected-warning@+7{{'dllimport' attribute ignored on inline function}}
// expected-warning@+7{{'dllimport' attribute ignored on inline function}}
#endif
  __declspec(dllimport)                void normalInclass() {}
  __declspec(dllimport)         inline void normalInlineDecl();
  __declspec(dllimport) virtual        void virtualInclass() {}
  __declspec(dllimport) virtual inline void virtualInlineDecl();
  __declspec(dllimport) static         void staticInclass() {}
  __declspec(dllimport) static  inline void staticInlineDecl();

protected:
  __declspec(dllimport)                void protectedDecl();
private:
  __declspec(dllimport)                void privateDecl();
public:

  __declspec(dllimport)                int  Field; // expected-warning{{'dllimport' attribute only applies to}}
  __declspec(dllimport) static         int  StaticField;
  __declspec(dllimport) static         int  StaticFieldDef; // expected-note{{attribute is here}}
  __declspec(dllimport) static  const  int  StaticConstField;
  __declspec(dllimport) static  const  int  StaticConstFieldDef; // expected-note{{attribute is here}}
  __declspec(dllimport) static  const  int  StaticConstFieldEqualInit = 1;
  __declspec(dllimport) static  const  int  StaticConstFieldBraceInit{1};
  __declspec(dllimport) constexpr static int ConstexprField = 1;
#if __cplusplus < 201703L && !defined(MS)
  // expected-note@+2{{attribute is here}}
#endif
  __declspec(dllimport) constexpr static int ConstexprFieldDef = 1;
};

// NB: MSVC is inconsistent here and disallows *InlineDef on class templates,
// but allows it on classes. We allow both.
#if defined(MS) || defined(WI)
// expected-warning@+5{{'ImportClassTmplMembers::normalDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+3{{'ImportClassTmplMembers::normalDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
template <typename T>
void ImportClassTmplMembers<T>::normalDef() {}
#ifdef GNU
// expected-warning@+2{{'ImportClassTmplMembers::normalInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
template<typename T> inline void ImportClassTmplMembers<T>::normalInlineDef() {}
template<typename T>        void ImportClassTmplMembers<T>::normalInlineDecl() {}
#if defined(MS) || defined(WI)
// expected-warning@+5{{'ImportClassTmplMembers::virtualDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+3{{'ImportClassTmplMembers::virtualDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
template <typename T>
void ImportClassTmplMembers<T>::virtualDef() {}
#ifdef GNU
// expected-warning@+2{{'ImportClassTmplMembers::virtualInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
template<typename T> inline void ImportClassTmplMembers<T>::virtualInlineDef() {}
template<typename T>        void ImportClassTmplMembers<T>::virtualInlineDecl() {}
#if defined(MS) || defined(WI)
// expected-warning@+5{{'ImportClassTmplMembers::staticDef' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+3{{'ImportClassTmplMembers::staticDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
template <typename T>
void ImportClassTmplMembers<T>::staticDef() {}
#ifdef GNU
// expected-warning@+2{{'ImportClassTmplMembers::staticInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#endif
template<typename T> inline void ImportClassTmplMembers<T>::staticInlineDef() {}
template<typename T>        void ImportClassTmplMembers<T>::staticInlineDecl() {}

template<typename T>        int  ImportClassTmplMembers<T>::StaticFieldDef; // expected-warning{{definition of dllimport static field}}
template<typename T> const  int  ImportClassTmplMembers<T>::StaticConstFieldDef = 1; // expected-warning{{definition of dllimport static field}}
#if __cplusplus < 201703L && !defined(MS)
// expected-warning@+2{{definition of dllimport static field}}
#endif
template<typename T> constexpr int ImportClassTmplMembers<T>::ConstexprFieldDef;


// Redeclarations cannot add dllimport.
template<typename T>
struct CTMR /*ClassTmplMemberRedecl*/ {
                 void normalDef();         // expected-note{{previous declaration is here}}
          inline void normalInlineDecl();  // expected-note{{previous declaration is here}}
  virtual        void virtualDef();        // expected-note{{previous declaration is here}}
  virtual inline void virtualInlineDecl(); // expected-note{{previous declaration is here}}
  static         void staticDef();         // expected-note{{previous declaration is here}}
  static  inline void staticInlineDecl();  // expected-note{{previous declaration is here}}

#if defined(MS) || defined(WI)
// expected-note@+4{{previous declaration is here}}
// expected-note@+4{{previous declaration is here}}
// expected-note@+4{{previous declaration is here}}
#endif
                 void normalInlineDef();
  virtual        void virtualInlineDef();
  static         void staticInlineDef();

  static         int  StaticField;         // expected-note{{previous declaration is here}}
  static  const  int  StaticConstField;    // expected-note{{previous declaration is here}}
  constexpr static int ConstexprField = 1; // expected-note-re{{previous {{(declaration|definition)}} is here}}
};

template<typename T> __declspec(dllimport)        void CTMR<T>::normalDef() {}         // expected-error{{redeclaration of 'CTMR::normalDef' cannot add 'dllimport' attribute}}
                                                                                       // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
template<typename T> __declspec(dllimport)        void CTMR<T>::normalInlineDecl() {}  // expected-error{{redeclaration of 'CTMR::normalInlineDecl' cannot add 'dllimport' attribute}}
template<typename T> __declspec(dllimport)        void CTMR<T>::virtualDef() {}        // expected-error{{redeclaration of 'CTMR::virtualDef' cannot add 'dllimport' attribute}}
                                                                                       // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
template<typename T> __declspec(dllimport)        void CTMR<T>::virtualInlineDecl() {} // expected-error{{redeclaration of 'CTMR::virtualInlineDecl' cannot add 'dllimport' attribute}}
template<typename T> __declspec(dllimport)        void CTMR<T>::staticDef() {}         // expected-error{{redeclaration of 'CTMR::staticDef' cannot add 'dllimport' attribute}}
                                                                                       // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
template<typename T> __declspec(dllimport)        void CTMR<T>::staticInlineDecl() {}  // expected-error{{redeclaration of 'CTMR::staticInlineDecl' cannot add 'dllimport' attribute}}

#if defined(MS) || defined(WI)
template<typename T> __declspec(dllimport) inline void CTMR<T>::normalInlineDef() {}   // expected-error{{redeclaration of 'CTMR::normalInlineDef' cannot add 'dllimport' attribute}}
template<typename T> __declspec(dllimport) inline void CTMR<T>::virtualInlineDef() {}  // expected-error{{redeclaration of 'CTMR::virtualInlineDef' cannot add 'dllimport' attribute}}
template<typename T> __declspec(dllimport) inline void CTMR<T>::staticInlineDef() {}   // expected-error{{redeclaration of 'CTMR::staticInlineDef' cannot add 'dllimport' attribute}}
#else
template<typename T> __declspec(dllimport) inline void CTMR<T>::normalInlineDef() {}   // expected-warning{{'dllimport' attribute ignored on inline function}}
template<typename T> __declspec(dllimport) inline void CTMR<T>::virtualInlineDef() {}  // expected-warning{{'dllimport' attribute ignored on inline function}}
template<typename T> __declspec(dllimport) inline void CTMR<T>::staticInlineDef() {}   // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif

template<typename T> __declspec(dllimport)        int  CTMR<T>::StaticField = 1;       // expected-error{{redeclaration of 'CTMR::StaticField' cannot add 'dllimport' attribute}}
                                                                                       // expected-warning@-1{{definition of dllimport static field}}
                                                                                       // expected-note@-2{{attribute is here}}
template<typename T> __declspec(dllimport) const  int  CTMR<T>::StaticConstField = 1;  // expected-error{{redeclaration of 'CTMR::StaticConstField' cannot add 'dllimport' attribute}}
                                                                                       // expected-warning@-1{{definition of dllimport static field}}
                                                                                       // expected-note@-2{{attribute is here}}

#if __cplusplus < 201703L && !defined(MS)
// expected-error@+6{{redeclaration of 'CTMR::ConstexprField' cannot add 'dllimport' attribute}}
// expected-warning@+5{{definition of dllimport static field}}
// expected-note@+4{{attribute is here}}
#else
// expected-warning@+2{{attribute declaration must precede definition}}
#endif
template<typename T> __declspec(dllimport) constexpr int CTMR<T>::ConstexprField;



//===----------------------------------------------------------------------===//
// Class template member templates
//===----------------------------------------------------------------------===//

template<typename T>
struct ImportClsTmplMemTmpl {
  template<typename U> __declspec(dllimport)               void normalDecl();
  template<typename U> __declspec(dllimport)               void normalDef(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename U> __declspec(dllimport)               void normalInlineDef();
  template<typename U> __declspec(dllimport) static        void staticDecl();
  template<typename U> __declspec(dllimport) static        void staticDef(); // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename U> __declspec(dllimport) static        void staticInlineDef();

#ifdef GNU
  // expected-warning@+5{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+8{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+8{{'dllimport' attribute ignored on inline function}}
  // expected-warning@+11{{'dllimport' attribute ignored on inline function}}
#endif
  template<typename U> __declspec(dllimport)               void normalInclass() {}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename U> __declspec(dllimport)        inline void normalInlineDecl();
  template<typename U> __declspec(dllimport) static        void staticInclass() {}
#if defined(MS) || defined(WI)
// expected-note@+2{{previous declaration is here}} expected-note@+2{{previous attribute is here}}
#endif
  template<typename U> __declspec(dllimport) static inline void staticInlineDecl();

#if __has_feature(cxx_variable_templates)
  template<typename U> __declspec(dllimport) static        int  StaticField;
  template<typename U> __declspec(dllimport) static        int  StaticFieldDef; // expected-note{{attribute is here}}
  template<typename U> __declspec(dllimport) static const  int  StaticConstField;
  template<typename U> __declspec(dllimport) static const  int  StaticConstFieldDef; // expected-note{{attribute is here}}
  template<typename U> __declspec(dllimport) static const  int  StaticConstFieldEqualInit = 1;
  template<typename U> __declspec(dllimport) static const  int  StaticConstFieldBraceInit{1};
  template<typename U> __declspec(dllimport) constexpr static int ConstexprField = 1;
#ifdef MS
  template<typename U> __declspec(dllimport) constexpr static int ConstexprFieldDef = 1;
#endif
#endif // __has_feature(cxx_variable_templates)
};

template<typename T> template<typename U>        void ImportClsTmplMemTmpl<T>::normalDef() {} // expected-warning{{'ImportClsTmplMemTmpl::normalDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T> template<typename U>        void ImportClsTmplMemTmpl<T>::staticDef() {} // expected-warning{{'ImportClsTmplMemTmpl::staticDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#ifdef GNU
template<typename T> template<typename U>        void ImportClsTmplMemTmpl<T>::normalInlineDecl() {}
template<typename T> template<typename U>        void ImportClsTmplMemTmpl<T>::staticInlineDecl() {}
#else
template<typename T> template<typename U>        void ImportClsTmplMemTmpl<T>::normalInlineDecl() {} // expected-warning{{'ImportClsTmplMemTmpl::normalInlineDecl' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T> template<typename U>        void ImportClsTmplMemTmpl<T>::staticInlineDecl() {} // expected-warning{{'ImportClsTmplMemTmpl::staticInlineDecl' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif

#ifdef GNU
template<typename T> template<typename U> inline void ImportClsTmplMemTmpl<T>::normalInlineDef() {} // expected-warning{{'ImportClsTmplMemTmpl::normalInlineDef' redeclared inline; 'dllimport' attribute ignored}}
template<typename T> template<typename U> inline void ImportClsTmplMemTmpl<T>::staticInlineDef() {} // expected-warning{{'ImportClsTmplMemTmpl::staticInlineDef' redeclared inline; 'dllimport' attribute ignored}}
#else
template<typename T> template<typename U> inline void ImportClsTmplMemTmpl<T>::normalInlineDef() {} // expected-warning{{'ImportClsTmplMemTmpl::normalInlineDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
template<typename T> template<typename U> inline void ImportClsTmplMemTmpl<T>::staticInlineDef() {} // expected-warning{{'ImportClsTmplMemTmpl::staticInlineDef' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif

#if __has_feature(cxx_variable_templates)
template<typename T> template<typename U>        int  ImportClsTmplMemTmpl<T>::StaticFieldDef; // expected-warning{{definition of dllimport static field}}
template<typename T> template<typename U> const  int  ImportClsTmplMemTmpl<T>::StaticConstFieldDef = 1; // expected-warning{{definition of dllimport static field}}
#ifdef MS
template<typename T> template<typename U> constexpr int ImportClsTmplMemTmpl<T>::ConstexprFieldDef;
#endif
#endif // __has_feature(cxx_variable_templates)


// Redeclarations cannot add dllimport.
template<typename T>
struct CTMTR /*ClassTmplMemberTmplRedecl*/ {
  template<typename U>               void normalDef();         // expected-note{{previous declaration is here}}
  template<typename U>        inline void normalInlineDecl();  // expected-note{{previous declaration is here}}
  template<typename U> static        void staticDef();         // expected-note{{previous declaration is here}}
  template<typename U> static inline void staticInlineDecl();  // expected-note{{previous declaration is here}}

#if defined(MS) || defined(WI)
  // expected-note@+3{{previous declaration is here}}
  // expected-note@+3{{previous declaration is here}}
#endif
  template<typename U>               void normalInlineDef();
  template<typename U> static        void staticInlineDef();

#if __has_feature(cxx_variable_templates)
  template<typename U> static        int  StaticField;         // expected-note{{previous declaration is here}}
  template<typename U> static const  int  StaticConstField;    // expected-note{{previous declaration is here}}
#ifdef MS
  template<typename U> constexpr static int ConstexprField = 1; // expected-note{{previous definition is here}}
#endif
#endif // __has_feature(cxx_variable_templates)
};

template<typename T> template<typename U> __declspec(dllimport)        void CTMTR<T>::normalDef() {}         // expected-error{{redeclaration of 'CTMTR::normalDef' cannot add 'dllimport' attribute}}
                                                                                                             // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
template<typename T> template<typename U> __declspec(dllimport)        void CTMTR<T>::normalInlineDecl() {}  // expected-error{{redeclaration of 'CTMTR::normalInlineDecl' cannot add 'dllimport' attribute}}
template<typename T> template<typename U> __declspec(dllimport)        void CTMTR<T>::staticDef() {}         // expected-error{{redeclaration of 'CTMTR::staticDef' cannot add 'dllimport' attribute}}
                                                                                                             // expected-error@-1{{dllimport cannot be applied to non-inline function definition}}
template<typename T> template<typename U> __declspec(dllimport)        void CTMTR<T>::staticInlineDecl() {}  // expected-error{{redeclaration of 'CTMTR::staticInlineDecl' cannot add 'dllimport' attribute}}

#if defined(MS) || defined(WI)
template<typename T> template<typename U> __declspec(dllimport) inline void CTMTR<T>::normalInlineDef() {}   // expected-error{{redeclaration of 'CTMTR::normalInlineDef' cannot add 'dllimport' attribute}}
template<typename T> template<typename U> __declspec(dllimport) inline void CTMTR<T>::staticInlineDef() {}   // expected-error{{redeclaration of 'CTMTR::staticInlineDef' cannot add 'dllimport' attribute}}
#else
template<typename T> template<typename U> __declspec(dllimport) inline void CTMTR<T>::normalInlineDef() {}   // expected-warning{{'dllimport' attribute ignored on inline function}}
template<typename T> template<typename U> __declspec(dllimport) inline void CTMTR<T>::staticInlineDef() {}   // expected-warning{{'dllimport' attribute ignored on inline function}}
#endif

#if __has_feature(cxx_variable_templates)
template<typename T> template<typename U> __declspec(dllimport)        int  CTMTR<T>::StaticField = 1;       // expected-error{{redeclaration of 'CTMTR::StaticField' cannot add 'dllimport' attribute}}
                                                                                                             // expected-warning@-1{{definition of dllimport static field}}
                                                                                                             // expected-note@-2{{attribute is here}}
template<typename T> template<typename U> __declspec(dllimport) const  int  CTMTR<T>::StaticConstField = 1;  // expected-error{{redeclaration of 'CTMTR::StaticConstField' cannot add 'dllimport' attribute}}
                                                                                                             // expected-warning@-1{{definition of dllimport static field}}
                                                                                                             // expected-note@-2{{attribute is here}}
#ifdef MS
// expected-warning@+1{{attribute declaration must precede definition}}
template<typename T> template<typename U> __declspec(dllimport) constexpr int CTMTR<T>::ConstexprField;
#endif
#endif // __has_feature(cxx_variable_templates)



//===----------------------------------------------------------------------===//
// Classes
//===----------------------------------------------------------------------===//

namespace {
  struct __declspec(dllimport) AnonymousClass {}; // expected-error{{(anonymous namespace)::AnonymousClass' must have external linkage when declared 'dllimport'}}
}

class __declspec(dllimport) ClassDecl;

class __declspec(dllimport) ClassDef { };

template <typename T> class ClassTemplate {};

#if defined(MS) || defined(WI)
// expected-note@+5{{previous attribute is here}}
// expected-note@+4{{previous attribute is here}}
// expected-error@+4{{attribute 'dllexport' cannot be applied to member of 'dllimport' class}}
// expected-error@+4{{attribute 'dllimport' cannot be applied to member of 'dllimport' class}}
#endif
class __declspec(dllimport) ImportClassWithDllMember {
  void __declspec(dllexport) foo();
  void __declspec(dllimport) bar();
};

#if defined(MS) || defined(WI)
// expected-note@+5{{previous attribute is here}}
// expected-note@+4{{previous attribute is here}}
// expected-error@+4{{attribute 'dllimport' cannot be applied to member of 'dllexport' class}}
// expected-error@+4{{attribute 'dllexport' cannot be applied to member of 'dllexport' class}}
#endif
template <typename T> class __declspec(dllexport) ExportClassWithDllMember {
  void __declspec(dllimport) foo();
  void __declspec(dllexport) bar();
};

namespace ImportedExplicitSpecialization {
template <typename T> struct S { static int x; };
template <typename T> int S<T>::x = sizeof(T);
template <> struct __declspec(dllimport) S<int> { static int x; }; // expected-note{{attribute is here}}
int S<int>::x = -1; // expected-error{{definition of dllimport static field not allowed}}
}

namespace PR19988 {
// Don't error about applying delete to dllimport member function when instantiating.
template <typename> struct __declspec(dllimport) S {
  void foo() = delete;
};
S<int> s;
}

#if defined(MS) || defined(WI)
// expected-warning@+3{{'dllimport' attribute ignored}}
#endif
template <typename T> struct PartiallySpecializedClassTemplate {};
template <typename T> struct __declspec(dllimport) PartiallySpecializedClassTemplate<T*> { void f() {} };

template <typename T> struct ExpliciallySpecializedClassTemplate {};
template <> struct __declspec(dllimport) ExpliciallySpecializedClassTemplate<int> { void f() {} };


//===----------------------------------------------------------------------===//
// Classes with template base classes
//===----------------------------------------------------------------------===//

template <typename T> class __declspec(dllexport) ExportedClassTemplate {};

template <typename T> class __declspec(dllimport) ImportedClassTemplate {};

// ClassTemplate<int> gets imported.
class __declspec(dllimport) DerivedFromTemplate : public ClassTemplate<int> {};

// ClassTemplate<int> is already imported.
class __declspec(dllimport) DerivedFromTemplate2 : public ClassTemplate<int> {};

// ImportedClassTemplate is expliitly imported.
class __declspec(dllimport) DerivedFromImportedTemplate : public ImportedClassTemplate<int> {};

// ExportedClassTemplate is explicitly exported.
class __declspec(dllimport) DerivedFromExportedTemplate : public ExportedClassTemplate<int> {};

class DerivedFromTemplateD : public ClassTemplate<double> {};
// Base class previously implicitly instantiated without attribute; it will get propagated.
class __declspec(dllimport) DerivedFromTemplateD2 : public ClassTemplate<double> {};

// Base class has explicit instantiation declaration; the attribute will get propagated.
extern template class ClassTemplate<float>;
class __declspec(dllimport) DerivedFromTemplateF : public ClassTemplate<float> {};

class __declspec(dllimport) DerivedFromTemplateB : public ClassTemplate<bool> {};
// The second derived class doesn't change anything, the attribute that was propagated first wins.
class __declspec(dllexport) DerivedFromTemplateB2 : public ClassTemplate<bool> {};

template <typename T> struct ExplicitlySpecializedTemplate { void func() {} };
#ifdef MS
// expected-note@+2{{class template 'ExplicitlySpecializedTemplate<int>' was explicitly specialized here}}
#endif
template <> struct ExplicitlySpecializedTemplate<int> { void func() {} };
template <typename T> struct ExplicitlyExportSpecializedTemplate { void func() {} };
template <> struct __declspec(dllexport) ExplicitlyExportSpecializedTemplate<int> { void func() {} };
template <typename T> struct ExplicitlyImportSpecializedTemplate { void func() {} };
template <> struct __declspec(dllimport) ExplicitlyImportSpecializedTemplate<int> { void func() {} };

template <typename T> struct ExplicitlyInstantiatedTemplate { void func() {} };
#ifdef MS
// expected-note@+2{{class template 'ExplicitlyInstantiatedTemplate<int>' was instantiated here}}
#endif
template struct ExplicitlyInstantiatedTemplate<int>;
template <typename T> struct ExplicitlyExportInstantiatedTemplate { void func() {} };
template struct __declspec(dllexport) ExplicitlyExportInstantiatedTemplate<int>;
template <typename T> struct ExplicitlyImportInstantiatedTemplate { void func() {} };
template struct __declspec(dllimport) ExplicitlyImportInstantiatedTemplate<int>;

#ifdef MS
// expected-warning@+3{{propagating dll attribute to explicitly specialized base class template without dll attribute is not supported}}
// expected-note@+2{{attribute is here}}
#endif
struct __declspec(dllimport) DerivedFromExplicitlySpecializedTemplate : public ExplicitlySpecializedTemplate<int> {};

// Base class already specialized with export attribute.
struct __declspec(dllimport) DerivedFromExplicitlyExportSpecializedTemplate : public ExplicitlyExportSpecializedTemplate<int> {};

// Base class already specialized with import attribute.
struct __declspec(dllimport) DerivedFromExplicitlyImportSpecializedTemplate : public ExplicitlyImportSpecializedTemplate<int> {};

#ifdef MS
// expected-warning@+3{{propagating dll attribute to already instantiated base class template without dll attribute is not supported}}
// expected-note@+2{{attribute is here}}
#endif
struct __declspec(dllimport) DerivedFromExplicitlyInstantiatedTemplate : public ExplicitlyInstantiatedTemplate<int> {};

// Base class already instantiated with export attribute.
struct __declspec(dllimport) DerivedFromExplicitlyExportInstantiatedTemplate : public ExplicitlyExportInstantiatedTemplate<int> {};

// Base class already instantiated with import attribute.
struct __declspec(dllimport) DerivedFromExplicitlyImportInstantiatedTemplate : public ExplicitlyImportInstantiatedTemplate<int> {};

template <typename T> struct ExplicitInstantiationDeclTemplateBase { void func() {} };
extern template struct ExplicitInstantiationDeclTemplateBase<int>;
struct __declspec(dllimport) DerivedFromExplicitInstantiationDeclTemplateBase : public ExplicitInstantiationDeclTemplateBase<int> {};

//===----------------------------------------------------------------------===//
// Lambdas
//===----------------------------------------------------------------------===//
// The MS ABI doesn't provide a stable mangling for lambdas, so they can't be imported or exported.
#if defined(MS) || defined(WI)
// expected-error@+4{{lambda cannot be declared 'dllimport'}}
#else
// expected-warning@+2{{'dllimport' attribute ignored on inline function}}
#endif
auto Lambda = []() __declspec(dllimport) -> bool { return true; };
