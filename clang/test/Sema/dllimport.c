// RUN: %clang_cc1 -triple i686-win32     -fsyntax-only -fms-extensions -verify -std=c99 -DMS %s
// RUN: %clang_cc1 -triple x86_64-win32   -fsyntax-only -fms-extensions -verify -std=c11 -DMS %s
// RUN: %clang_cc1 -triple i686-mingw32   -fsyntax-only -fms-extensions -verify -std=c11 -DGNU %s
// RUN: %clang_cc1 -triple x86_64-mingw32 -fsyntax-only -fms-extensions -verify -std=c99 -DGNU %s
// RUN: %clang_cc1 -triple aarch64-win32  -fsyntax-only -fms-extensions -verify -std=c99 -DMS %s

// Invalid usage.
__declspec(dllimport) typedef int typedef1;
// expected-warning@-1{{'dllimport' attribute only applies to variables and functions}}
typedef __declspec(dllimport) int typedef2;
// expected-warning@-1{{'dllimport' attribute only applies to variables and functions}}
typedef int __declspec(dllimport) typedef3;
// expected-warning@-1{{'dllimport' attribute only applies to variables and functions}}
typedef __declspec(dllimport) void (*FunTy)();
// expected-warning@-1{{'dllimport' attribute only applies to variables and functions}}
enum __declspec(dllimport) Enum { EnumVal };
// expected-warning@-1{{'dllimport' attribute only applies to variables and functions}}
struct __declspec(dllimport) Record {};
// expected-warning@-1{{'dllimport' attribute only applies to variables and functions}}



//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

// Import declaration.
__declspec(dllimport) extern int ExternGlobalDecl;

// dllimport implies a declaration.
__declspec(dllimport) int GlobalDecl;
int **__attribute__((dllimport))* GlobalDeclChunkAttr;
int GlobalDeclAttr __attribute__((dllimport));

// Address of variables can't be used for initialization in C language modes.
int *VarForInit = &GlobalDecl; // expected-error{{initializer element is not a compile-time constant}}

// Not allowed on definitions.
__declspec(dllimport) extern int ExternGlobalInit = 1; // expected-error{{definition of dllimport data}}
__declspec(dllimport) int GlobalInit1 = 1; // expected-error{{definition of dllimport data}}
int __declspec(dllimport) GlobalInit2 = 1; // expected-error{{definition of dllimport data}}

// Declare, then reject definition.
#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
__declspec(dllimport) extern int ExternGlobalDeclInit; // expected-note{{previous declaration is here}}
#ifdef MS
// expected-warning@+4{{'ExternGlobalDeclInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'ExternGlobalDeclInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int ExternGlobalDeclInit = 1;

#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
__declspec(dllimport) int GlobalDeclInit; // expected-note{{previous declaration is here}}
#ifdef MS
// expected-warning@+4{{'GlobalDeclInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'GlobalDeclInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int GlobalDeclInit = 1;

#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
int *__attribute__((dllimport)) GlobalDeclChunkAttrInit; // expected-note{{previous declaration is here}}
#ifdef MS
// expected-warning@+4{{'GlobalDeclChunkAttrInit' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
// expected-warning@+2{{'GlobalDeclChunkAttrInit' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
int *GlobalDeclChunkAttrInit = 0;

#ifdef GNU
// expected-note@+2{{previous attribute is here}}
#endif
int GlobalDeclAttrInit __attribute__((dllimport)); // expected-note{{previous declaration is here}}
#ifdef MS
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

// We follow GCC and drop the dllimport with a warning.
__declspec(dllimport) extern int GlobalRedecl3; // expected-note{{previous declaration is here}} expected-note{{previous attribute is here}}
                      extern int GlobalRedecl3; // expected-warning{{'GlobalRedecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}

// Adding an attribute on redeclaration.
                      extern int GlobalRedecl4; // expected-note{{previous declaration is here}}
int useGlobalRedecl4() { return GlobalRedecl4; }
__declspec(dllimport) extern int GlobalRedecl4; // expected-error{{redeclaration of 'GlobalRedecl4' cannot add 'dllimport' attribute}}

// Allow with a warning if the decl hasn't been used yet.
                      extern int GlobalRedecl5; // expected-note{{previous declaration is here}}
__declspec(dllimport) extern int GlobalRedecl5; // expected-warning{{redeclaration of 'GlobalRedecl5' should not add 'dllimport' attribute}}


// External linkage is required.
__declspec(dllimport) static int StaticGlobal; // expected-error{{'StaticGlobal' must have external linkage when declared 'dllimport'}}

// Thread local variables are invalid.
__declspec(dllimport) __thread int ThreadLocalGlobal; // expected-error{{'ThreadLocalGlobal' cannot be thread local when declared 'dllimport'}}

// Import in local scope.
__declspec(dllimport) float LocalRedecl1; // expected-note{{previous declaration is here}}
__declspec(dllimport) float LocalRedecl2; // expected-note{{previous declaration is here}}
__declspec(dllimport) float LocalRedecl3; // expected-note{{previous declaration is here}}
__declspec(dllimport) float LocalRedecl4;
void functionScope() {
  __declspec(dllimport) int LocalRedecl1; // expected-error{{redeclaration of 'LocalRedecl1' with a different type: 'int' vs 'float'}}
  int *__attribute__((dllimport)) LocalRedecl2; // expected-error{{redeclaration of 'LocalRedecl2' with a different type: 'int *' vs 'float'}}
  int LocalRedecl3 __attribute__((dllimport)); // expected-error{{redeclaration of 'LocalRedecl3' with a different type: 'int' vs 'float'}}

  __declspec(dllimport)        int LocalVarDecl;
  __declspec(dllimport)        int LocalVarDef = 1; // expected-error{{definition of dllimport data}}
  __declspec(dllimport) extern int ExternLocalVarDecl;
  __declspec(dllimport) extern int ExternLocalVarDef = 1; // expected-error{{definition of dllimport data}}
  __declspec(dllimport) static int StaticLocalVar; // expected-error{{'StaticLocalVar' must have external linkage when declared 'dllimport'}}

  // Local extern redeclaration does not drop the attribute.
  extern float LocalRedecl4;
}



//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

// Import function declaration. Check different placements.
__attribute__((dllimport)) void decl1A(); // Sanity check with __attribute__
__declspec(dllimport)      void decl1B();

void __attribute__((dllimport)) decl2A();
void __declspec(dllimport)      decl2B();

// Address of functions can be used for initialization in C language modes.
// However, the address of the thunk wrapping the function is used instead of
// the address in the import address table.
void (*FunForInit)() = &decl2A;

// Not allowed on function definitions.
__declspec(dllimport) void def() {} // expected-error{{dllimport cannot be applied to non-inline function definition}}

// Import inline function.
#ifdef GNU
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
#endif
__declspec(dllimport) inline void inlineFunc1() {}
inline void __attribute__((dllimport)) inlineFunc2() {}

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
#ifdef MS
                      // expected-warning@+4{{'redecl3' redeclared without 'dllimport' attribute: 'dllexport' attribute added}}
#else
                      // expected-warning@+2{{'redecl3' redeclared without 'dllimport' attribute: previous 'dllimport' ignored}}
#endif
                      void redecl3() {}

                      void redecl4(); // expected-note{{previous declaration is here}}
void useRedecl4() { redecl4(); }
__declspec(dllimport) void redecl4(); // expected-warning{{redeclaration of 'redecl4' should not add 'dllimport' attribute}}

// Allow with a warning if the decl hasn't been used yet.
                      void redecl5(); // expected-note{{previous declaration is here}}
__declspec(dllimport) void redecl5(); // expected-warning{{redeclaration of 'redecl5' should not add 'dllimport' attribute}}


// Inline redeclarations.
#ifdef GNU
// expected-warning@+3{{'redecl6' redeclared inline; 'dllimport' attribute ignored}}
#endif
__declspec(dllimport) void redecl6();
                      inline void redecl6() {}

#ifdef MS
// expected-note@+5{{previous declaration is here}}
// expected-warning@+5{{redeclaration of 'redecl7' should not add 'dllimport' attribute}}
#else
// expected-warning@+3{{'dllimport' attribute ignored on inline function}}
#endif
                      void redecl7();
__declspec(dllimport) inline void redecl7() {}

// PR31069: Don't crash trying to merge attributes for redeclaration of invalid
// decl.
void __declspec(dllimport) redecl8(unknowntype X); // expected-error{{unknown type name 'unknowntype'}}
void redecl8(unknowntype X) { } // expected-error{{unknown type name 'unknowntype'}}
// PR32021: Similarly, don't crash trying to merge attributes from a valid
// decl to an invalid redeclaration.
void __declspec(dllimport) redecl9(void); // expected-note{{previous declaration is here}}
int redecl9(void) {} // expected-error{{conflicting types for 'redecl9'}}

// External linkage is required.
__declspec(dllimport) static int staticFunc(); // expected-error{{'staticFunc' must have external linkage when declared 'dllimport'}}

// Static locals don't count as having external linkage.
void staticLocalFunc() {
  __declspec(dllimport) static int staticLocal; // expected-error{{'staticLocal' must have external linkage when declared 'dllimport'}}
}
