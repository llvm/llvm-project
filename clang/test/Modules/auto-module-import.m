// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify -DERRORS
// RUN: %clang_cc1 -Wauto-import -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify
// RUN: %clang_cc1 -Wauto-import -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -xobjective-c++ %s -verify
// 
// Test both with and without the declarations that refer to unimported
// entities. For error recovery, those cases implicitly trigger an import.

#include <DependsOnModule/DependsOnModule.h> // expected-warning{{treating #include as an import of module 'DependsOnModule'}}

#ifdef MODULE_H_MACRO
#  error MODULE_H_MACRO should have been hidden
#endif

#ifdef DEPENDS_ON_MODULE
#  error DEPENDS_ON_MODULE should have been hidden
#endif

#ifdef ERRORS
Module *mod; // expected-error{{declaration of 'Module' must be imported from module 'Module' before it is required}}
// expected-note@Inputs/Module.framework/Headers/Module.h:15 {{previous}}
#else
#import <AlsoDependsOnModule/AlsoDependsOnModule.h> // expected-warning{{treating #import as an import of module 'AlsoDependsOnModule'}}
#endif
Module *mod2;

int getDependsOther() { return depends_on_module_other; }

void testSubframeworkOther() {
#ifdef ERRORS
  double *sfo1 = sub_framework_other; // expected-error{{declaration of 'sub_framework_other' must be imported from module 'DependsOnModule.SubFramework.Other'}}
  // expected-note@Inputs/DependsOnModule.framework/Frameworks/SubFramework.framework/Headers/Other.h:15 {{previous}}
#endif
}

// Test umbrella-less submodule includes
#include <NoUmbrella/A.h> // expected-warning{{treating #include as an import of module 'NoUmbrella.A'}}
int getNoUmbrellaA() { return no_umbrella_A; } 

// Test umbrella-less submodule includes
#include <NoUmbrella/SubDir/C.h> // expected-warning{{treating #include as an import of module 'NoUmbrella.SubDir.C'}}
int getNoUmbrellaC() { return no_umbrella_C; } 

#ifndef ERRORS
// Test header cross-subframework include pattern.
#include <DependsOnModule/../Frameworks/SubFramework.framework/Headers/Other.h> // expected-warning{{treating #include as an import of module 'DependsOnModule.SubFramework.Other'}}
#endif

void testSubframeworkOtherAgain() {
  double *sfo1 = sub_framework_other;
}

void testModuleSubFramework() {
  char *msf = module_subframework;
}

#include <Module/../Frameworks/SubFramework.framework/Headers/SubFramework.h> // expected-warning{{treating #include as an import of module 'Module.SubFramework'}}

void testModuleSubFrameworkAgain() {
  char *msf = module_subframework;
}

// Test inclusion of private headers.
#include <DependsOnModule/DependsOnModulePrivate.h> // expected-warning{{treating #include as an import of module 'DependsOnModule.Private.DependsOnModule'}}

int getDependsOnModulePrivate() { return depends_on_module_private; }

#include <Module/ModulePrivate.h> // includes the header

int getModulePrivate() { return module_private; }

#include <NoUmbrella/A_Private.h> // expected-warning{{treating #include as an import of module 'NoUmbrella.Private.A_Private'}}
int getNoUmbrellaAPrivate() { return no_umbrella_A_private; }

int getNoUmbrellaBPrivateFail() { return no_umbrella_B_private; } // expected-error{{declaration of 'no_umbrella_B_private' must be imported from module 'NoUmbrella.Private.B_Private'}}
// expected-note@Inputs/NoUmbrella.framework/PrivateHeaders/B_Private.h:1 {{previous}}

// Test inclusion of headers that are under an umbrella directory but
// not actually part of the module.
#include <Module/NotInModule.h> // expected-warning{{treating #include as an import of module 'Module.NotInModule'}} \
  // expected-warning{{missing submodule 'Module.NotInModule'}}

int getNotInModule() {
  return not_in_module;
}

void includeNotAtTopLevel() { // expected-note {{function 'includeNotAtTopLevel' begins here}}
  #include <NoUmbrella/A.h> // expected-warning {{treating #include as an import}} \
			       expected-error {{redundant #include of module 'NoUmbrella.A' appears within function 'includeNotAtTopLevel'}}
}

#ifdef __cplusplus
namespace NS { // expected-note {{begins here}}
#include <NoUmbrella/A.h> // expected-warning {{treating #include as an import}} \
                             expected-error {{redundant #include of module 'NoUmbrella.A' appears within namespace 'NS'}}
}
extern "C" { // expected-note {{begins here}}
#include <NoUmbrella/A.h> // expected-warning {{treating #include as an import}} \
                             expected-error {{import of C++ module 'NoUmbrella.A' appears within extern "C"}}
}
#endif
