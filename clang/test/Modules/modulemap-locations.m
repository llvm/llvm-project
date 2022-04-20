// RUN: rm -rf %t 
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/ModuleMapLocations/Module_ModuleMap -I %S/Inputs/ModuleMapLocations/Both -F %S/Inputs/ModuleMapLocations -I %S/Inputs/ModuleMapLocations -F %S/Inputs -x objective-c -fsyntax-only %s -verify -Wno-private-module

// regular
@import module_modulemap;
@import both;
// framework
@import Module_ModuleMap_F;
@import Module_ModuleMap_F.Private;
@import Both_F;
@import Inferred;

void test(void) {
  will_be_found1();
  wont_be_found1(); // expected-error{{call to undeclared function 'wont_be_found1'; ISO C99 and later do not support implicit function declarations}} \
                       expected-note {{did you mean 'will_be_found1'?}} \
                       expected-note@Inputs/ModuleMapLocations/Module_ModuleMap/a.h:1 {{'will_be_found1' declared here}}
  will_be_found2();
  wont_be_found2(); // expected-error{{call to undeclared function 'wont_be_found2'; ISO C99 and later do not support implicit function declarations}}
}
