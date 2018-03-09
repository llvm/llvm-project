// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -F %S/Inputs/DependsOnModule.framework/Frameworks %s -verify
// RUN: %clang_cc1 -x objective-c++ -Wauto-import -Wno-private-module -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs -F %S/Inputs/DependsOnModule.framework/Frameworks %s -verify

@import DependsOnModule;

void testSubFramework() {
  float *sf1 = sub_framework; // expected-error{{declaration of 'sub_framework' must be imported from module 'DependsOnModule.SubFramework' before it is required}}
  // expected-note@Inputs/DependsOnModule.framework/Frameworks/SubFramework.framework/Headers/SubFramework.h:2 {{previous}}
}

@import DependsOnModule.SubFramework;

void testSubFrameworkAgain() {
  float *sf2 = sub_framework;
  double *sfo1 = sub_framework_other;
}

#ifdef __cplusplus
@import DependsOnModule.CXX;

CXXOnly cxxonly;
#endif

@import HasSubModules;

// expected-warning@Inputs/HasSubModules.framework/Frameworks/Sub.framework/PrivateHeaders/SubPriv.h:1{{treating #include as an import of module 'HasSubModules.Sub.Types'}}
#import <HasSubModules/HasSubModulesPriv.h>

struct FrameworkSubStruct ss;
