// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fno-objc-infer-related-result-type -Werror -Wno-error=incomplete-umbrella -fmodules-cache-path=%t -F %S/Inputs -I %S/Inputs -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fno-objc-infer-related-result-type -Werror -Wno-error=incomplete-umbrella -x objective-c++ -fmodules-cache-path=%t -F %S/Inputs -I %S/Inputs -verify %s
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fno-objc-infer-related-result-type -Werror -Wno-error=incomplete-umbrella -fmodules-cache-path=%t -F %S/Inputs -I %S/Inputs -verify %s
#define FOO
@import Module;
@interface OtherClass
@end

// expected-note@Inputs/Module.framework/Headers/Module.h:17{{class method 'alloc' is assumed to return an instance of its receiver type ('Module *')}}
void test_getModuleVersion(void) {
  const char *version = getModuleVersion();
  const char *version2 = [Module version];

  OtherClass *other = [Module alloc]; // expected-error{{init}}
}

#ifdef MODULE_SUBFRAMEWORK_H
#  error MODULE_SUBFRAMEWORK_H should be hidden
#endif

@import subdir;

const char *getSubdirTest(void) { return getSubdir(); }
