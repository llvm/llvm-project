// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fno-objc-infer-related-result-type -Werror -Wno-error=incomplete-umbrella -fmodule-cache-path %t -F %S/Inputs -I %S/Inputs -verify %s
// RUN: %clang_cc1 -fmodules -fno-objc-infer-related-result-type -Werror -Wno-error=incomplete-umbrella -x objective-c++ -fmodule-cache-path %t -F %S/Inputs -I %S/Inputs -verify %s
// RUN: %clang_cc1 -fmodules -fno-objc-infer-related-result-type -Werror -Wno-error=incomplete-umbrella -fmodule-cache-path %t -F %S/Inputs -I %S/Inputs -verify %s
#define FOO
@__experimental_modules_import Module;
@interface OtherClass
@end

// in module: expected-note@17{{class method 'alloc' is assumed to return an instance of its receiver type ('Module *')}}
void test_getModuleVersion() {
  const char *version = getModuleVersion();
  const char *version2 = [Module version];

  OtherClass *other = [Module alloc]; // expected-error{{init}}
}

#ifdef MODULE_SUBFRAMEWORK_H
#  error MODULE_SUBFRAMEWORK_H should be hidden
#endif

@__experimental_modules_import subdir;

const char *getSubdirTest() { return getSubdir(); }
