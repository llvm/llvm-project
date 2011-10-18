// RUN: rm -rf %t
// RUN: %clang_cc1 -fno-objc-infer-related-result-type -Werror -fmodule-cache-path %t -F %S/Inputs -verify %s
// RUN: %clang_cc1 -fno-objc-infer-related-result-type -Werror -x objective-c++ -fmodule-cache-path %t -F %S/Inputs -verify %s
// RUN: %clang_cc1 -fno-objc-infer-related-result-type -Werror -fmodule-cache-path %t -F %S/Inputs -verify %s
#define FOO
__import_module__ Module;
@interface OtherClass
@end
// in module: expected-note{{class method 'alloc' is assumed to return an instance of its receiver type ('Module *')}}
void test_getModuleVersion() {
  const char *version = getModuleVersion();
  const char *version2 = [Module version];

  OtherClass *other = [Module alloc]; // expected-error{{init}}
}


