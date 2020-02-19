// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x objective-c -fmodule-name=objc_type_param -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs %s -verify

@import objc_type_param;

id make(BCallback callback, id arg) {
  return callback(arg); // expected-error {{too few arguments to function call}}
}
