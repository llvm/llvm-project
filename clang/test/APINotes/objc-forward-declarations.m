// RUN: rm -rf %t && mkdir -p %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/ModulesCache -fapinotes-modules -fsyntax-only -F %S/Inputs/Frameworks %s -verify

@import LayeredKit;

void test(
  UpwardClass *okayClass,
  id <UpwardProto> okayProto,
  PerfectlyNormalClass *badClass // expected-error {{'PerfectlyNormalClass' is unavailable}}
) {
  // expected-note@LayeredKitImpl/LayeredKitImpl.h:4 {{'PerfectlyNormalClass' has been explicitly marked unavailable here}}
}
