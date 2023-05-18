// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/pch-with-module-name -emit-pch -o %t-A.pch %S/Inputs/pch-with-module-name/test.h -fmodule-name=Contacts -x objective-c-header
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I %S/Inputs/pch-with-module-name -include-pch %t-A.pch %s -fsyntax-only -fmodule-name=Contacts -verify
// expected-no-diagnostics 
#include "C.h"
