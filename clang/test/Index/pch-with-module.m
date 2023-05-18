// REQUIRES: x86-registered-target
// RUN: rm -rf %t.cache
// RUN: rm -rf %t.h.pch
// RUN: c-index-test -write-pch %t.h.pch %s -target x86_64-apple-macosx10.7 -fobjc-arc -fmodules-cache-path=%t.cache -fmodules -F %S/../Modules/Inputs -Xclang -fdisable-module-hash
// RUN: %clang -fsyntax-only %s -target x86_64-apple-macosx10.7 -include %t.h -fobjc-arc -fmodules-cache-path=%t.cache -fmodules -F %S/../Modules/Inputs \
// RUN:      -Xclang -fdisable-module-hash -Xclang -detailed-preprocessing-record -Xclang -verify

// expected-no-diagnostics

#ifndef PCH_HEADER
#define PCH_HEADER

#include <Module/Module.h>

@interface Module(PCHCat)
-(id)PCH_meth;
@end

#else

void foo(Module *m) {
  [m PCH_meth];
}

#endif
