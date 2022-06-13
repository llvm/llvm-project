// RUN: rm -rf %t
// RUN: split-file %s %t

// Scan repeatedly
// RUN: c-index-test core -scan-deps %S -- clang_tool -c %t/main.c -fmodules -fmodules-cache-path=%t/module-cache -fimplicit-modules -fimplicit-module-maps 2>&1 > %t/output1
// RUN: c-index-test core -scan-deps %S -- clang_tool -c %t/main.c -fmodules -fmodules-cache-path=%t/module-cache -fimplicit-modules -fimplicit-module-maps 2>&1 > %t/output2
// RUN: c-index-test core -scan-deps %S -- clang_tool -c %t/main.c -fmodules -fmodules-cache-path=%t/module-cache -fimplicit-modules -fimplicit-module-maps 2>&1 > %t/output3

// Ensure the output is identical each time
// RUN: diff %t/output1 %t/output2
// RUN: diff %t/output1 %t/output3

//--- module.modulemap
module FromMain1 { header "FromMain1.h" }
module FromMain2 { header "FromMain2.h" }
module FromMod1 { header "FromMod1.h" }
module FromMod2 { header "FromMod2.h" }

//--- FromMain1.h
#include "FromMod1.h"
#include "FromMod2.h"

//--- FromMain2.h
void fromMain2(void);

//--- FromMod1.h
void fromMod1(void);

//--- FromMod2.h
void fromMod2(void);

//--- main.c
#include "FromMain1.h"
#include "FromMain2.h"
void m() {
  fromMod1();
  fromMod2();
  fromMain2();
}
