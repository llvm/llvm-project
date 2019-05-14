// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -analyzer-output=plist-multi-file %s -o %t.plist
// RUN: tail -n +11 %t.plist | %diff_plist %S/Inputs/expected-plists/plist-diagnostics-include-check.cpp.plist -

#include "Inputs/include/plist-diagnostics-include-check-macro.h"

void foo() {
  PlistCheckMacro()
#define PLIST_DEF_MACRO .run();
#include "Inputs/include/plist-diagnostics-include-check-macro.def"
}
