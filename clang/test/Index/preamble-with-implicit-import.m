// RUN: rm -rf %t
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 2 none %s -I %S/Inputs -fmodules -fmodules-cache-path=%t -fspell-checking 2>&1 | FileCheck %s
// CHECK: error: declaration of 'Typo' must be imported
// CHECK: error: declaration of 'Typo' must be imported

#include "preamble-with-implicit-import.h"
