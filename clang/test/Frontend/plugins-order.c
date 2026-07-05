// REQUIRES: plugins, examples

// RUN: %clang_cc1 -load %llvmshlibdir/PluginsOrder%pluginext %s 2>&1 | FileCheck %s --check-prefix=ALWAYS
// ALWAYS: always-before
// ALWAYS-NEXT: always-after

// RUN: %clang_cc1 -load %llvmshlibdir/PluginsOrder%pluginext %s -add-plugin cmd-after -add-plugin cmd-before 2>&1 | FileCheck %s --check-prefix=ALL
// RUN: %clang_cc1 -load %llvmshlibdir/PluginsOrder%pluginext %s -add-plugin cmd-before -add-plugin cmd-after 2>&1 | FileCheck %s --check-prefix=ALL
// ALL: cmd-before
// ALL-NEXT: always-before
// ALL-NEXT: cmd-after
// ALL-NEXT: always-after
