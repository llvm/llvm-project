// REQUIRES: shell

// RUN: rm -rf %t && mkdir -p %t

// RUN: %clang -cc1depscan -fdepscan=inline -o %t/inc.rsp -cc1-args -cc1 -triple x86_64-apple-macos11 -emit-obj \
// RUN:   %s -o %t/output.o -fcas-path %t/cas
// RUN: %clang @%t/inc.rsp -Rcompile-job-cache 2> %t/output-tree.txt

// RUN: cat %t/output-tree.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key-tree

// RUN: clang-cas-test -print-compile-job-cache-key -cas %t/cas @%t/cache-key-tree | FileCheck %s -check-prefix=INCLUDE_TREE_KEY -check-prefix=INCLUDE_TREE -DSRC_FILE=%s
//
// INCLUDE_TREE_KEY: command-line: llvmcas://
// INCLUDE_TREE_KEY: computation: llvmcas://
// INCLUDE_TREE_KEY: include-tree: llvmcas://
// INCLUDE_TREE: [[SRC_FILE]] llvmcas://
// INCLUDE_TREE: Files:
// INCLUDE_TREE-NEXT: [[SRC_FILE]] llvmcas://

// RUN: cat %t/inc.rsp | sed \
// RUN:   -e "s/^.*\"-fcas-include-tree\" \"//" \
// RUN:   -e "s/\" .*$//" > %t/include-tree-id

// RUN: clang-cas-test -print-include-tree -cas %t/cas @%t/include-tree-id | FileCheck %s -check-prefix=INCLUDE_TREE -DSRC_FILE=%s

// Print key from plugin CAS.
// RUN: %clang -cc1depscan -fdepscan=inline -o %t/inc-plugin.rsp -cc1-args -cc1 -triple x86_64-apple-macos11 -emit-obj \
// RUN:   %s -o %t/output-plugin.o -fcas-path %t/cas-plugin -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext
// RUN: %clang @%t/inc-plugin.rsp -Rcompile-job-cache-miss 2> %t/output-plugin.txt
// RUN: cat %t/output-plugin.txt | sed \
// RUN:   -e "s/^.*miss for '//" \
// RUN:   -e "s/' .*$//" > %t/cache-key-plugin
// RUN: clang-cas-test -print-compile-job-cache-key -cas %t/cas-plugin @%t/cache-key-plugin \
// RUN:   -fcas-plugin-path %llvmshlibdir/libCASPluginTest%pluginext | FileCheck %s -check-prefix=INCLUDE_TREE_KEY -check-prefix=INCLUDE_TREE -DSRC_FILE=%s
