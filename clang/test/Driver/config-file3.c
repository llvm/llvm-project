// REQUIRES: shell
// REQUIRES: x86-registered-target

// RUN: rm -rf %t && mkdir %t

//--- If config file is specified by relative path (workdir/cfg-s2), it is searched for by that path.

// RUN: mkdir -p %t/workdir/subdir
// RUN: echo "@subdir/cfg-s2" > %t/workdir/cfg-1
// RUN: echo "-Wundefined-var-template" > %t/workdir/subdir/cfg-s2
//
// RUN: ( cd %t && %clang --config workdir/cfg-1 -c -### %s 2>&1 | FileCheck %s -check-prefix CHECK-REL )
//
// CHECK-REL: Configuration file: {{.*}}/workdir/cfg-1
// CHECK-REL: -Wundefined-var-template


//--- Invocation qqq-clang-g++ tries to find config file qqq-clang-g++.cfg first.
//
// RUN: mkdir %t/testdmode
// RUN: ln -s %clang %t/testdmode/qqq-clang-g++
// RUN: echo "-Wundefined-func-template" > %t/testdmode/qqq-clang-g++.cfg
// RUN: echo "-Werror" > %t/testdmode/qqq.cfg
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir= --config-user-dir= -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix FULL-NAME
//
// FULL-NAME: Configuration file: {{.*}}/testdmode/qqq-clang-g++.cfg
// FULL-NAME: -Wundefined-func-template
// FULL-NAME-NOT: -Werror
//
//--- Invocation qqq-clang-g++ tries to find config file qqq-clang-g++.cfg even without -no-canonical-prefixes.
// (As the clang executable and symlink are in different directories, this
// requires specifying the path via --config-*-dir= though.)
//
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir= --config-user-dir=%t/testdmode -c -### %s 2>&1 | FileCheck %s -check-prefix SYMLINK
//
// SYMLINK: Configuration file: {{.*}}/testdmode/qqq-clang-g++.cfg
//
//--- File specified by --config overrides config inferred from clang executable.
//
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --config i386-qqq.cfg -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-EXPLICIT
//
// CHECK-EXPLICIT: Configuration file: {{.*}}/Inputs/config/i386-qqq.cfg

//--- --no-default-config disables config search.
//
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir= --config-user-dir=%t/testdmode --no-default-config -c -### %s 2>&1 | FileCheck %s -check-prefix NO-DEFAULT-CONFIG
//
// NO-DEFAULT-CONFIG-NOT: Configuration file:

//--- Invocation qqq-clang-g++ tries to find config file qqq.cfg if qqq-clang-g++.cfg is not found.
//
// RUN: rm %t/testdmode/qqq-clang-g++.cfg
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir= --config-user-dir= -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix SHORT-NAME
//
// SHORT-NAME: Configuration file: {{.*}}/testdmode/qqq.cfg
// SHORT-NAME: -Werror
// SHORT-NAME-NOT: -Wundefined-func-template


//--- Config files are searched for in binary directory as well.
//
// RUN: mkdir %t/testbin
// RUN: ln -s %clang %t/testbin/clang
// RUN: echo "-Werror" > %t/testbin/aaa.cfg
// RUN: %t/testbin/clang --config-system-dir= --config-user-dir= --config aaa.cfg -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-BIN
//
// CHECK-BIN: Configuration file: {{.*}}/testbin/aaa.cfg
// CHECK-BIN: -Werror


//--- If command line contains options that change triple (for instance, -m32), clang tries
//    reloading config file.

//--- When reloading config file, x86_64-clang-g++ tries to find config i386-clang-g++.cfg first.
//
// RUN: mkdir %t/testreload
// RUN: ln -s %clang %t/testreload/x86_64-clang-g++
// RUN: echo "-Wundefined-func-template" > %t/testreload/i386-clang-g++.cfg
// RUN: echo "-Werror" > %t/testreload/i386.cfg
// RUN: echo "-Wall" > %t/testreload/x86_64-clang-g++.cfg
// RUN: %t/testreload/x86_64-clang-g++ --config-system-dir= --config-user-dir= -c -m32 -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD
//
// CHECK-RELOAD: Configuration file: {{.*}}/testreload/i386-clang-g++.cfg
// CHECK-RELOAD: -Wundefined-func-template
// CHECK-RELOAD-NOT: -Werror
// CHECK-RELOAD-NOT: -Wall

//--- Same for -target in place of -m32.
// RUN: %t/testreload/x86_64-clang-g++ --config-system-dir= --config-user-dir= -c -target i386 -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD

//--- `-target i386 -m64` should load the 64-bit config.
// RUN: %t/testreload/x86_64-clang-g++ --config-system-dir= --config-user-dir= -c -target i386 -m64 -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD1a
//
// CHECK-RELOAD1a: Configuration file: {{.*}}/testreload/x86_64-clang-g++.cfg
// CHECK-RELOAD1a: -Wall
// CHECK-RELOAD1a-NOT: -Werror
// CHECK-RELOAD1a-NOT: -Wundefined-func-template

//--- x86_64-clang-g++ tries to find config i386.cfg if i386-clang-g++.cfg is not found.
//
// RUN: rm %t/testreload/i386-clang-g++.cfg
// RUN: %t/testreload/x86_64-clang-g++ --config-system-dir= --config-user-dir= -c -m32 -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD1d
//
// CHECK-RELOAD1d: Configuration file: {{.*}}/testreload/i386.cfg
// CHECK-RELOAD1d: -Werror
// CHECK-RELOAD1d-NOT: -Wundefined-func-template
// CHECK-RELOAD1d-NOT: -Wall

//--- x86_64-clang-g++ uses x86_64-clang-g++.cfg if i386*.cfg are not found.
//
// RUN: rm %t/testreload/i386.cfg
// RUN: %t/testreload/x86_64-clang-g++ --config-system-dir= --config-user-dir= -c -m32 -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-RELOAD1a
