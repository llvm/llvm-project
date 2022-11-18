// REQUIRES: shell
// REQUIRES: x86-registered-target

// RUN: unset CLANG_NO_DEFAULT_CONFIG
// RUN: rm -rf %t && mkdir %t

//--- If config file is specified by relative path (workdir/cfg-s2), it is searched for by that path.

// RUN: mkdir -p %t/workdir/subdir
// RUN: echo "@subdir/cfg-s2" > %t/workdir/cfg-1
// RUN: echo "-Wundefined-var-template" > %t/workdir/subdir/cfg-s2
//
// RUN: ( cd %t && %clang --config=workdir/cfg-1 -c -### %s 2>&1 | FileCheck %s -check-prefix CHECK-REL )
//
// CHECK-REL: Configuration file: {{.*}}/workdir/cfg-1
// CHECK-REL: -Wundefined-var-template

//--- Config files are searched for in binary directory as well.
//
// RUN: mkdir %t/testbin
// RUN: ln -s %clang %t/testbin/clang
// RUN: echo "-Werror" > %t/testbin/aaa.cfg
// RUN: %t/testbin/clang --config-system-dir= --config-user-dir= --config=aaa.cfg -c -no-canonical-prefixes -### %s 2>&1 | FileCheck %s -check-prefix CHECK-BIN
//
// CHECK-BIN: Configuration file: {{.*}}/testbin/aaa.cfg
// CHECK-BIN: -Werror

//--- Invocation x86_64-unknown-linux-gnu-clang-g++ tries x86_64-unknown-linux-gnu-clang++.cfg first.
//
// RUN: mkdir %t/testdmode
// RUN: ln -s %clang %t/testdmode/cheribsd-riscv64-hybrid-clang++
// RUN: ln -s %clang %t/testdmode/qqq-clang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-clang
// RUN: ln -s %clang %t/testdmode/i386-unknown-linux-gnu-clang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-unknown-linux-gnu-clang-g++
// RUN: ln -s %clang %t/testdmode/x86_64-unknown-linux-gnu-clang
// RUN: touch %t/testdmode/cheribsd-riscv64-hybrid-clang++.cfg
// RUN: touch %t/testdmode/cheribsd-riscv64-hybrid.cfg
// RUN: touch %t/testdmode/qqq-clang-g++.cfg
// RUN: touch %t/testdmode/qqq.cfg
// RUN: touch %t/testdmode/x86_64-clang.cfg
// RUN: touch %t/testdmode/x86_64.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu-clang++.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu-clang-g++.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu-clang.cfg
// RUN: touch %t/testdmode/x86_64-unknown-linux-gnu.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu-clang++.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu-clang-g++.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu-clang.cfg
// RUN: touch %t/testdmode/i386-unknown-linux-gnu.cfg
// RUN: touch %t/testdmode/clang++.cfg
// RUN: touch %t/testdmode/clang-g++.cfg
// RUN: touch %t/testdmode/clang.cfg
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'
//
// FULL1: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-clang++.cfg

//--- -m32 overrides triple.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-I386 --implicit-check-not 'Configuration file:'
//
// FULL1-I386: Configuration file: {{.*}}/testdmode/i386-unknown-linux-gnu-clang++.cfg

//--- --target= also works for overriding triple.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --target=i386-unknown-linux-gnu --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-I386 --implicit-check-not 'Configuration file:'

//--- With --target= + -m64, -m64 takes precedence.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --target=i386-unknown-linux-gnu -m64 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- i386 prefix also works for 32-bit.
//
// RUN: %t/testdmode/i386-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-I386 --implicit-check-not 'Configuration file:'

//--- i386 prefix + -m64 also works for 64-bit.
//
// RUN: %t/testdmode/i386-unknown-linux-gnu-clang-g++ -m64 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- File specified by --config= is loaded after the one inferred from the executable.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --config=i386-qqq.cfg -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix EXPLICIT --implicit-check-not 'Configuration file:'
//
// EXPLICIT: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-clang++.cfg
// EXPLICIT-NEXT: Configuration file: {{.*}}/Inputs/config/i386-qqq.cfg

//--- --no-default-config --config= loads only specified file.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --no-default-config --config=i386-qqq.cfg -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix EXPLICIT-ONLY --implicit-check-not 'Configuration file:'
//
// EXPLICIT-ONLY: Configuration file: {{.*}}/Inputs/config/i386-qqq.cfg

//--- --no-default-config disables default filenames.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir=%S/Inputs/config --config-user-dir= --no-default-config -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix NO-CONFIG
//
// NO-CONFIG-NOT: Configuration file:

//--- --driver-mode= is respected.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --driver-mode=gcc --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-GCC --implicit-check-not 'Configuration file:'
//
// FULL1-GCC: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-clang.cfg

//--- "clang" driver symlink should yield the "*-clang" configuration file.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1-GCC --implicit-check-not 'Configuration file:'

//--- "clang" + --driver-mode= should yield "*-clang++".
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang --driver-mode=g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- Clang started via name prefix that is not valid is forcing that prefix instead of target triple.
//
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix QQQ --implicit-check-not 'Configuration file:'
//
// QQQ: Configuration file: {{.*}}/testdmode/qqq-clang-g++.cfg

//--- Explicit --target= overrides the triple even with non-standard name prefix.
//
// RUN: %t/testdmode/qqq-clang-g++ --target=x86_64-unknown-linux-gnu --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL1 --implicit-check-not 'Configuration file:'

//--- "x86_64" prefix does not form a valid triple either.
//
// RUN: %t/testdmode/x86_64-clang --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix X86_64 --implicit-check-not 'Configuration file:'
//
// X86_64: Configuration file: {{.*}}/testdmode/x86_64-clang.cfg

//--- Try cheribsd prefix using misordered triple components.
//
// RUN: %t/testdmode/cheribsd-riscv64-hybrid-clang++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix CHERIBSD --implicit-check-not 'Configuration file:'
//
// CHERIBSD: Configuration file: {{.*}}/testdmode/cheribsd-riscv64-hybrid-clang++.cfg

//--- Test fallback to x86_64-unknown-linux-gnu-clang-g++.cfg.
//
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu-clang++.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu-clang++.cfg
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL2 --implicit-check-not 'Configuration file:'
//
// FULL2: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu-clang-g++.cfg

//--- FULL2 + -m32.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL2-I386 --implicit-check-not 'Configuration file:'
//
// FULL2-I386: Configuration file: {{.*}}/testdmode/i386-unknown-linux-gnu-clang-g++.cfg

//--- Test fallback to x86_64-unknown-linux-gnu-clang.cfg + clang++.cfg.
//
// RUN: rm %t/testdmode/cheribsd-riscv64-hybrid-clang++.cfg
// RUN: rm %t/testdmode/qqq-clang-g++.cfg
// RUN: rm %t/testdmode/x86_64-clang.cfg
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu-clang-g++.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu-clang-g++.cfg
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu-clang.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu-clang.cfg
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3 --implicit-check-not 'Configuration file:'
//
// FULL3: Configuration file: {{.*}}/testdmode/clang++.cfg
// FULL3: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu.cfg

//--- FULL3 + -m32.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3-I386 --implicit-check-not 'Configuration file:'
//
// FULL3-I386: Configuration file: {{.*}}/testdmode/clang++.cfg
// FULL3-I386: Configuration file: {{.*}}/testdmode/i386-unknown-linux-gnu.cfg

//--- FULL3 + --driver-mode=.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --driver-mode=gcc --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL3-GCC --implicit-check-not 'Configuration file:'
//
// FULL3-GCC: Configuration file: {{.*}}/testdmode/clang.cfg
// FULL3-GCC: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu.cfg

//--- QQQ fallback.
//
// RUN: %t/testdmode/qqq-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix QQQ-FALLBACK --implicit-check-not 'Configuration file:'
//
// QQQ-FALLBACK: Configuration file: {{.*}}/testdmode/clang++.cfg
// QQQ-FALLBACK: Configuration file: {{.*}}/testdmode/qqq.cfg

//--- "x86_64" falback.
//
// RUN: %t/testdmode/x86_64-clang --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix X86_64-FALLBACK --implicit-check-not 'Configuration file:'
//
// X86_64-FALLBACK: Configuration file: {{.*}}/testdmode/clang.cfg
// X86_64-FALLBACK: Configuration file: {{.*}}/testdmode/x86_64.cfg

//--- cheribsd fallback.
//
// RUN: %t/testdmode/cheribsd-riscv64-hybrid-clang++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix CHERIBSD-FALLBACK --implicit-check-not 'Configuration file:'
//
// CHERIBSD-FALLBACK: Configuration file: {{.*}}/testdmode/clang++.cfg
// CHERIBSD-FALLBACK: Configuration file: {{.*}}/testdmode/cheribsd-riscv64-hybrid.cfg

//--- Test fallback to x86_64-unknown-linux-gnu.cfg + clang-g++.cfg.
//
// RUN: rm %t/testdmode/clang++.cfg
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL4 --implicit-check-not 'Configuration file:'
//
// FULL4: Configuration file: {{.*}}/testdmode/clang-g++.cfg
// FULL4: Configuration file: {{.*}}/testdmode/x86_64-unknown-linux-gnu.cfg

//--- Test fallback to clang-g++.cfg if x86_64-unknown-linux-gnu-clang.cfg does not exist.
//
// RUN: rm %t/testdmode/x86_64-unknown-linux-gnu.cfg
// RUN: rm %t/testdmode/i386-unknown-linux-gnu.cfg
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL5 --implicit-check-not 'Configuration file:'
//
// FULL5: Configuration file: {{.*}}/testdmode/clang-g++.cfg

//--- FULL5 + -m32.
//
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ -m32 --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix FULL5-I386 --implicit-check-not 'Configuration file:'
//
// FULL5-I386: Configuration file: {{.*}}/testdmode/clang-g++.cfg

//--- Test that incorrect driver mode config file is not used.
//
// RUN: rm %t/testdmode/clang-g++.cfg
// RUN: %t/testdmode/x86_64-unknown-linux-gnu-clang-g++ --config-system-dir= --config-user-dir= -no-canonical-prefixes --version 2>&1 | FileCheck %s -check-prefix NO-CONFIG

//--- Tilde expansion in user configuration file directory
//
// RUN: HOME=%S/Inputs/config %clang --config-user-dir=~ -v 2>&1 | FileCheck %s -check-prefix CHECK-TILDE
// CHECK-TILDE: User configuration file directory: {{.*}}/Inputs/config
