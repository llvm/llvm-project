// Basic test
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/a/b
// RUN: echo > %t.dir/a/b/x.h
// RUN: cd %t.dir
// RUN: %clang -MD -MF - %s -fsyntax-only -I a/b | FileCheck -check-prefix=CHECK-ONE %s
// CHECK-ONE: {{ }}a{{[/\\]}}b{{[/\\]}}x.h

// PR8974 (-include flag)
// RUN: %clang -MD -MF - %s -fsyntax-only -include a/b/x.h -DINCLUDE_FLAG_TEST | FileCheck -check-prefix=CHECK-TWO %s
// CHECK-TWO: {{ }}a{{[/\\]}}b{{[/\\]}}x.h

// rdar://problem/9734352 (paths involving ".")
// RUN: %clang -MD -MF - %s -fsyntax-only -I ./a/b | FileCheck -check-prefix=CHECK-THREE %s
// CHECK-THREE: {{ }}a{{[/\\]}}b{{[/\\]}}x.h
// RUN: %clang -MD -MF - %s -fsyntax-only -I .//./a/b/ | FileCheck -check-prefix=CHECK-FOUR %s
// CHECK-FOUR: {{ }}a{{[/\\]}}b{{[/\\]}}x.h
// RUN: %clang -MD -MF - %s -fsyntax-only -I a/b/. | FileCheck -check-prefix=CHECK-FIVE %s
// CHECK-FIVE: {{ }}a{{[/\\]}}b{{[/\\]}}.{{[/\\]}}x.h
// RUN: cd a/b
// RUN: %clang -MD -MF - %s -fsyntax-only -I ./ | FileCheck -check-prefix=CHECK-SIX %s
// CHECK-SIX: {{ }}x.h
// RUN: echo "fun:foo" > %t.ignorelist
// RUN: %clang -MD -MF - %s -fsyntax-only -resource-dir=%S/Inputs/resource_dir_with_sanitizer_ignorelist -fsanitize=undefined -flto -fvisibility=hidden -fsanitize-ignorelist=%t.ignorelist -I ./ | FileCheck -check-prefix=CHECK-SEVEN %s
// CHECK-SEVEN: .ignorelist
// CHECK-SEVEN: {{ }}x.h
#ifndef INCLUDE_FLAG_TEST
#include <x.h>
#endif

// RUN: echo "fun:foo" > %t.ignorelist1
// RUN: echo "fun:foo" > %t.ignorelist2
// RUN: %clang -MD -MF - %s -fsyntax-only -resource-dir=%S/Inputs/resource_dir_with_sanitizer_ignorelist -fsanitize=undefined -flto -fvisibility=hidden -fsanitize-ignorelist=%t.ignorelist1 -fsanitize-ignorelist=%t.ignorelist2 -I ./ | FileCheck -check-prefix=TWO-IGNORE-LISTS %s
// TWO-IGNORE-LISTS: dependency-gen.o:
// TWO-IGNORE-LISTS-DAG: ignorelist1
// TWO-IGNORE-LISTS-DAG: ignorelist2
// TWO-IGNORE-LISTS-DAG: x.h
// TWO-IGNORE-LISTS-DAG: dependency-gen.c

// RUN: %clang -MD -MF - %s -fsyntax-only -resource-dir=%S/Inputs/resource_dir_with_sanitizer_ignorelist -fsanitize=undefined -flto -fvisibility=hidden -I ./ | FileCheck -check-prefix=USER-AND-SYS-DEPS %s
// USER-AND-SYS-DEPS: dependency-gen.o:
// USER-AND-SYS-DEPS-DAG: ubsan_ignorelist.txt

// RUN: %clang -MMD -MF - %s -fsyntax-only -resource-dir=%S/Inputs/resource_dir_with_sanitizer_ignorelist -fsanitize=undefined -flto -fvisibility=hidden -I ./ | FileCheck -check-prefix=ONLY-USER-DEPS %s
// ONLY-USER-DEPS: dependency-gen.o:
// NOT-ONLY-USER-DEPS: ubsan_ignorelist.txt
