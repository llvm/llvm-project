/* Clang supports a very limited subset of -traditional-cpp, basically we only
 * intend to add support for things that people actually rely on when doing
 * things like using /usr/bin/cpp to preprocess non-source files. */

/*
 RUN: %clang_cc1 -traditional-cpp %s -E -o %t
 RUN: FileCheck < %t %s
*/

/* CHECK: foo // bar
 */
foo // bar
