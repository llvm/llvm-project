// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic %s
// expected-no-diagnostics

/* WG14 N3262: Yes
 * Usability of a byte-wise copy of va_list
 *
 * NB: Clang explicitly documents this as being undefined behavior. A
 * diagnostic is produced for some targets but not for others for assignment or
 * initialization, but no diagnostic is possible to produce for use with memcpy
 * in the general case, nor with a manual bytewise copy via a for loop.
 *
 * Therefore, nothing is tested in this file; it serves as a reminder that we
 * validated our documentation against the paper. See
 * clang/docs/LanguageExtensions.rst for more details.
 *
 * FIXME: it would be nice to add ubsan support for recognizing when an invalid
 * copy is made and diagnosing on copy (or on use of the copied va_list).
 */

int main() {}
