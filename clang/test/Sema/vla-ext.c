/* RUN: %clang_cc1 -verify=off -std=c89 %s
 * RUN: %clang_cc1 -verify=off -Wall -std=c89 %s
 * RUN: %clang_cc1 -verify -pedantic -std=c89 %s
 * RUN: %clang_cc1 -verify -Wvla-extension -std=c89 %s
 * RUN: %clang_cc1 -verify=off -Wvla-cxx-extension -std=c89 %s
 * RUN: %clang_cc1 -verify=off -pedantic -std=c99 %s
 * RUN: %clang_cc1 -verify=off -Wall -std=c99 %s
 * RUN: %clang_cc1 -verify=off -std=c99 -Wvla-extension %s
 * The next run line still issues the extension warning because VLAs are an
 * extension in C89, but the line after it will issue the congratulatory
 * diagnostic.
 * RUN: %clang_cc1 -verify -Wvla -std=c89 %s
 * RUN: %clang_cc1 -verify=wvla -Wvla -std=c99 %s
 */

/* off-no-diagnostics */

void func(int n) {
  int array[n]; /* expected-warning {{variable length arrays are a C99 feature}}
                   wvla-warning {{variable length array used}}
                 */
  (void)array;
}

