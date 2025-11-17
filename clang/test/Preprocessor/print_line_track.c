/* RUN: %clang_cc1 -E %s | awk '/a/,/3/{print; exit 0} {exit 1}'
 * RUN: %clang_cc1 -E %s | awk '/b/,/16/{print; exit 0} {exit 1}'
 * RUN: %clang_cc1 -E -P %s | awk '/a/,/3/{print; exit 0} {exit 1}'
 * RUN: %clang_cc1 -E -P %s | awk '/b/,/16/{print; exit 0} {exit 1}'
 * RUN: %clang_cc1 -E %s | not grep '# 0 '
 * RUN: %clang_cc1 -E -P %s | count 4
 * PR1848 PR3437 PR7360
*/

#define t(x) x

t(a
3)

t(b
__LINE__)
