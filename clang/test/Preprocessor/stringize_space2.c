/* RUN: %clang_cc1 -E %s | grep 'a c'
 */
#define t(x) #x
t(a
c)

