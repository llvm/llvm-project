.. title:: clang-tidy - modernize-deprecated-headers

modernize-deprecated-headers
============================

Some headers from C library were deprecated in C++ and are no longer welcome in
C++ codebases. Some have no effect in C++. For more details refer to the C++ 14
Standard [depr.c.headers] section.

This check replaces C standard library headers with their C++ alternatives and
removes redundant ones.

Improtant note: the Standard doesn't guarantee that the C++ headers declare all
the same functions in the global namespace. The check in its current form can
break the code that uses library symbols from the global namespace.

* `<assert.h>`
* `<complex.h>`
* `<ctype.h>`
* `<errno.h>`
* `<fenv.h>`     // deprecated since C++11
* `<float.h>`
* `<inttypes.h>`
* `<limits.h>`
* `<locale.h>`
* `<math.h>`
* `<setjmp.h>`
* `<signal.h>`
* `<stdarg.h>`
* `<stddef.h>`
* `<stdint.h>`
* `<stdio.h>`
* `<stdlib.h>`
* `<string.h>`
* `<tgmath.h>`   // deprecated since C++11
* `<time.h>`
* `<uchar.h>`    // deprecated since C++11
* `<wchar.h>`
* `<wctype.h>`

If the specified standard is older than C++11 the check will only replace
headers deprecated before C++11, otherwise -- every header that appeared in
the previous list.

These headers don't have effect in C++:

* `<iso646.h>`
* `<stdalign.h>`
* `<stdbool.h>`
