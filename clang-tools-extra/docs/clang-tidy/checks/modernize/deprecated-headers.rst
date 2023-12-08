.. title:: clang-tidy - modernize-deprecated-headers

modernize-deprecated-headers
============================

Some headers from C library were deprecated in C++ and are no longer welcome in
C++ codebases. Some have no effect in C++. For more details refer to the C++ 14
Standard [depr.c.headers] section.

This check replaces C standard library headers with their C++ alternatives and
removes redundant ones.

.. code-block:: c++

  // C++ source file...
  #include <assert.h>
  #include <stdbool.h>

  // becomes

  #include <cassert>
  // No 'stdbool.h' here.

Important note: the Standard doesn't guarantee that the C++ headers declare all
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

The checker ignores `include` directives within `extern "C" { ... }` blocks,
since a library might want to expose some API for C and C++ libraries.

.. code-block:: c++

  // C++ source file...
  extern "C" {
  #include <assert.h>  // Left intact.
  #include <stdbool.h> // Left intact.
  }

Options
-------

.. option:: CheckHeaderFile

   `clang-tidy` cannot know if the header file included by the currently
   analyzed C++ source file is not included by any other C source files.
   Hence, to omit false-positives and wrong fixit-hints, we ignore emitting
   reports into header files. One can set this option to `true` if they know
   that the header files in the project are only used by C++ source file.
   Default is `false`.
