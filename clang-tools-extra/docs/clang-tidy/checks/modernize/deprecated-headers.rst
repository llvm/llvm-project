.. title:: clang-tidy - modernize-deprecated-headers

modernize-deprecated-headers
============================

There exist headers that produce no effect when included, which are there
solely to ease migrating code. The check will suggest removing them.
In C++, they are:

* ``stdalign.h`` / ``cstdalign``
* ``stdbool.h`` / ``cstdbool``
* ``iso646.h`` / ``ciso646``

And in C they are:

* ``stdalign.h`` // No-op since C23
* ``stdbool.h`` // No-op since C23
* ``stdnoreturn.h`` // No-op since C23

In C++, there is additionally a number of headers intended for
interoperability with C, which should not be used in pure C++ code.
The check will suggest replacing them with their C++ counterparts
(e.g. replacing ``<signal.h>`` with ``<csignal>``). These headers are:

* ``<assert.h>``
* ``<complex.h>``
* ``<ctype.h>``
* ``<errno.h>``
* ``<fenv.h>``     // deprecated since C++11
* ``<float.h>``
* ``<inttypes.h>``
* ``<limits.h>``
* ``<locale.h>``
* ``<math.h>``
* ``<setjmp.h>``
* ``<signal.h>``
* ``<stdarg.h>``
* ``<stddef.h>``
* ``<stdint.h>``
* ``<stdio.h>``
* ``<stdlib.h>``
* ``<string.h>``
* ``<tgmath.h>``   // deprecated since C++11
* ``<time.h>``
* ``<uchar.h>``    // deprecated since C++11
* ``<wchar.h>``
* ``<wctype.h>``

Important note: the C++ headers are not identical to their C counterparts.
The C headers provide names in the global namespace (e.g. ``<stdio.h>``
provides ``printf``), but the C++ headers might provide them only in the
``std`` namespace (e.g. ``<cstdio>`` provides ``std::printf``, but not
necessarily ``printf``). The check can break code that uses the unqualified
names.

.. code-block:: c++

  // C++ source file...
  #include <assert.h>
  #include <stdbool.h>

  // becomes

  #include <cassert>
  // No 'stdbool.h' here.

The check will ignore `include` directives within `extern "C" { ... }`
blocks, under the assumption that such code is an API meant to compile as
both C and C++:

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
   that the header files in the project are only used by C++ source files.
   Default is `false`.
