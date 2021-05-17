.. title:: clang-tidy - bugprone-implicit-widening-of-multiplication-result

bugprone-implicit-widening-of-multiplication-result
===================================================

The check diagnoses instances where a result of a multiplication is implicitly
widened, and suggests (with fix-it) to either silence the code by making
widening explicit, or to perform the multiplication in a wider type,
to avoid the widening afterwards.

This is mainly useful when operating on a very large buffers.
For example, consider:

.. code-block:: c++

  void zeroinit(char* base, unsigned width, unsigned height) {
    for(unsigned row = 0; row != height; ++row) {
      for(unsigned col = 0; col != width; ++col) {
        char* ptr = base + row * width + col;
        *ptr = 0;
      }
    }
  }

This is fine in general, but iff ``width * height`` overflows,
you end up wrapping back to the beginning of ``base``
instead of processing the entire requested buffer.

Indeed, this only matters for pretty large buffers (4GB+),
but that can happen very easily for example in image processing,
where for that to happen you "only" need a ~269MPix image.


Options
-------

.. option:: UseCXXStaticCastsInCppSources

   When suggesting fix-its for C++ code, should C++-style ``static_cast<>()``'s
   be suggested, or C-style casts. Defaults to ``true``.

.. option:: UseCXXHeadersInCppSources

   When suggesting to include the appropriate header in C++ code,
   should ``<cstddef>`` header be suggested, or ``<stddef.h>``.
   Defaults to ``true``.


Examples:

.. code-block:: c++

  long mul(int a, int b) {
    return a * b; // warning: performing an implicit widening conversion to type 'long' of a multiplication performed in type 'int'
  }

  char* ptr_add(char *base, int a, int b) {
    return base + a * b; // warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ssize_t'
  }

  char ptr_subscript(char *base, int a, int b) {
    return base[a * b]; // warning: result of multiplication in type 'int' is used as a pointer offset after an implicit widening conversion to type 'ssize_t'
  }
