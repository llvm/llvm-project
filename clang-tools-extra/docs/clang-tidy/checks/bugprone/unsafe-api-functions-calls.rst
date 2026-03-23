.. title:: clang-tidy - bugprone-unsafe-api-functions-calls

bugprone-unsafe-api-functions-calls
===================================

Checks for C standard function calls that are used in an undefined or
unsafe manner.

Unsafe usage of ``setvbuf()`` or ``setbuf()``
---------------------------------------------

Enabling this check will warn when ``setvbuf()`` or ``setbuf()`` is
called with a stack-allocated buffer.

The C standard (`C11 §7.21.5.6`_) requires that the buffer passed to
``setvbuf()`` must have a lifetime at least as long as the open
stream. Since ``setbuf(stream, buf)`` is defined to be equivalent to
``(void)setvbuf(stream, buf, _IOFBF, BUFSIZ)`` (`C11 §7.21.5.5`_), the
same requirement applies.

.. _C11 §7.21.5.6: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1548.pdf
.. _C11 §7.21.5.5: https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1548.pdf

Passing a local (automatic storage duration) buffer leads to undefined
behavior when the function returns but the stream remains open. After
the return the stream will continue to use the now-dangling buffer.

.. code-block:: c

   void bad_setvbuf(void) {
     char buf[BUFSIZ];
     setvbuf(stdout, buf, _IOFBF, BUFSIZ);  // warning
     // buf goes out of scope, but stdout keeps using it!
   }

   void bad_setbuf(void) {
     char buf[BUFSIZ];
     setbuf(stdout, buf);  // warning
     // buf goes out of scope, but stdout keeps using it!
   }

Safe alternatives:

- ``static`` local buffer: ``static char buf[BUFSIZ];``
- Global buffer: ``char buf[BUFSIZ];`` at file scope
- Dynamically allocated: ``char *buf = malloc(BUFSIZ);``
- Unbuffered: ``setvbuf(stream, NULL, _IONBF, 0);`` or ``setbuf(stream, NULL);``
