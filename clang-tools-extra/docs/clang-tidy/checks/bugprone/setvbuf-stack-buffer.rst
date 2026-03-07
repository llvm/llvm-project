.. title:: clang-tidy - bugprone-setvbuf-stack-buffer

bugprone-setvbuf-stack-buffer
=============================

Warns when ``setvbuf()`` is called with a stack-allocated buffer.

The C standard (C11 §7.21.5.6) requires that the buffer passed to ``setvbuf()``
must have a lifetime at least as long as the open stream. Passing a local
(automatic storage duration) buffer leads to undefined behavior when the function
returns but the stream remains open — the stream will continue to use the
now-dangling buffer.

.. code-block:: c

   void bad(void) {
     char buf[BUFSIZ];
     setvbuf(stdout, buf, _IOFBF, BUFSIZ);  // warning
     // buf goes out of scope, but stdout keeps using it!
   }

Safe alternatives:

- ``static`` local buffer: ``static char buf[BUFSIZ];``
- Global buffer: ``char buf[BUFSIZ];`` at file scope
- Dynamically allocated: ``char *buf = malloc(BUFSIZ);``
- Unbuffered: ``setvbuf(stream, NULL, _IONBF, 0);``
