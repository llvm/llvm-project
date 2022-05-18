.. title:: clang-tidy - bugprone-no-escape

bugprone-no-escape
==================

Finds pointers with the ``noescape`` attribute that are captured by an
asynchronously-executed block. The block arguments in ``dispatch_async()`` and
``dispatch_after()`` are guaranteed to escape, so it is an error if a pointer with the
``noescape`` attribute is captured by one of these blocks.

The following is an example of an invalid use of the ``noescape`` attribute.

  .. code-block:: objc

    void foo(__attribute__((noescape)) int *p) {
      dispatch_async(queue, ^{
        *p = 123;
      });
    });
