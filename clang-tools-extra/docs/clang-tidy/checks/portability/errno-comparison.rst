.. title:: clang-tidy - portability-errno-comparison

portability-errno-comparison
============================

Flags comparisons of ``errno`` against an integer literal.

The values of the ``errno`` error constants are implementation-defined, so
comparing ``errno`` against a hard-coded number such as ``errno == 5`` is not
portable. Use the ``E``-prefixed macros (e.g. ``EINVAL``) instead.

.. code-block:: c

  if (errno == 5) {}       // warning
  if (errno == EINVAL) {}  // ok, compared against the named macro
  if (errno == 0) {}       // ok, 0 is the standard "no error" value

Comparisons with ``0`` and comparisons whose literal comes from a macro (such as
the ``E``-prefixed constants themselves) are not flagged.
