.. title:: clang-tidy - modernize-replace-with-std-copy

modernize-replace-with-std-copy
===================================

This check will flag all calls to ``memmove`` that can be possibly replaced with ``std::copy_n``.
In some specific cases it will provide a fix automatically.
``memcpy`` may also be flagged as opt-in. It is disabled by default because it performs no safety checks for overlapping ranges
in the way ``memmove`` and ``std::copy_n`` do.
``wmemmove`` and ``wmemcpy`` are also supported.

Example:

.. code-block:: c++
  std::vector<int> dst(64);
  memcpy(dst.data(), std::data(src), N);

becomes

.. code-block:: c++
  std::vector<int> dst(64);
  std::copy_n(std::cbegin(src), (N) / sizeof(int), std::begin(dst));

Known limitations
----------------
For now, the check works only on a limited, recognizable subset of calls, where it can infer the arguments are pointers to valid collections
in the sense that ``std::copy_n`` understands. More specifically, source/destination should be one of:
- a call to ``std::data`` or the corresponding member method.
- a fixed-size C-array.

Moreover, a fix will not be issued in more complicated cases, e.g. when source and destination are collections of types that have different sizes.

Options
-------

.. option:: IncludeStyle
   A string specifying which include-style is used, `llvm` or `google`. Default is `llvm`.

.. option:: FlagMemcpy
   A boolean specifying whether to flag calls to ``memcpy`` as well. Default is `false`.