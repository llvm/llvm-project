.. title:: clang-tidy - abseil-string-find-startswith

abseil-string-find-startswith
=============================

Checks whether a ``std::string::find()`` result is compared with 0, and
suggests replacing with ``absl::StartsWith()``. This is both a readability and
performance issue.

.. code-block:: c++

  string s = "...";
  if (s.find("Hello World") == 0) { /* do something */ }

becomes


.. code-block:: c++

  string s = "...";
  if (absl::StartsWith(s, "Hello World")) { /* do something */ }


Options
-------

.. option:: StringLikeClasses

   Semicolon-separated list of names of string-like classes. By default only
   ``std::basic_string`` is considered. The list of methods to considered is
   fixed.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: AbseilStringsMatchHeader

   The location of Abseil's ``strings/match.h``. Defaults to
   ``absl/strings/match.h``.
