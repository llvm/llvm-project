.. title:: clang-tidy - abseil-string-find-startswith

abseil-string-find-startswith
=============================

Checks whether a ``std::string::find()`` or ``std::string::rfind()`` (and
corresponding ``std::string_view`` methods) result is compared with 0, and
suggests replacing with ``absl::StartsWith()``. This is both a readability and
performance issue.

``starts_with`` was added as a built-in function on those types in C++20. If
available, prefer enabling :doc:`modernize-use-starts-ends-with
<../modernize/use-starts-ends-with>` instead of this check.

.. code-block:: c++

  string s = "...";
  if (s.find("Hello World") == 0) { /* do something */ }
  if (s.rfind("Hello World", 0) == 0) { /* do something */ }

becomes


.. code-block:: c++

  string s = "...";
  if (absl::StartsWith(s, "Hello World")) { /* do something */ }
  if (absl::StartsWith(s, "Hello World")) { /* do something */ }


Options
-------

.. option:: StringLikeClasses

   Semicolon-separated list of names of string-like classes. By default both
   ``std::basic_string`` and ``std::basic_string_view`` are considered. The list
   of methods to be considered is fixed.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: AbseilStringsMatchHeader

   The location of Abseil's ``strings/match.h``. Defaults to
   ``absl/strings/match.h``.
