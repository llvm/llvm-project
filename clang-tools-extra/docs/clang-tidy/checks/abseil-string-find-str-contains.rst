.. title:: clang-tidy - abseil-string-find-str-contains

abseil-string-find-str-contains
===============================

Finds ``s.find(...) == string::npos`` comparisons (for various string-like types)
and suggests replacing with ``absl::StrContains()``.

This improves readability and reduces the likelihood of accidentally mixing
``find()`` and ``npos`` from different string-like types.

By default, "string-like types" includes ``::std::basic_string``,
``::std::basic_string_view``, and ``::absl::string_view``.  See the
StringLikeClasses option to change this.

.. code-block:: c++

  std::string s = "...";
  if (s.find("Hello World") == std::string::npos) { /* do something */ }

  absl::string_view a = "...";
  if (absl::string_view::npos != a.find("Hello World")) { /* do something */ }

becomes

.. code-block:: c++

  std::string s = "...";
  if (!absl::StrContains(s, "Hello World")) { /* do something */ }

  absl::string_view a = "...";
  if (absl::StrContains(a, "Hello World")) { /* do something */ }


Options
-------

.. option:: StringLikeClasses

   Semicolon-separated list of names of string-like classes. By default includes
   ``::std::basic_string``, ``::std::basic_string_view``, and
   ``::absl::string_view``.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`. Default
   is `llvm`.

.. option:: AbseilStringsMatchHeader

   The location of Abseil's ``strings/match.h``. Defaults to
   ``absl/strings/match.h``.
