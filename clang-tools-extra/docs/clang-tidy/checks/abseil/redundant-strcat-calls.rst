.. title:: clang-tidy - abseil-redundant-strcat-calls

abseil-redundant-strcat-calls
=============================

Suggests removal of unnecessary calls to ``absl::StrCat`` when the result is
being passed to another call to ``absl::StrCat`` or ``absl::StrAppend``.

The extra calls cause unnecessary temporary strings to be constructed. Removing
them makes the code smaller and faster.

Examples:

.. code-block:: c++

  std::string s = absl::StrCat("A", absl::StrCat("B", absl::StrCat("C", "D")));
  //before

  std::string s = absl::StrCat("A", "B", "C", "D");
  //after

  absl::StrAppend(&s, absl::StrCat("E", "F", "G"));
  //before

  absl::StrAppend(&s, "E", "F", "G");
  //after
