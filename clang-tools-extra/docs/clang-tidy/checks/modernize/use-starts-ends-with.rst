.. title:: clang-tidy - modernize-use-starts-ends-with

modernize-use-starts-ends-with
==============================

Checks for common roundabout ways to express ``starts_with`` and ``ends_with``
and suggests replacing with ``starts_with`` when the method is available.
Notably, this will work with ``std::string`` and ``std::string_view``.

.. code-block:: c++

  std::string s = "...";
  if (s.find("prefix") == 0) { /* do something */ }
  if (s.rfind("prefix", 0) == 0) { /* do something */ }
  if (s.compare(0, strlen("prefix"), "prefix") == 0) { /* do something */ }

becomes

.. code-block:: c++

  std::string s = "...";
  if (s.starts_with("prefix")) { /* do something */ }
  if (s.starts_with("prefix")) { /* do something */ }
  if (s.starts_with("prefix")) { /* do something */ }
