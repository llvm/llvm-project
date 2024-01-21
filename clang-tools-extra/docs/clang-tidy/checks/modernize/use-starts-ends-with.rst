.. title:: clang-tidy - modernize-use-starts-ends-with

modernize-use-starts-ends-with
==============================

Checks whether a ``find`` or ``rfind`` result is compared with 0 and suggests
replacing with ``starts_with`` when the method exists in the class. Notably,
this will work with ``std::string`` and ``std::string_view``.

.. code-block:: c++

  std::string s = "...";
  if (s.find("prefix") == 0) { /* do something */ }
  if (s.rfind("prefix", 0) == 0) { /* do something */ }

becomes

.. code-block:: c++

  std::string s = "...";
  if (s.starts_with("prefix")) { /* do something */ }
  if (s.starts_with("prefix")) { /* do something */ }
