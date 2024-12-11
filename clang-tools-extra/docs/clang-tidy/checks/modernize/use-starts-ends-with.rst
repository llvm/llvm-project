.. title:: clang-tidy - modernize-use-starts-ends-with

modernize-use-starts-ends-with
==============================

Checks for common roundabout ways to express ``starts_with`` and ``ends_with``
and suggests replacing with the simpler method when it is available. Notably, 
this will work with ``std::string`` and ``std::string_view``.

Covered scenarios:

==================================================== =====================
Expression                                           Replacement
---------------------------------------------------- ---------------------
``u.find(v) == 0``                                   ``u.starts_with(v)``
``u.rfind(v, 0) != 0``                               ``!u.starts_with(v)``
``u.compare(0, v.size(), v) == 0``                   ``u.starts_with(v)``
``u.substr(0, v.size()) == v``                       ``u.starts_with(v)``
``v != u.substr(0, v.size())``                       ``!u.starts_with(v)``
``u.compare(u.size() - v.size(), v.size(), v) == 0`` ``u.ends_with(v)``
``u.rfind(v) == u.size() - v.size()``                ``u.ends_with(v)``
==================================================== =====================
