.. title:: clang-tidy - readability-container-contains

readability-container-contains
==============================

Finds usages of ``container.count()`` and ``container.find() == container.end()`` which should be replaced by a call to the ``container.contains()`` method introduced in C++ 20.

Whether an element is contained inside a container should be checked with ``contains`` instead of ``count``/``find`` because ``contains`` conveys the intent more clearly. Furthermore, for containers which permit multiple entries per key (``multimap``, ``multiset``, ...), ``contains`` is more efficient than ``count`` because ``count`` has to do unnecessary additional work.

Examples:

===========================================  ==============================
Initial expression                           Result
-------------------------------------------  ------------------------------
``myMap.find(x) == myMap.end()``             ``!myMap.contains(x)``
``myMap.find(x) != myMap.end()``             ``myMap.contains(x)``
``if (myMap.count(x))``                      ``if (myMap.contains(x))``
``bool exists = myMap.count(x)``             ``bool exists = myMap.contains(x)``
``bool exists = myMap.count(x) > 0``         ``bool exists = myMap.contains(x)``
``bool exists = myMap.count(x) >= 1``        ``bool exists = myMap.contains(x)``
``bool missing = myMap.count(x) == 0``       ``bool missing = !myMap.contains(x)``
===========================================  ==============================

This check applies to ``std::set``, ``std::unordered_set``, ``std::map``, ``std::unordered_map`` and the corresponding multi-key variants.
It is only active for C++20 and later, as the ``contains`` method was only added in C++20.
