.. title:: clang-tidy - readability-container-contains

readability-container-contains
==============================

Finds usages of ``container.count()`` and
``container.find() == container.end()`` which should be replaced by a call to
the ``container.contains()`` method.

Whether an element is contained inside a container should be checked with
``contains`` instead of ``count``/``find`` because ``contains`` conveys the
intent more clearly. Furthermore, for containers which permit multiple entries
per key (``multimap``, ``multiset``, ...), ``contains`` is more efficient than
``count`` because ``count`` has to do unnecessary additional work.

Examples:

======================================  =====================================
Initial expression                      Result
--------------------------------------  -------------------------------------
``myMap.find(x) == myMap.end()``        ``!myMap.contains(x)``
``myMap.find(x) != myMap.end()``        ``myMap.contains(x)``
``if (myMap.count(x))``                 ``if (myMap.contains(x))``
``bool exists = myMap.count(x)``        ``bool exists = myMap.contains(x)``
``bool exists = myMap.count(x) > 0``    ``bool exists = myMap.contains(x)``
``bool exists = myMap.count(x) >= 1``   ``bool exists = myMap.contains(x)``
``bool missing = myMap.count(x) == 0``  ``bool missing = !myMap.contains(x)``
======================================  =====================================

This check will apply to any class that has a ``contains`` method, notably
including ``std::set``, ``std::unordered_set``, ``std::map``, and
``std::unordered_map`` as of C++20, and ``std::string`` and ``std::string_view``
as of C++23.
