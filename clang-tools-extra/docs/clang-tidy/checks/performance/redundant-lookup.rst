.. title:: clang-tidy - performance-redundant-lookup

performance-redundant-lookup
============================

This check warns about potential redundant container lookup operations within
a function.

Examples
--------

.. code-block:: c++

    if (map.count(key) && map[key] < threshold) {
      // do stuff
    }

In this example, we would check if the key is present in the map,
and then do a second lookup to actually load the value.
We could refactor the code into this, to use a single lookup:

.. code-block:: c++

    if (auto it = map.find(key); it != map.end() && it->second < threshold) {
      // do stuff
    }

In this example, we do three lookups while calculating, caching and then
using the result of some expensive computation:

.. code-block:: c++

    if (!cache.contains(key)) {
        cache[key] = computeExpensiveValue();
    }
    use(cache[key]);

We could refactor this code using ``try_emplace`` to fill up the cache entry
if wasn't present, and just use it if we already computed the value.
This way we would only have a single unavoidable lookup:

.. code-block:: c++

    auto [cacheSlot, inserted] cache.try_emplace(key);
    if (inserted) {
        cacheSlot->second = computeExpensiveValue();
    }
    use(cacheSlot->second);


What is a "lookup"?
-------------------

All container operations that walk the internal structure of the container
should be considered as "lookups".
This means that checking if an element is present or inserting an element
is also considered as a "lookup".

For example, ``contains``, ``count`` but even the ``operator[]``
should be considered as "lookups".

Technically ``insert``, ``emplace`` or ``try_emplace`` are also lookups,
even if due to limitations, they are not recognized as such.

Lookups inside macros are ignored, thus not considered as "lookups".
For example:

.. code-block:: c++

    assert(map.count(key) == 0); // Not considered as a "lookup".

Options
-------

.. option:: ContainerNameRegex

   The regular expression matching the type of the container objects.
   This is matched in a case insensitive manner.
   Default is `set|map`.

.. option:: LookupMethodNames

   Member function names to consider as **lookup** operation.
   These methods must have exactly 1 argument.
   Default is `at;contains;count;find_as;find`.

Limitations
-----------

 - The "redundant lookups" may span across a large chunk of code.
   Such reports can be considered as false-positives because it's hard to judge
   if the container is definitely not mutated between the lookups.
   It would be hard to split the lookup groups in a stable and meaningful way,
   and a threshold for proximity would be just an arbitrary limit.

 - The "redundant lookups" may span across different control-flow constructs,
   making it impossible to refactor.
   It may be that the code was deliberately structured like it was, thus the
   report is considered a false-positive.
   Use your best judgement to see if anything needs to be fixed or not.
   For example:

   .. code-block:: c++

    if (coin())
        map[key] = foo();
    else
        map[key] = bar();

   Could be refactored into:

   .. code-block:: c++

    map[key] = coin() ? foo() : bar();

   However, the following code could be considered intentional:

   .. code-block:: c++

    // Handle the likely case.
    if (auto it = map.find(key); it != map.end()) {
        return process(*it);
    }

    // Commit the dirty items, and check again.
    for (const auto &item : dirtyList) {
        commit(item, map); // Updates the "map".
    }

    // Do a final check.
    if (auto it = map.find(key); it != map.end()) {
        return process(*it);
    }

 - The key argument of a lookup may have sideffects. Sideffects are ignored when identifying lookups.
   This can introduce some false-positives. For example:

   .. code-block:: c++

    m.contains(rng(++n));
    m.contains(rng(++n)); // FP: This is considered a redundant lookup.

 - Lookup member functions must have exactly 1 argument to match.
   There are technically lookup functions, such as ``insert`` or ``try_emplace``,
   but it would be hard to identify the "key" part of the argument,
   while leaving the implementation open for user-configuration via the
   `LookupMethodNames` option.
