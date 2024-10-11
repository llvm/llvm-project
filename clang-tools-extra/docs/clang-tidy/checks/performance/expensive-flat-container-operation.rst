.. title:: clang-tidy - performance-expensive-flat-container-operation

performance-expensive-flat-container-operation
==============================================

Warns when calling an O(N) operation on a flat container.

This check operates on vector-based flat containers such as
``std::flat_(map|set)``, ``boost::container::flat_(map|set)``, or
``folly::sorted_vector_(map|set)``. While these containers' behavior is
identical to usual maps/sets, the insert and erase operations are O(N). This
check flags such operations, which are a common bad pattern, notably in loops.

Below is an example of a typical bad pattern: inserting some values one by one
into a flat container. This is O(N^2), as the container will need to shift
elements right after each insertion.

.. code-block:: c++

    std::random_device generator;
    std::uniform_int_distribution<int> distribution;

    std::flat_set<int> set;
    for (auto i = 0; i < N; ++i) {
        set.insert(distribution(generator));
    }

The code above can be improved using a temporary vector, later inserting all
values at once into the ``flat_set``.

.. code-block:: c++

    std::vector<int> temporary;
    for (auto i = 0; i < N; ++i) {
        temporary.push_back(distribution(generator));
    }
    std::flat_set<int> set(temporary.begin(), temporary.end());

    // Or even better when possible, moving the temporary container:
    // std::flat_set<int> set(std::move(temporary));

For expensive-to-copy objects, ``std::move_iterator`` should be used.
When possible, the temporary container can be moved directly into the flat
container. When it is known that the inserted keys are sorted and uniqued, such
as cases when they come from another flat container, ``std::sorted_unique`` can
be used when inserting to save more cycles. Finally, if order is not important,
hash-based containers can provide better performance.

Limitations
-----------

This check is not capable of flagging insertions into a map via ``operator[]``,
as it is not possible at compile-time to know whether it will trigger an
insertion or a simple lookup. These cases have to be detected using dynamic
profiling.

This check is also of course not able to detect single element operations in
loops crossing function boundaries. A more robust static analysis would be
necessary to detect these cases.

Options
-------

.. option:: WarnOutsideLoops

    When disabled, the check will only warn when the single element operation is
    directly enclosed by a loop, hence directly actionable. At the very least,
    these cases can be improved using some temporary container.

    When enabled, all insert and erase operations will be flagged.

    Default is `false`.

.. option:: FlatContainers

    A semicolon-separated list of flat containers, with ``insert``, ``emplace``
    and/or ``erase`` operations.

    Default includes ``std::flat_(map|set)``, ``flat_multi(map|set)``,
    ``boost::container::flat_(map|set)``,
    ``boost::container::flat_multi(map|set)``, and
    ``folly::sorted_vector_(map|set)``.
