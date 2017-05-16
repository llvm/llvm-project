.. title:: clang-tidy - modernize-use-emplace

modernize-use-emplace
=====================

The check flags insertions to an STL-style container done by calling the
``push_back`` method with an explicitly-constructed temporary of the container
element type. In this case, the corresponding ``emplace_back`` method
results in less verbose and potentially more efficient code.
Right now the check doesn't support ``push_front`` and ``insert``.
It also doesn't support ``insert`` functions for associative containers
because replacing ``insert`` with ``emplace`` may result in
`speed regression <http://htmlpreview.github.io/?https://github.com/HowardHinnant/papers/blob/master/insert_vs_emplace.html>`_, but it might get support with some addition flag in the future.

By default only ``std::vector``, ``std::deque``, ``std::list`` are considered.
This list can be modified using the :option:`ContainersWithPushBack` option.

Before:

.. code-block:: c++

    std::vector<MyClass> v;
    v.push_back(MyClass(21, 37));

    std::vector<std::pair<int, int>> w;

    w.push_back(std::pair<int, int>(21, 37));
    w.push_back(std::make_pair(21L, 37L));

After:

.. code-block:: c++

    std::vector<MyClass> v;
    v.emplace_back(21, 37);

    std::vector<std::pair<int, int>> w;
    w.emplace_back(21, 37);
    w.emplace_back(21L, 37L);

By default, the check is able to remove unnecessary ``std::make_pair`` and
``std::make_tuple`` calls from ``push_back`` calls on containers of
``std::pair`` and ``std::tuple``. Custom tuple-like types can be modified by
the :option:`TupleTypes` option; custom make functions can be modified by the
:option:`TupleMakeFunctions` option.

The other situation is when we pass arguments that will be converted to a type
inside a container.

Before:

.. code-block:: c++

    std::vector<boost::optional<std::string> > v;
    v.push_back("abc");

After:

.. code-block:: c++

    std::vector<boost::optional<std::string> > v;
    v.emplace_back("abc");


In some cases the transformation would be valid, but the code wouldn't be
exception safe. In this case the calls of ``push_back`` won't be replaced.

.. code-block:: c++

    std::vector<std::unique_ptr<int>> v;
    v.push_back(std::unique_ptr<int>(new int(0)));
    auto *ptr = new int(1);
    v.push_back(std::unique_ptr<int>(ptr));

This is because replacing it with ``emplace_back`` could cause a leak of this
pointer if ``emplace_back`` would throw exception before emplacement (e.g. not
enough memory to add a new element).

For more info read item 42 - "Consider emplacement instead of insertion." of
Scott Meyers "Effective Modern C++".

The default smart pointers that are considered are ``std::unique_ptr``,
``std::shared_ptr``, ``std::auto_ptr``. To specify other smart pointers or
other classes use the :option:`SmartPointers` option.


Check also doesn't fire if any argument of the constructor call would be:

  - a bit-field (bit-fields can't bind to rvalue/universal reference)

  - a ``new`` expression (to avoid leak)

  - if the argument would be converted via derived-to-base cast.

This check requires C++11 or higher to run.

Options
-------

.. option:: ContainersWithPushBack

   Semicolon-separated list of class names of custom containers that support
   ``push_back``.

.. option:: SmartPointers

   Semicolon-separated list of class names of custom smart pointers.

.. option:: TupleTypes

    Semicolon-separated list of ``std::tuple``-like class names.

.. option:: TupleMakeFunctions

    Semicolon-separated list of ``std::make_tuple``-like function names. Those
    function calls will be removed from ``push_back`` calls and turned into
    ``emplace_back``.

Example
^^^^^^^

.. code-block:: c++

  std::vector<MyTuple<int, bool, char>> x;
  x.push_back(MakeMyTuple(1, false, 'x'));

transforms to:

.. code-block:: c++

  std::vector<MyTuple<int, bool, char>> x;
  x.emplace_back(1, false, 'x');

when :option:`TupleTypes` is set to ``MyTuple`` and :option:`TupleMakeFunctions`
is set to ``MakeMyTuple``.
