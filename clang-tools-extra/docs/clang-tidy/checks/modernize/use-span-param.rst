.. title:: clang-tidy - modernize-use-span-param

modernize-use-span-param
========================

Finds function parameters declared as ``const std::vector<T>&``
that are only used for read-only element access, and suggests
using ``std::span<const T>`` instead.

Using ``std::span`` makes the interface more generic, allowing
callers to pass arrays, spans, or other contiguous ranges without
requiring a ``std::vector``.

For example:

.. code-block:: c++

  // Before
  void process(const std::vector<int> &v) {
    for (auto i = 0u; i < v.size(); ++i)
      use(v[i]);
  }

  // After
  void process(std::span<const int> v) {
    for (auto i = 0u; i < v.size(); ++i)
      use(v[i]);
  }

The check only triggers when all uses of the parameter are
read-only operations also available on ``std::span``:

- ``operator[]``, ``at``, ``data()``, ``size()``, ``empty()``
- ``front()``, ``back()``, ``begin()``, ``end()``
  (and their ``c``/``r``/``cr`` variants)
- Range-based ``for`` loops
- Passing to functions accepting ``const std::vector<T>&``
  or ``const T*``

The check does not trigger for:

- Virtual functions (signature cannot be changed)
- Template functions
- Functions without a body (declarations only)

This check requires C++20.
