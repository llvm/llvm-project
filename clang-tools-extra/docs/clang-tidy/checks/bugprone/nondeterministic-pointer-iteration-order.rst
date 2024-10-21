.. title:: clang-tidy - bugprone-nondeterministic-pointer-iteration-order

bugprone-nondeterministic-pointer-iteration-order
=================================================

Finds nondeterministic usages of pointers in unordered containers.

One canonical example is iteration across a container of pointers.

.. code-block:: c++

  {
    int a = 1, b = 2;
    std::unordered_set<int *> UnorderedPtrSet = {&a, &b};
    for (auto i : UnorderedPtrSet)
      f(i);
  }

Another such example is sorting a container of pointers.

.. code-block:: c++

  {
    int a = 1, b = 2;
    std::vector<int *> VectorOfPtr = {&a, &b};
    std::sort(VectorOfPtr.begin(), VectorOfPtr.end());
  }

Iteration of a containers of pointers may present the order of different
pointers differently across different runs of a program. In some cases this
may be acceptable behavior, in others this may be unexpected behavior. This
check is advisory for this reason.

This check only detects range-based for loops over unordered sets and maps.
Other similar usages will not be found and are false negatives.
