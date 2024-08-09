.. title:: clang-tidy - bugprone-incorrect-iterators

bugprone-incorrect-iterators
============================

Detects calls to iterator algorithms where they are called with potentially
invalid arguments.

Different ranges
================

Looks for calls where the range for the ``begin`` argument is different to the
``end`` argument.

.. code-block:: c++

  std::find(a.begin(), b.end(), 0);
  std::find(std::begin(a), std::end(b));

Mismatched Begin/End
====================

Looks for calls where the ``begin`` parameter is passed as an ``end`` argument or
vice versa.

.. code-block:: c++

  std::find(a.begin(), a.begin(), 0); // Second argument should be a.end().
  std::find(a.end(), a.end(), 0); // First argument should be a.begin().

3 argument ranges
=================

Looks for calls which accept a range defined by [first, middle, last] and
ensures correct arguments are supplied.

.. code-block:: c++

  std::rotate(a.begin(), a.end(), pivot); // a.end() likely should be 3rd argument.
  std::rotate(a.bgein(), pivot(), b.end()) // different range past to begin/end.

Container Methods
=================

Looks for calls to methods on containers that expect an iterator inside the
container but are given a different container.

.. code-block:: c++

  vec.insert(other.begin(), 5); // The iterator is invalid for this container.
  std::vector<int> Vec{a.begin(), a.begin()}; // Second argument should be a.end().
  vec.assign(a.begin(), a.begin()) // Second argument should be a.end().

Output Iterators
================

Looks for calls which accept a single output iterator but are passed the end of
a container.

.. code-block:: c++

  std::copy(correct.begin(), correct.end(), incorrect.end());

Iterator Advancing
==================

Looks for calls that advance an iterator outside its range.

.. code-block:: c++

  auto Iter = std::next(Cont.end());

Reverse Iteration
=================

The check understands ``rbegin`` and ``rend`` and ensures they are in the
correct places.

.. code-block:: c++

  std::find(a.rbegin(), a.rend(), 0); // OK.
  std::find(a.rend(), a.rbegin(), 0); // Arguments are swapped.

Manually creating a reverse iterator using the ``std::make_reverse_iterator`` is
also supported, In this case the check looks for calls to ``end`` for the
``begin`` parameter and vice versa. The name of functions for creating reverse
iterator can be configured with the option :option:`MakeReverseIterator`.

.. code-block:: c++

  std::find(std::make_reverse_iterator(a.begin()),
            std::make_reverse_iterator(a.end()), 0); // Arguments are swapped.
  std::find(std::make_reverse_iterator(a.end()),
            std::make_reverse_iterator(a.begin()), 0); // OK.
  // Understands this spaghetti looking code is actually doing the correct thing.
  std::find(a.rbegin(), std::make_reverse_iterator(a.begin()), 0);

Options
-------

.. option:: BeginFree

  A semi-colon seperated list of free function names that return an iterator to
  the start of a range. Default value is `::std::begin;std::cbegin`.

.. option:: EndFree

  A semi-colon seperated list of free function names that return an iterator to
  the end of a range. Default value is `::std::end;std::cend`.

.. option:: BeginMethod

  A semi-colon seperated list of method names that return an iterator to
  the start of a range. Default value is `begin;cbegin`.

.. option:: EndMethod

  A semi-colon seperated list of method names that return an iterator to
  the end of a range. Default value is `end;cend`.

.. option:: RBeginFree

  A semi-colon seperated list of free function names that return a reverse 
  iterator to the start of a range. Default value is `::std::rbegin;std::crbegin`.

.. option:: REndFree

  A semi-colon seperated list of free function names that return a reverse 
  iterator to the end of a range. Default value is `::std::rend;std::crend`.

.. option:: RBeginMethod

  A semi-colon seperated list of method names that return a reverse 
  iterator to the start of a range. Default value is `rbegin;crbegin`.

.. option:: REndMethod

  A semi-colon seperated list of method names that return a reverse 
  iterator to the end of a range. Default value is `rend;crend`.

.. option:: MakeReverseIterator

  A semi-colon seperated list of free functions that convert an interator into a
  reverse iterator. Default value is `::std::make_reverse_iterator`.
