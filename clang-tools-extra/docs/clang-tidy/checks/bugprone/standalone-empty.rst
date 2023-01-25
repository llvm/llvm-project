.. title:: clang-tidy - bugprone-standalone-empty

bugprone-standalone-empty
=========================

Warns when ``empty()`` is used on a range and the result is ignored. Suggests
``clear()`` if it is an existing member function.

The ``empty()`` method on several common ranges returns a Boolean indicating
whether or not the range is empty, but is often mistakenly interpreted as
a way to clear the contents of a range. Some ranges offer a ``clear()``
method for this purpose. This check warns when a call to empty returns a
result that is ignored, and suggests replacing it with a call to ``clear()``
if it is available as a member function of the range.

For example, the following code could be used to indicate whether a range
is empty or not, but the result is ignored:

.. code-block:: c++

  std::vector<int> v;
  ...
  v.empty();

A call to ``clear()`` would appropriately clear the contents of the range:

.. code-block:: c++

  std::vector<int> v;
  ...
  v.clear();

Limitations:
- Doesn't warn if ``empty()`` is defined and used with the ignore result in the
  class template definition (for example in the library implementation). These
  error cases can be caught with ``[[nodiscard]]`` attribute.
