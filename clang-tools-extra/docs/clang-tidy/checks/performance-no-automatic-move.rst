.. title:: clang-tidy - performance-no-automatic-move

performance-no-automatic-move
=============================

Finds local variables that cannot be automatically moved due to constness.

Under
`certain conditions <https://en.cppreference.com/w/cpp/language/return#automatic_move_from_local_variables_and_parameters>`_,
local values are automatically moved out when returning from a function. A
common mistake is to declare local ``lvalue`` variables ``const``, which
prevents the move.

Example `[1] <https://godbolt.org/z/x7SYYA>`_:

.. code-block:: c++

  StatusOr<std::vector<int>> Cool() {
    std::vector<int> obj = ...;
    return obj;  // calls StatusOr::StatusOr(std::vector<int>&&)
  }
  
  StatusOr<std::vector<int>> NotCool() {
    const std::vector<int> obj = ...;
    return obj;  // calls `StatusOr::StatusOr(const std::vector<int>&)`
  }

The former version (``Cool``) should be preferred over the latter (``Uncool``)
as it will avoid allocations and potentially large memory copies.

Semantics
---------

In the example above, ``StatusOr::StatusOr(T&&)`` have the same semantics as
long as the copy and move constructors for ``T`` have the same semantics. Note
that there is no guarantee that ``S::S(T&&)`` and ``S::S(const T&)`` have the
same semantics for any single ``S``, so we're not providing automated fixes for
this check, and judgement should be exerted when making the suggested changes.

-Wreturn-std-move
-----------------

Another case where the move cannot happen is the following:

.. code-block:: c++

  StatusOr<std::vector<int>> Uncool() {
    std::vector<int>&& obj = ...;
    return obj;  // calls `StatusOr::StatusOr(const std::vector<int>&)`
  }

In that case the fix is more consensual: just `return std::move(obj)`.
This is handled by the `-Wreturn-std-move` warning.
