.. title:: clang-tidy - modernize-use-if-consteval

modernize-use-if-consteval
==========================

Replaces direct ``std::is_constant_evaluated()`` checks in ``if`` statements
with C++23's ``if consteval`` syntax.

In C++20, code often spells constant-evaluation branches as:

.. code-block:: c++

  if (std::is_constant_evaluated()) {
    return slow_but_constexpr_path();
  } else {
    return fast_runtime_path();
  }

In C++23, the language has a dedicated spelling for the same idea:

.. code-block:: c++

  if consteval {
    return slow_but_constexpr_path();
  } else {
    return fast_runtime_path();
  }

The check also recognizes the direct negated form and rewrites it to
``if !consteval``.

The check only diagnoses direct uses in ``if`` statements whose condition is a
call to ``std::is_constant_evaluated()`` or a direct negation of that call.
Lookup through ``using`` declarations, ``using namespace`` directives, and
namespace aliases is handled.

When the rewrite is source-safe, fix-its are provided. This includes inserting
braces around unbraced ``then`` and ``else`` branches because ``if consteval``
always requires compound statements.

The check still diagnoses, but does not provide fix-its for:

- ``if`` statements with an init-statement
- ``if`` statements with a condition variable
- cases where the necessary edits would cross unsafe macro or source ranges

``if constexpr`` and existing ``if consteval`` statements are ignored.
