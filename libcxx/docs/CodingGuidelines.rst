.. _CodingGuidelines:

========================
libc++ Coding Guidelines
========================

.. contents::
  :local:

Use ``__ugly_names`` for implementation details
===============================================

Libc++ uses ``__ugly_names`` or ``_UglyNames`` for implementation details. These names are reserved for implementations,
so users may not use them in their own applications. When using a name like ``T``, a user may have defined a macro that
changes the meaning of ``T``. By using ``__ugly_names`` we avoid that problem.

This is partially enforced by the clang-tidy check ``readability-identifier-naming`` and
``libcxx/test/libcxx/system_reserved_names.gen.py``.

Don't use argument-dependent lookup unless required by the standard
===================================================================

Unqualified function calls are susceptible to
`argument-dependent lookup (ADL) <https://en.cppreference.com/w/cpp/language/adl>`_. This means calling
``move(UserType)`` might not call ``std::move``. Therefore, function calls must use qualified names to avoid ADL. Some
functions in the standard library `require ADL usage <http://eel.is/c++draft/contents#3>`_. Names of classes, variables,
concepts, and type aliases are not subject to ADL. They don't need to be qualified.

Function overloading also applies to operators. Using ``&user_object`` may call a user-defined ``operator&``. Use
``std::addressof`` instead. Similarly, to avoid invoking a user-defined ``operator,``, make sure to cast the result to
``void`` when using the ``,`` or avoid it in the first place. For example:

.. code-block:: cpp

    for (; __first1 != __last1; ++__first1, (void)++__first2) {
      ...
    }

This is mostly enforced by the clang-tidy checks ``libcpp-robust-against-adl`` and ``libcpp-qualify-declval``.

Avoid including public headers
==============================

libc++ uses implementation-detail headers for most code. These are in a directory that starts with two underscores
(e.g. ``<__type_traits/decay.h>``). These detail headers are significantly smaller than their public counterparts.
This reduces the amount of code that is included in a single public header, which reduces compile times.

Add ``_LIBCPP_HIDE_FROM_ABI`` unless you know better
====================================================

``_LIBCPP_HIDE_FROM_ABI`` should be on every function in the library unless there is a reason not to do so. The main
reason not to add ``_LIBCPP_HIDE_FROM_ABI`` is if a function is exported from the libc++ built library. In that case the
function should be marked with ``_LIBCPP_EXPORTED_FROM_ABI``. Virtual functions should be marked with
``_LIBCPP_HIDE_FROM_ABI_VIRTUAL`` instead.

This is mostly enforced by the clang-tidy checks ``libcpp-hide-from-abi`` and ``libcpp-avoid-abi-tag-on-virtual``.

Define configuration macros to 0 or 1
=====================================

Macros should usually be defined in all configurations, instead of defining them when they're enabled and leaving them
undefined otherwise. For example, use

.. code-block:: cpp

  #if SOMETHING
  #  define _LIBCPP_SOMETHING_ENABLED 1
  #else
  #  define _LIBCPP_SOMETHING_ENABLED 0
  #endif

and then check for ``#if _LIBCPP_SOMETHING_ENABLED`` instead of

.. code-block:: cpp

  #if SOMETHING
  #  define _LIBCPP_SOMETHING_ENABLED
  #endif

and then checking for ``#ifdef _LIBCPP_SOMETHING_ENABLED``.

This makes it significantly easier to catch missing includes, since Clang and GCC will warn when using and undefined
marco inside an ``#if`` statement when using ``-Wundef``. Some macros in libc++ don't use this style yet, so this only
applies when introducing a new macro.

This is partially enforced by the clang-tidy check ``libcpp-internal-ftms``.

Use ``_LIBCPP_STD_VER``
=======================

libc++ defines the macro ``_LIBCPP_STD_VER`` for the different libc++ dialects. This should be used instead of
``__cplusplus``.

This is mostly enforced by the clang-tidy check ``libcpp-cpp-version-check``.

Use ``__ugly__`` spellings of vendor attributes
===============================================

Vendor attributes should always be ``__uglified__`` to avoid naming clashes with user-defined macros. For gnu-style
attributes this takes the form ``__attribute__((__foo__))``. C++11-style attributes look like ``[[_Clang::__foo__]]`` or
``[[__gnu__::__foo__]]`` for Clang or GCC attributes respectively. Clang and GCC also support standard attributes in
earlier language dialects than they were introduced. These should be spelled as ``[[__foo__]]``. MSVC currently doesn't
provide alternative spellings for their attributes, so these should be avoided if at all possible.

This is enforced by the clang-tidy check ``libcpp-uglify-attributes``.

Use C++11 extensions in C++03 code if they simplify the code
============================================================

libc++ only supports Clang in C++98/03 mode. Clang provides many C++11 features in C++03, making it possible to write a
lot of code in a simpler way than if we were restricted to C++03 features. Some use of extensions is even mandatory,
since libc++ supports move semantics in C++03.

Use ``using`` aliases instead of ``typedef``
============================================

``using`` aliases are generally easier to read and support templates. Some code in libc++ uses ``typedef`` for
historical reasons.

Write SFINAE with ``requires`` clauses in C++20-only code
=========================================================

``requires`` clauses can be significantly easier to read than ``enable_if`` and friends in some cases, since concepts
subsume other concepts. This means that overloads based on traits can be written without negating more general cases.
They also show intent better.

Write ``enable_if`` as ``enable_if_t<conditon, int> = 0``
=========================================================

The form ``enable_if_t<condition, int> = 0`` is the only one that works in every language mode and for overload sets
using the same template arguments otherwise. If the code must work in C++11 or C++03, the libc++-internal alias
``__enable_if_t`` can be used instead.

Prefer alias templates over class templates
===========================================

Alias templates are much more lightweight than class templates, since they don't require new instantiations for
different types. If the only member of a class is an alias, like in type traits, alias templates should be used if
possible. They do force more eager evaluation though, which can be a problem in some cases.

Apply ``[[nodiscard]]`` where relevant
======================================

Libc++ adds ``[[nodiscard]]`` whenever relevant to catch potential bugs. The standards committee has decided to _not_
have a recommended practice where to put them, so libc++ applies it whenever it makes sense to catch potential bugs.

``[[nodiscard]]`` should be applied to functions

- where discarding the return value is most likely a correctness issue. For example a locking constructor in
  ``unique_lock``.

- where discarding the return value likely points to the user wanting to do something different. For example
  ``vector::empty()``, which probably should have been ``vector::clear()``.

  This can help spotting bugs easily which otherwise may take a very long time to find.

- which return a constant. For example ``numeric_limits::min()``.
- which only observe a value. For example ``string::size()``.

  Code that discards values from these kinds of functions is dead code. It can either be removed, or the programmer
  meant to do something different.

- where discarding the value is most likely a misuse of the function. For example ``std::find(first, last, val)``.

  This protects programmers from assuming too much about how the internals of a function work, making code more robust
  in the presence of future optimizations.

Applications of ``[[nodiscard]]`` are code like any other code, so we aim to test them on public interfaces. This can be
done with a ``.verify.cpp`` test. Many examples are available. Just look for tests with the suffix
``.nodiscard.verify.cpp``.

Don't use public API names for symbols on the ABI boundary
==========================================================

Most functions in libc++ are defined in headers either as templates or as ``inline`` functions. However, we sometimes
need or want to define functions in the built library. Symbols that are declared in the headers and defined in the
built library become part of the ABI of libc++, which must be preserved for backwards compatibility. This means that
we can't easily remove or rename such symbols except in special cases.

When adding a symbol to the built library, make sure not to use a public name directly. Instead, define a
``_LIBCPP_HIDE_FROM_ABI`` function in the headers with the public name and have it call a private function in the built
library. This approach makes it easier to make changes to libc++ like move something from the built library to the
headers (which is sometimes required for ``constexpr`` support).

When defining a function at the ABI boundary, it can also be useful to consider which attributes (like ``[[gnu::pure]]``
and ``[[clang::noescape]]``) can be added to the function to improve the compiler's ability to optimize.
