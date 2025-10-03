.. title:: clang-tidy - readability-use-concise-preprocessor-directives

readability-use-concise-preprocessor-directives
===============================================

Finds uses of ``#if`` that can be simplified to ``#ifdef`` or ``#ifndef`` and,
since C23 and C++23, uses of ``#elif`` that can be simplified to ``#elifdef``
or ``#elifndef``:

.. code-block:: c++

  #if defined(MEOW)
  #if !defined(MEOW)

  // becomes

  #ifdef MEOW
  #ifndef MEOW

Since C23 and C++23:

.. code-block:: c++

  #elif defined(MEOW)
  #elif !defined(MEOW)

  // becomes

  #elifdef MEOW
  #elifndef MEOW
