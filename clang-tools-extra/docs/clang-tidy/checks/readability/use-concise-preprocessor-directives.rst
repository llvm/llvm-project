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

Options
-------

.. option:: PreserveConsistency

   If `true`, directives will not be simplified if doing so would break 
   consistency with other directives in a chain. Default is `false`.

Example
^^^^^^^

.. code-block:: c++

  // Not simplified.
  #if defined(FOO)
  #elif defined(BAR) || defined(BAZ)
  #endif

  // Only simplified in C23 or C++23.
  #if defined(FOO)
  #elif defined(BAR)
  #endif

  // Consistency among *different* chains is not taken into account.
  #if defined(FOO)
  	#if defined(BAR) || defined(BAZ)
  	#endif
  #elif defined(HAZ)
  #endif

  // becomes

  #ifdef FOO
  	#if defined(BAR) || defined(BAZ)
  	#endif
  #elifdef HAZ
  #endif
