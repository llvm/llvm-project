.. title:: clang-tidy - modernize-use-concise-preprocessor-directives

modernize-use-concise-preprocessor-directives
=============================================

Shortens `#if` preprocessor conditions:

.. code-block:: c++

  #if defined(MEOW)
  #if !defined(MEOW)

  // becomes

  #ifdef MEOW
  #ifndef MEOW

And, since C23 and C++23, shortens `#elif` conditions too:

.. code-block:: c++

  #elif defined(MEOW)
  #elif !defined(MEOW)

  // becomes

  #elifdef MEOW
  #elifndef MEOW
