.. title:: clang-tidy - readability-avoid-unconditional-preprocessor-if

readability-avoid-unconditional-preprocessor-if
===============================================

Finds code blocks that are constantly enabled or disabled in preprocessor
directives by analyzing ``#if`` conditions, such as ``#if 0`` and ``#if 1``,
etc.

.. code-block:: c++

    #if 0
        // some disabled code
    #endif

    #if 1
       // some enabled code that can be disabled manually
    #endif

Unconditional preprocessor directives, such as ``#if 0`` for disabled code
and ``#if 1`` for enabled code, can lead to dead code and always enabled code,
respectively. Dead code can make understanding the codebase more difficult,
hinder readability, and may be a sign of unfinished functionality or abandoned
features. This can cause maintenance issues, confusion for future developers,
and potential compilation problems.

As a solution for both cases, consider using preprocessor macros or defines,
like ``#ifdef DEBUGGING_ENABLED``, to control code enabling or disabling.
This approach provides better coordination and flexibility when working with
different parts of the codebase. Alternatively, you can comment out the entire
code using ``/* */`` block comments and add a hint, such as ``@todo``,
to indicate future actions.
