.. title:: clang-tidy - cppcoreguidelines-macro-usage

cppcoreguidelines-macro-usage
=============================

Finds macro usage that is considered problematic because better language
constructs exist for the task.

The relevant sections in the C++ Core Guidelines are
`ES.31 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es31-dont-use-macros-for-constants-or-functions>`_, and
`ES.32 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es32-use-all_caps-for-all-macro-names>`_.

Examples:

.. code-block:: c++

  #define C 0
  #define F1(x, y) ((a) > (b) ? (a) : (b))
  #define F2(...) (__VA_ARGS__)
  #define COMMA ,
  #define NORETURN [[noreturn]]
  #define DEPRECATED attribute((deprecated))
  #if LIB_EXPORTS
  #define DLLEXPORTS __declspec(dllexport)
  #else
  #define DLLEXPORTS __declspec(dllimport)
  #endif

results in the following warnings::

  4 warnings generated.
  test.cpp:1:9: warning: macro 'C' used to declare a constant; consider using a 'constexpr' constant [cppcoreguidelines-macro-usage]
  #define C 0
          ^
  test.cpp:2:9: warning: function-like macro 'F1' used; consider a 'constexpr' template function [cppcoreguidelines-macro-usage]
  #define F1(x, y) ((a) > (b) ? (a) : (b))
          ^
  test.cpp:3:9: warning: variadic macro 'F2' used; consider using a 'constexpr' variadic template function [cppcoreguidelines-macro-usage]
  #define F2(...) (__VA_ARGS__)
          ^


Options
-------

.. option:: AllowedRegexp

    A regular expression to filter allowed macros. For example
    `DEBUG*|LIBTORRENT*|TORRENT*|UNI*` could be applied to filter `libtorrent`.
    Default value is `^DEBUG_*`.

.. option:: CheckCapsOnly

    Boolean flag to warn on all macros except those with CAPS_ONLY names.
    This option is intended to ease introduction of this check into older
    code bases. Default value is `false`.

.. option:: IgnoreCommandLineMacros

    Boolean flag to toggle ignoring command-line-defined macros.
    Default value is `true`.
