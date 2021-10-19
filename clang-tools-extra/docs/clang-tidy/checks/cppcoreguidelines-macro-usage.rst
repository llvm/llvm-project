.. title:: clang-tidy - cppcoreguidelines-macro-usage

cppcoreguidelines-macro-usage
=============================

Finds macro usage that is considered problematic because better language
constructs exist for the task.

The relevant sections in the C++ Core Guidelines are 
`Enum.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#enum1-prefer-enumerations-over-macros>`_,
`ES.30 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es30-dont-use-macros-for-program-text-manipulation>`_,
`ES.31 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es31-dont-use-macros-for-constants-or-functions>`_ and
`ES.33 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es33-if-you-must-use-macros-give-them-unique-names>`_.

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
