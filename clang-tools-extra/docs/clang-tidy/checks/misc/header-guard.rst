.. title:: clang-tidy - llvm-header-guard

misc-header-guard
=================

Finds and fixes header guards that do not conform to the configured style
options from this check.

All following examples consider header file
``/path/to/include/component/header.hpp``

By default, the check ensures following header guard:

.. code-block:: c++

   #ifndef COMPONENT_HEADER_HPP
   #define COMPONENT_HEADER_HPP
   ...
   # endif

Options
-------

.. option:: HeaderDirs

  A semicolon-separated list of one or more header directory names. Header
  directories may contain `/` as path separator. The list is searched for the
  first matching string. The header guard will start from this path
  component. Default is `include`.

  E.g. :option:`HeaderDirs` is set to one of the following values:

  - `component`
  - `include/component`
  - `component;include`

  It results in the same following header guard:

  .. code-block:: c++

    #ifndef HEADER_HPP
    #define HEADER_HPP
    ...
    # endif

  .. warning::

    The :option:`HeaderDirs` list is searched until first directory name
    matches the header file path. E.g. if :option:`HeaderDirs` is set to
    `include;component`, the check will result in default behavior (since
    `include` is found first).

.. option:: Prefix

  A string specifying an optional prefix that is applied to each header guard.
  Default is an empty string.

  E.g. :option:`Prefix` is set to `MY_OWN_PREFIX_`:

  .. code-block:: c++

    #ifndef MY_OWN_PREFIX_COMPONENT_HEADER_HPP
    #define MY_OWN_PREFIX_COMPONENT_HEADER_HPP
    ...
    # endif

.. option:: EndifComment

  A boolean that controls whether the endif namespace comment is suggested.
  Default is `false`.

  E.g. :option:`EndifComment` is set to `true`:

  .. code-block:: c++

    #ifndef COMPONENT_HEADER_HPP
    #define COMPONENT_HEADER_HPP
    ...
    # endif // COMPONENT_HEADER_HPP

.. option:: AllowPragmaOnce

  A boolean that controls whether ``#pragma once`` directive is allowed.
  Default is `false`.

  E.g. with option :option:`AllowPragmaOnce` set to `true`, ``#pragma once``
  is allowed as header guard:

  .. code-block:: c++

    #pragma once
    ...
