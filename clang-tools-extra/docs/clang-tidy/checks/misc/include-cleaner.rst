.. title:: clang-tidy - misc-include-cleaner

misc-include-cleaner
====================

Checks for unused and missing includes. Generates findings only for
the main file of a translation unit. Optionally, direct includes can be
treated as fragments of the main file for usage scanning.
Findings correspond to https://clangd.llvm.org/design/include-cleaner.

Example:

.. code-block:: c++

   // foo.h
   class Foo{};
   // bar.h
   #include "baz.h"
   class Bar{};
   // baz.h
   class Baz{};
   // main.cc
   #include "bar.h" // OK: uses class Bar from bar.h
   #include "foo.h" // warning: unused include "foo.h"
   Bar bar;
   Baz baz; // warning: missing include "baz.h"

Options
-------

.. option:: IgnoreHeaders

   A semicolon-separated list of regexes to disable insertion/removal of header
   files that match this regex as a suffix.  E.g., `foo/.*` disables
   insertion/removal for all headers under the directory `foo`. Default is an
   empty string, no headers will be ignored.

.. option:: FragmentHeaders

   A semicolon-separated list of regular expressions that match against
   normalized resolved include paths (POSIX-style separators). Direct includes
   of the main file that match are treated as fragments of the main file for
   usage scanning. This is intended for non-self-contained include fragments
   such as TableGen ``.inc``/``.def`` files or generated headers. Only direct
   includes are considered; includes inside fragments are not treated as
   fragments.

   Diagnostics remain anchored to the main file, but symbol uses inside
   fragments can keep prerequisite includes in the main file from being
   removed or marked missing. Note that include-cleaner does not support
   ``// IWYU pragma: associated``.

   Example configuration:

   .. code-block:: yaml

      CheckOptions:
        - key: misc-include-cleaner.FragmentHeaders
          value: 'gen-out/;generated/;\\.(inc|def)$'

.. option:: DeduplicateFindings

   A boolean that controls whether the check should deduplicate findings for the
   same symbol. Defaults to `true`.

.. option:: UnusedIncludes

   A boolean that controls whether the check should report unused includes
   (includes that are not used directly). Defaults to `true`.

.. option:: MissingIncludes

   A boolean that controls whether the check should report missing includes
   (header files from which symbols are used but which are not directly included).
   Defaults to `true`.
