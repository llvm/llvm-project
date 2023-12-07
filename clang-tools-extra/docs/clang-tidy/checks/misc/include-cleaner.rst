.. title:: clang-tidy - misc-include-cleaner

misc-include-cleaner
====================

Checks for unused and missing includes. Generates findings only for
the main file of a translation unit.
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
   insertion/removal for all headers under the directory `foo`. By default, no 
   headers will be ignored.

.. option:: DeduplicateFindings

   A boolean that controls whether the check should deduplicate findings for the
   same symbol. Defaults to `true`.
