============================
Clang Compiler User's Manual
============================

.. contents::
   :local:

Introduction
============

The Clang Compiler is an open-source compiler for the C family of
programming languages, aiming to be the best in class implementation of
these languages. Clang builds on the LLVM optimizer and code generator,
allowing it to provide high-quality optimization and code generation
support for many targets. For more general information, please see the
`Clang Web Site <http://clang.llvm.org>`_ or the `LLVM Web
Site <http://llvm.org>`_.

This document describes important notes about using Clang as a compiler
for an end-user, documenting the supported features, command line
options, etc. If you are interested in using Clang to build a tool that
processes code, please see :doc:`InternalsManual`. If you are interested in the
`Clang Static Analyzer <http://clang-analyzer.llvm.org>`_, please see its web
page.

Clang is designed to support the C family of programming languages,
which includes :ref:`C <c>`, :ref:`Objective-C <objc>`, :ref:`C++ <cxx>`, and
:ref:`Objective-C++ <objcxx>` as well as many dialects of those. For
language-specific information, please see the corresponding language
specific section:

-  :ref:`C Language <c>`: K&R C, ANSI C89, ISO C90, ISO C94 (C89+AMD1), ISO
   C99 (+TC1, TC2, TC3).
-  :ref:`Objective-C Language <objc>`: ObjC 1, ObjC 2, ObjC 2.1, plus
   variants depending on base language.
-  :ref:`C++ Language <cxx>`
-  :ref:`Objective C++ Language <objcxx>`

In addition to these base languages and their dialects, Clang supports a
broad variety of language extensions, which are documented in the
corresponding language section. These extensions are provided to be
compatible with the GCC, Microsoft, and other popular compilers as well
as to improve functionality through Clang-specific features. The Clang
driver and language features are intentionally designed to be as
compatible with the GNU GCC compiler as reasonably possible, easing
migration from GCC to Clang. In most cases, code "just works".
Clang also provides an alternative driver, :ref:`clang-cl`, that is designed
to be compatible with the Visual C++ compiler, cl.exe.

In addition to language specific features, Clang has a variety of
features that depend on what CPU architecture or operating system is
being compiled for. Please see the :ref:`Target-Specific Features and
Limitations <target_features>` section for more details.

The rest of the introduction introduces some basic :ref:`compiler
terminology <terminology>` that is used throughout this manual and
contains a basic :ref:`introduction to using Clang <basicusage>` as a
command line compiler.

.. _terminology:

Terminology
-----------

Front end, parser, backend, preprocessor, undefined behavior,
diagnostic, optimizer

.. _basicusage:

Basic Usage
-----------

Intro to how to use a C compiler for newbies.

compile + link compile then link debug info enabling optimizations
picking a language to use, defaults to C99 by default. Autosenses based
on extension. using a makefile

Command Line Options
====================

This section is generally an index into other sections. It does not go
into depth on the ones that are covered by other sections. However, the
first part introduces the language selection and other high level
options like :option:`-c`, :option:`-g`, etc.

Options to Control Error and Warning Messages
---------------------------------------------

.. option:: -Werror

  Turn warnings into errors.

.. This is in plain monospaced font because it generates the same label as
.. -Werror, and Sphinx complains.

``-Werror=foo``

  Turn warning "foo" into an error.

.. option:: -Wno-error=foo

  Turn warning "foo" into an warning even if :option:`-Werror` is specified.

.. option:: -Wfoo

  Enable warning "foo".

.. option:: -Wno-foo

  Disable warning "foo".

.. option:: -w

  Disable all diagnostics.

.. option:: -Weverything

  :ref:`Enable all diagnostics. <diagnostics_enable_everything>`

.. option:: -pedantic

  Warn on language extensions.

.. option:: -pedantic-errors

  Error on language extensions.

.. option:: -Wsystem-headers

  Enable warnings from system headers.

.. option:: -ferror-limit=123

  Stop emitting diagnostics after 123 errors have been produced. The default is
  20, and the error limit can be disabled with :option:`-ferror-limit=0`.

.. option:: -ftemplate-backtrace-limit=123

  Only emit up to 123 template instantiation notes within the template
  instantiation backtrace for a single warning or error. The default is 10, and
  the limit can be disabled with :option:`-ftemplate-backtrace-limit=0`.

.. _cl_diag_formatting:

Formatting of Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^

Clang aims to produce beautiful diagnostics by default, particularly for
new users that first come to Clang. However, different people have
different preferences, and sometimes Clang is driven by another program
that wants to parse simple and consistent output, not a person. For
these cases, Clang provides a wide range of options to control the exact
output format of the diagnostics that it generates.

.. _opt_fshow-column:

**-f[no-]show-column**
   Print column number in diagnostic.

   This option, which defaults to on, controls whether or not Clang
   prints the column number of a diagnostic. For example, when this is
   enabled, Clang will print something like:

   ::

         test.c:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^
                //

   When this is disabled, Clang will print "test.c:28: warning..." with
   no column number.

   The printed column numbers count bytes from the beginning of the
   line; take care if your source contains multibyte characters.

.. _opt_fshow-source-location:

**-f[no-]show-source-location**
   Print source file/line/column information in diagnostic.

   This option, which defaults to on, controls whether or not Clang
   prints the filename, line number and column number of a diagnostic.
   For example, when this is enabled, Clang will print something like:

   ::

         test.c:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^
                //

   When this is disabled, Clang will not print the "test.c:28:8: "
   part.

.. _opt_fcaret-diagnostics:

**-f[no-]caret-diagnostics**
   Print source line and ranges from source code in diagnostic.
   This option, which defaults to on, controls whether or not Clang
   prints the source line, source ranges, and caret when emitting a
   diagnostic. For example, when this is enabled, Clang will print
   something like:

   ::

         test.c:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^
                //

**-f[no-]color-diagnostics**
   This option, which defaults to on when a color-capable terminal is
   detected, controls whether or not Clang prints diagnostics in color.

   When this option is enabled, Clang will use colors to highlight
   specific parts of the diagnostic, e.g.,

   .. nasty hack to not lose our dignity

   .. raw:: html

       <pre>
         <b><span style="color:black">test.c:28:8: <span style="color:magenta">warning</span>: extra tokens at end of #endif directive [-Wextra-tokens]</span></b>
         #endif bad
                <span style="color:green">^</span>
                <span style="color:green">//</span>
       </pre>

   When this is disabled, Clang will just print:

   ::

         test.c:2:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^
                //

**-fansi-escape-codes**
   Controls whether ANSI escape codes are used instead of the Windows Console
   API to output colored diagnostics. This option is only used on Windows and
   defaults to off.

.. option:: -fdiagnostics-format=clang/msvc/vi

   Changes diagnostic output format to better match IDEs and command line tools.

   This option controls the output format of the filename, line number,
   and column printed in diagnostic messages. The options, and their
   affect on formatting a simple conversion diagnostic, follow:

   **clang** (default)
       ::

           t.c:3:11: warning: conversion specifies type 'char *' but the argument has type 'int'

   **msvc**
       ::

           t.c(3,11) : warning: conversion specifies type 'char *' but the argument has type 'int'

   **vi**
       ::

           t.c +3:11: warning: conversion specifies type 'char *' but the argument has type 'int'

.. _opt_fdiagnostics-show-option:

**-f[no-]diagnostics-show-option**
   Enable ``[-Woption]`` information in diagnostic line.

   This option, which defaults to on, controls whether or not Clang
   prints the associated :ref:`warning group <cl_diag_warning_groups>`
   option name when outputting a warning diagnostic. For example, in
   this output:

   ::

         test.c:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^
                //

   Passing **-fno-diagnostics-show-option** will prevent Clang from
   printing the [:ref:`-Wextra-tokens <opt_Wextra-tokens>`] information in
   the diagnostic. This information tells you the flag needed to enable
   or disable the diagnostic, either from the command line or through
   :ref:`#pragma GCC diagnostic <pragma_GCC_diagnostic>`.

.. _opt_fdiagnostics-show-category:

.. option:: -fdiagnostics-show-category=none/id/name

   Enable printing category information in diagnostic line.

   This option, which defaults to "none", controls whether or not Clang
   prints the category associated with a diagnostic when emitting it.
   Each diagnostic may or many not have an associated category, if it
   has one, it is listed in the diagnostic categorization field of the
   diagnostic line (in the []'s).

   For example, a format string warning will produce these three
   renditions based on the setting of this option:

   ::

         t.c:3:11: warning: conversion specifies type 'char *' but the argument has type 'int' [-Wformat]
         t.c:3:11: warning: conversion specifies type 'char *' but the argument has type 'int' [-Wformat,1]
         t.c:3:11: warning: conversion specifies type 'char *' but the argument has type 'int' [-Wformat,Format String]

   This category can be used by clients that want to group diagnostics
   by category, so it should be a high level category. We want dozens
   of these, not hundreds or thousands of them.

.. _opt_fdiagnostics-fixit-info:

**-f[no-]diagnostics-fixit-info**
   Enable "FixIt" information in the diagnostics output.

   This option, which defaults to on, controls whether or not Clang
   prints the information on how to fix a specific diagnostic
   underneath it when it knows. For example, in this output:

   ::

         test.c:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^
                //

   Passing **-fno-diagnostics-fixit-info** will prevent Clang from
   printing the "//" line at the end of the message. This information
   is useful for users who may not understand what is wrong, but can be
   confusing for machine parsing.

.. _opt_fdiagnostics-print-source-range-info:

**-fdiagnostics-print-source-range-info**
   Print machine parsable information about source ranges.
   This option makes Clang print information about source ranges in a machine
   parsable format after the file/line/column number information. The
   information is a simple sequence of brace enclosed ranges, where each range
   lists the start and end line/column locations. For example, in this output:

   ::

       exprs.c:47:15:{47:8-47:14}{47:17-47:24}: error: invalid operands to binary expression ('int *' and '_Complex float')
          P = (P-42) + Gamma*4;
              ~~~~~~ ^ ~~~~~~~

   The {}'s are generated by -fdiagnostics-print-source-range-info.

   The printed column numbers count bytes from the beginning of the
   line; take care if your source contains multibyte characters.

.. option:: -fdiagnostics-parseable-fixits

   Print Fix-Its in a machine parseable form.

   This option makes Clang print available Fix-Its in a machine
   parseable format at the end of diagnostics. The following example
   illustrates the format:

   ::

        fix-it:"t.cpp":{7:25-7:29}:"Gamma"

   The range printed is a half-open range, so in this example the
   characters at column 25 up to but not including column 29 on line 7
   in t.cpp should be replaced with the string "Gamma". Either the
   range or the replacement string may be empty (representing strict
   insertions and strict erasures, respectively). Both the file name
   and the insertion string escape backslash (as "\\\\"), tabs (as
   "\\t"), newlines (as "\\n"), double quotes(as "\\"") and
   non-printable characters (as octal "\\xxx").

   The printed column numbers count bytes from the beginning of the
   line; take care if your source contains multibyte characters.

.. option:: -fno-elide-type

   Turns off elision in template type printing.

   The default for template type printing is to elide as many template
   arguments as possible, removing those which are the same in both
   template types, leaving only the differences. Adding this flag will
   print all the template arguments. If supported by the terminal,
   highlighting will still appear on differing arguments.

   Default:

   ::

       t.cc:4:5: note: candidate function not viable: no known conversion from 'vector<map<[...], map<float, [...]>>>' to 'vector<map<[...], map<double, [...]>>>' for 1st argument;

   -fno-elide-type:

   ::

       t.cc:4:5: note: candidate function not viable: no known conversion from 'vector<map<int, map<float, int>>>' to 'vector<map<int, map<double, int>>>' for 1st argument;

.. option:: -fdiagnostics-show-template-tree

   Template type diffing prints a text tree.

   For diffing large templated types, this option will cause Clang to
   display the templates as an indented text tree, one argument per
   line, with differences marked inline. This is compatible with
   -fno-elide-type.

   Default:

   ::

       t.cc:4:5: note: candidate function not viable: no known conversion from 'vector<map<[...], map<float, [...]>>>' to 'vector<map<[...], map<double, [...]>>>' for 1st argument;

   With :option:`-fdiagnostics-show-template-tree`:

   ::

       t.cc:4:5: note: candidate function not viable: no known conversion for 1st argument;
         vector<
           map<
             [...],
             map<
               [float != double],
               [...]>>>

.. _cl_diag_warning_groups:

Individual Warning Groups
^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: Generate this from tblgen. Define one anchor per warning group.

.. _opt_wextra-tokens:

.. option:: -Wextra-tokens

   Warn about excess tokens at the end of a preprocessor directive.

   This option, which defaults to on, enables warnings about extra
   tokens at the end of preprocessor directives. For example:

   ::

         test.c:28:8: warning: extra tokens at end of #endif directive [-Wextra-tokens]
         #endif bad
                ^

   These extra tokens are not strictly conforming, and are usually best
   handled by commenting them out.

.. option:: -Wambiguous-member-template

   Warn about unqualified uses of a member template whose name resolves to
   another template at the location of the use.

   This option, which defaults to on, enables a warning in the
   following code:

   ::

       template<typename T> struct set{};
       template<typename T> struct trait { typedef const T& type; };
       struct Value {
         template<typename T> void set(typename trait<T>::type value) {}
       };
       void foo() {
         Value v;
         v.set<double>(3.2);
       }

   C++ [basic.lookup.classref] requires this to be an error, but,
   because it's hard to work around, Clang downgrades it to a warning
   as an extension.

.. option:: -Wbind-to-temporary-copy

   Warn about an unusable copy constructor when binding a reference to a
   temporary.

   This option, which defaults to on, enables warnings about binding a
   reference to a temporary when the temporary doesn't have a usable
   copy constructor. For example:

   ::

         struct NonCopyable {
           NonCopyable();
         private:
           NonCopyable(const NonCopyable&);
         };
         void foo(const NonCopyable&);
         void bar() {
           foo(NonCopyable());  // Disallowed in C++98; allowed in C++11.
         }

   ::

         struct NonCopyable2 {
           NonCopyable2();
           NonCopyable2(NonCopyable2&);
         };
         void foo(const NonCopyable2&);
         void bar() {
           foo(NonCopyable2());  // Disallowed in C++98; allowed in C++11.
         }

   Note that if ``NonCopyable2::NonCopyable2()`` has a default argument
   whose instantiation produces a compile error, that error will still
   be a hard error in C++98 mode even if this warning is turned off.

Options to Control Clang Crash Diagnostics
------------------------------------------

As unbelievable as it may sound, Clang does crash from time to time.
Generally, this only occurs to those living on the `bleeding
edge <http://llvm.org/releases/download.html#svn>`_. Clang goes to great
lengths to assist you in filing a bug report. Specifically, Clang
generates preprocessed source file(s) and associated run script(s) upon
a crash. These files should be attached to a bug report to ease
reproducibility of the failure. Below are the command line options to
control the crash diagnostics.

.. option:: -fno-crash-diagnostics

  Disable auto-generation of preprocessed source files during a clang crash.

The -fno-crash-diagnostics flag can be helpful for speeding the process
of generating a delta reduced test case.

.. _opt_rpass:

Options to Emit Optimization Reports
------------------------------------

Optimization reports trace, at a high-level, all the major decisions
done by compiler transformations. For instance, when the inliner
decides to inline function ``foo()`` into ``bar()``, or the loop unroller
decides to unroll a loop N times, or the vectorizer decides to
vectorize a loop body.

Clang offers a family of flags which the optimizers can use to emit
a diagnostic in three cases:

1. When the pass makes a transformation (:option:`-Rpass`).

2. When the pass fails to make a transformation (:option:`-Rpass-missed`).

3. When the pass determines whether or not to make a transformation
   (:option:`-Rpass-analysis`).

NOTE: Although the discussion below focuses on :option:`-Rpass`, the exact
same options apply to :option:`-Rpass-missed` and :option:`-Rpass-analysis`.

Since there are dozens of passes inside the compiler, each of these flags
take a regular expression that identifies the name of the pass which should
emit the associated diagnostic. For example, to get a report from the inliner,
compile the code with:

.. code-block:: console

   $ clang -O2 -Rpass=inline code.cc -o code
   code.cc:4:25: remark: foo inlined into bar [-Rpass=inline]
   int bar(int j) { return foo(j, j - 2); }
                           ^

Note that remarks from the inliner are identified with `[-Rpass=inline]`.
To request a report from every optimization pass, you should use
:option:`-Rpass=.*` (in fact, you can use any valid POSIX regular
expression). However, do not expect a report from every transformation
made by the compiler. Optimization remarks do not really make sense
outside of the major transformations (e.g., inlining, vectorization,
loop optimizations) and not every optimization pass supports this
feature.

Current limitations
^^^^^^^^^^^^^^^^^^^

1. Optimization remarks that refer to function names will display the
   mangled name of the function. Since these remarks are emitted by the
   back end of the compiler, it does not know anything about the input
   language, nor its mangling rules.

2. Some source locations are not displayed correctly. The front end has
   a more detailed source location tracking than the locations included
   in the debug info (e.g., the front end can locate code inside macro
   expansions). However, the locations used by :option:`-Rpass` are
   translated from debug annotations. That translation can be lossy,
   which results in some remarks having no location information.


Language and Target-Independent Features
========================================

Controlling Errors and Warnings
-------------------------------

Clang provides a number of ways to control which code constructs cause
it to emit errors and warning messages, and how they are displayed to
the console.

Controlling How Clang Displays Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When Clang emits a diagnostic, it includes rich information in the
output, and gives you fine-grain control over which information is
printed. Clang has the ability to print this information, and these are
the options that control it:

#. A file/line/column indicator that shows exactly where the diagnostic
   occurs in your code [:ref:`-fshow-column <opt_fshow-column>`,
   :ref:`-fshow-source-location <opt_fshow-source-location>`].
#. A categorization of the diagnostic as a note, warning, error, or
   fatal error.
#. A text string that describes what the problem is.
#. An option that indicates how to control the diagnostic (for
   diagnostics that support it)
   [:ref:`-fdiagnostics-show-option <opt_fdiagnostics-show-option>`].
#. A :ref:`high-level category <diagnostics_categories>` for the diagnostic
   for clients that want to group diagnostics by class (for diagnostics
   that support it)
   [:ref:`-fdiagnostics-show-category <opt_fdiagnostics-show-category>`].
#. The line of source code that the issue occurs on, along with a caret
   and ranges that indicate the important locations
   [:ref:`-fcaret-diagnostics <opt_fcaret-diagnostics>`].
#. "FixIt" information, which is a concise explanation of how to fix the
   problem (when Clang is certain it knows)
   [:ref:`-fdiagnostics-fixit-info <opt_fdiagnostics-fixit-info>`].
#. A machine-parsable representation of the ranges involved (off by
   default)
   [:ref:`-fdiagnostics-print-source-range-info <opt_fdiagnostics-print-source-range-info>`].

For more information please see :ref:`Formatting of
Diagnostics <cl_diag_formatting>`.

Diagnostic Mappings
^^^^^^^^^^^^^^^^^^^

All diagnostics are mapped into one of these 5 classes:

-  Ignored
-  Note
-  Remark
-  Warning
-  Error
-  Fatal

.. _diagnostics_categories:

Diagnostic Categories
^^^^^^^^^^^^^^^^^^^^^

Though not shown by default, diagnostics may each be associated with a
high-level category. This category is intended to make it possible to
triage builds that produce a large number of errors or warnings in a
grouped way.

Categories are not shown by default, but they can be turned on with the
:ref:`-fdiagnostics-show-category <opt_fdiagnostics-show-category>` option.
When set to "``name``", the category is printed textually in the
diagnostic output. When it is set to "``id``", a category number is
printed. The mapping of category names to category id's can be obtained
by running '``clang   --print-diagnostic-categories``'.

Controlling Diagnostics via Command Line Flags
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: -W flags, -pedantic, etc

.. _pragma_gcc_diagnostic:

Controlling Diagnostics via Pragmas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang can also control what diagnostics are enabled through the use of
pragmas in the source code. This is useful for turning off specific
warnings in a section of source code. Clang supports GCC's pragma for
compatibility with existing source code, as well as several extensions.

The pragma may control any warning that can be used from the command
line. Warnings may be set to ignored, warning, error, or fatal. The
following example code will tell Clang or GCC to ignore the -Wall
warnings:

.. code-block:: c

  #pragma GCC diagnostic ignored "-Wall"

In addition to all of the functionality provided by GCC's pragma, Clang
also allows you to push and pop the current warning state. This is
particularly useful when writing a header file that will be compiled by
other people, because you don't know what warning flags they build with.

In the below example :option:`-Wmultichar` is ignored for only a single line of
code, after which the diagnostics return to whatever state had previously
existed.

.. code-block:: c

  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wmultichar"

  char b = 'df'; // no warning.

  #pragma clang diagnostic pop

The push and pop pragmas will save and restore the full diagnostic state
of the compiler, regardless of how it was set. That means that it is
possible to use push and pop around GCC compatible diagnostics and Clang
will push and pop them appropriately, while GCC will ignore the pushes
and pops as unknown pragmas. It should be noted that while Clang
supports the GCC pragma, Clang and GCC do not support the exact same set
of warnings, so even when using GCC compatible #pragmas there is no
guarantee that they will have identical behaviour on both compilers.

In addition to controlling warnings and errors generated by the compiler, it is
possible to generate custom warning and error messages through the following
pragmas:

.. code-block:: c

  // The following will produce warning messages
  #pragma message "some diagnostic message"
  #pragma GCC warning "TODO: replace deprecated feature"

  // The following will produce an error message
  #pragma GCC error "Not supported"

These pragmas operate similarly to the ``#warning`` and ``#error`` preprocessor
directives, except that they may also be embedded into preprocessor macros via
the C99 ``_Pragma`` operator, for example:

.. code-block:: c

  #define STR(X) #X
  #define DEFER(M,...) M(__VA_ARGS__)
  #define CUSTOM_ERROR(X) _Pragma(STR(GCC error(X " at line " DEFER(STR,__LINE__))))

  CUSTOM_ERROR("Feature not available");

Controlling Diagnostics in System Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Warnings are suppressed when they occur in system headers. By default,
an included file is treated as a system header if it is found in an
include path specified by ``-isystem``, but this can be overridden in
several ways.

The ``system_header`` pragma can be used to mark the current file as
being a system header. No warnings will be produced from the location of
the pragma onwards within the same file.

.. code-block:: c

  char a = 'xy'; // warning

  #pragma clang system_header

  char b = 'ab'; // no warning

The :option:`--system-header-prefix=` and :option:`--no-system-header-prefix=`
command-line arguments can be used to override whether subsets of an include
path are treated as system headers. When the name in a ``#include`` directive
is found within a header search path and starts with a system prefix, the
header is treated as a system header. The last prefix on the
command-line which matches the specified header name takes precedence.
For instance:

.. code-block:: console

  $ clang -Ifoo -isystem bar --system-header-prefix=x/ \
      --no-system-header-prefix=x/y/

Here, ``#include "x/a.h"`` is treated as including a system header, even
if the header is found in ``foo``, and ``#include "x/y/b.h"`` is treated
as not including a system header, even if the header is found in
``bar``.

A ``#include`` directive which finds a file relative to the current
directory is treated as including a system header if the including file
is treated as a system header.

.. _diagnostics_enable_everything:

Enabling All Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the traditional ``-W`` flags, one can enable **all**
diagnostics by passing :option:`-Weverything`. This works as expected
with
:option:`-Werror`, and also includes the warnings from :option:`-pedantic`.

Note that when combined with :option:`-w` (which disables all warnings), that
flag wins.

Controlling Static Analyzer Diagnostics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While not strictly part of the compiler, the diagnostics from Clang's
`static analyzer <http://clang-analyzer.llvm.org>`_ can also be
influenced by the user via changes to the source code. See the available
`annotations <http://clang-analyzer.llvm.org/annotations.html>`_ and the
analyzer's `FAQ
page <http://clang-analyzer.llvm.org/faq.html#exclude_code>`_ for more
information.

.. _usersmanual-precompiled-headers:

Precompiled Headers
-------------------

`Precompiled headers <http://en.wikipedia.org/wiki/Precompiled_header>`__
are a general approach employed by many compilers to reduce compilation
time. The underlying motivation of the approach is that it is common for
the same (and often large) header files to be included by multiple
source files. Consequently, compile times can often be greatly improved
by caching some of the (redundant) work done by a compiler to process
headers. Precompiled header files, which represent one of many ways to
implement this optimization, are literally files that represent an
on-disk cache that contains the vital information necessary to reduce
some of the work needed to process a corresponding header file. While
details of precompiled headers vary between compilers, precompiled
headers have been shown to be highly effective at speeding up program
compilation on systems with very large system headers (e.g., Mac OS X).

Generating a PCH File
^^^^^^^^^^^^^^^^^^^^^

To generate a PCH file using Clang, one invokes Clang with the
:option:`-x <language>-header` option. This mirrors the interface in GCC
for generating PCH files:

.. code-block:: console

  $ gcc -x c-header test.h -o test.h.gch
  $ clang -x c-header test.h -o test.h.pch

Using a PCH File
^^^^^^^^^^^^^^^^

A PCH file can then be used as a prefix header when a :option:`-include`
option is passed to ``clang``:

.. code-block:: console

  $ clang -include test.h test.c -o test

The ``clang`` driver will first check if a PCH file for ``test.h`` is
available; if so, the contents of ``test.h`` (and the files it includes)
will be processed from the PCH file. Otherwise, Clang falls back to
directly processing the content of ``test.h``. This mirrors the behavior
of GCC.

.. note::

  Clang does *not* automatically use PCH files for headers that are directly
  included within a source file. For example:

  .. code-block:: console

    $ clang -x c-header test.h -o test.h.pch
    $ cat test.c
    #include "test.h"
    $ clang test.c -o test

  In this example, ``clang`` will not automatically use the PCH file for
  ``test.h`` since ``test.h`` was included directly in the source file and not
  specified on the command line using :option:`-include`.

Relocatable PCH Files
^^^^^^^^^^^^^^^^^^^^^

It is sometimes necessary to build a precompiled header from headers
that are not yet in their final, installed locations. For example, one
might build a precompiled header within the build tree that is then
meant to be installed alongside the headers. Clang permits the creation
of "relocatable" precompiled headers, which are built with a given path
(into the build directory) and can later be used from an installed
location.

To build a relocatable precompiled header, place your headers into a
subdirectory whose structure mimics the installed location. For example,
if you want to build a precompiled header for the header ``mylib.h``
that will be installed into ``/usr/include``, create a subdirectory
``build/usr/include`` and place the header ``mylib.h`` into that
subdirectory. If ``mylib.h`` depends on other headers, then they can be
stored within ``build/usr/include`` in a way that mimics the installed
location.

Building a relocatable precompiled header requires two additional
arguments. First, pass the ``--relocatable-pch`` flag to indicate that
the resulting PCH file should be relocatable. Second, pass
:option:`-isysroot /path/to/build`, which makes all includes for your library
relative to the build directory. For example:

.. code-block:: console

  # clang -x c-header --relocatable-pch -isysroot /path/to/build /path/to/build/mylib.h mylib.h.pch

When loading the relocatable PCH file, the various headers used in the
PCH file are found from the system header root. For example, ``mylib.h``
can be found in ``/usr/include/mylib.h``. If the headers are installed
in some other system root, the :option:`-isysroot` option can be used provide
a different system root from which the headers will be based. For
example, :option:`-isysroot /Developer/SDKs/MacOSX10.4u.sdk` will look for
``mylib.h`` in ``/Developer/SDKs/MacOSX10.4u.sdk/usr/include/mylib.h``.

Relocatable precompiled headers are intended to be used in a limited
number of cases where the compilation environment is tightly controlled
and the precompiled header cannot be generated after headers have been
installed.

Controlling Code Generation
---------------------------

Clang provides a number of ways to control code generation. The options
are listed below.

**-f[no-]sanitize=check1,check2,...**
   Turn on runtime checks for various forms of undefined or suspicious
   behavior.

   This option controls whether Clang adds runtime checks for various
   forms of undefined or suspicious behavior, and is disabled by
   default. If a check fails, a diagnostic message is produced at
   runtime explaining the problem. The main checks are:

   -  .. _opt_fsanitize_address:

      ``-fsanitize=address``:
      :doc:`AddressSanitizer`, a memory error
      detector.
   -  ``-fsanitize=integer``: Enables checks for undefined or
      suspicious integer behavior.
   -  .. _opt_fsanitize_thread:

      ``-fsanitize=thread``: :doc:`ThreadSanitizer`, a data race detector.
   -  .. _opt_fsanitize_memory:

      ``-fsanitize=memory``: :doc:`MemorySanitizer`,
      an *experimental* detector of uninitialized reads. Not ready for
      widespread use.
   -  .. _opt_fsanitize_undefined:

      ``-fsanitize=undefined``: Fast and compatible undefined behavior
      checker. Enables the undefined behavior checks that have small
      runtime cost and no impact on address space layout or ABI. This
      includes all of the checks listed below other than
      ``unsigned-integer-overflow``.

   -  ``-fsanitize=undefined-trap``: This includes all sanitizers
      included by ``-fsanitize=undefined``, except those that require
      runtime support. This group of sanitizers is intended to be
      used in conjunction with the ``-fsanitize-undefined-trap-on-error``
      flag. This includes all of the checks listed below other than
      ``unsigned-integer-overflow`` and ``vptr``.
   -  ``-fsanitize=dataflow``: :doc:`DataFlowSanitizer`, a general data
      flow analysis.

   The following more fine-grained checks are also available:

   -  ``-fsanitize=alignment``: Use of a misaligned pointer or creation
      of a misaligned reference.
   -  ``-fsanitize=bool``: Load of a ``bool`` value which is neither
      ``true`` nor ``false``.
   -  ``-fsanitize=bounds``: Out of bounds array indexing, in cases
      where the array bound can be statically determined.
   -  ``-fsanitize=enum``: Load of a value of an enumerated type which
      is not in the range of representable values for that enumerated
      type.
   -  ``-fsanitize=float-cast-overflow``: Conversion to, from, or
      between floating-point types which would overflow the
      destination.
   -  ``-fsanitize=float-divide-by-zero``: Floating point division by
      zero.
   -  ``-fsanitize=function``: Indirect call of a function through a
      function pointer of the wrong type (Linux, C++ and x86/x86_64 only).
   -  ``-fsanitize=integer-divide-by-zero``: Integer division by zero.
   -  ``-fsanitize=null``: Use of a null pointer or creation of a null
      reference.
   -  ``-fsanitize=object-size``: An attempt to use bytes which the
      optimizer can determine are not part of the object being
      accessed. The sizes of objects are determined using
      ``__builtin_object_size``, and consequently may be able to detect
      more problems at higher optimization levels.
   -  ``-fsanitize=return``: In C++, reaching the end of a
      value-returning function without returning a value.
   -  ``-fsanitize=shift``: Shift operators where the amount shifted is
      greater or equal to the promoted bit-width of the left hand side
      or less than zero, or where the left hand side is negative. For a
      signed left shift, also checks for signed overflow in C, and for
      unsigned overflow in C++.
   -  ``-fsanitize=signed-integer-overflow``: Signed integer overflow,
      including all the checks added by ``-ftrapv``, and checking for
      overflow in signed division (``INT_MIN / -1``).
   -  ``-fsanitize=unreachable``: If control flow reaches
      ``__builtin_unreachable``.
   -  ``-fsanitize=unsigned-integer-overflow``: Unsigned integer
      overflows.
   -  ``-fsanitize=vla-bound``: A variable-length array whose bound
      does not evaluate to a positive value.
   -  ``-fsanitize=vptr``: Use of an object whose vptr indicates that
      it is of the wrong dynamic type, or that its lifetime has not
      begun or has ended. Incompatible with ``-fno-rtti``.

   You can turn off or modify checks for certain source files, functions
   or even variables by providing a special file:

   -  ``-fsanitize-blacklist=/path/to/blacklist/file``: disable or modify
      sanitizer checks for objects listed in the file. See
      :doc:`SanitizerSpecialCaseList` for file format description.
   -  ``-fno-sanitize-blacklist``: don't use blacklist file, if it was
      specified earlier in the command line.

   Extra features of MemorySanitizer (require explicit
   ``-fsanitize=memory``):

   -  ``-fsanitize-memory-track-origins[=level]``: Enables origin tracking in
      MemorySanitizer. Adds a second section to MemorySanitizer
      reports pointing to the heap or stack allocation the
      uninitialized bits came from. Slows down execution by additional
      1.5x-2x.

      Possible values for level are 0 (off), 1 (default), 2. Level 2 adds more
      sections to MemorySanitizer reports describing the order of memory stores
      the uninitialized value went through. Beware, this mode may use a lot of
      extra memory.

   Extra features of UndefinedBehaviorSanitizer:

   -  ``-fno-sanitize-recover``: By default, after a sanitizer diagnoses
      an issue, it will attempt to continue executing the program if there
      is a reasonable behavior it can give to the faulting operation. This
      option causes the program to abort instead.
   -  ``-fsanitize-undefined-trap-on-error``: Causes traps to be emitted
      rather than calls to runtime libraries when a problem is detected.
      This option is intended for use in cases where the sanitizer runtime
      cannot be used (for instance, when building libc or a kernel module).
      This is only compatible with the sanitizers in the ``undefined-trap``
      group.

   The ``-fsanitize=`` argument must also be provided when linking, in
   order to link to the appropriate runtime library. When using
   ``-fsanitize=vptr`` (or a group that includes it, such as
   ``-fsanitize=undefined``) with a C++ program, the link must be
   performed by ``clang++``, not ``clang``, in order to link against the
   C++-specific parts of the runtime library.

   It is not possible to combine more than one of the ``-fsanitize=address``,
   ``-fsanitize=thread``, and ``-fsanitize=memory`` checkers in the same
   program. The ``-fsanitize=undefined`` checks can be combined with other
   sanitizers.

.. option:: -fno-assume-sane-operator-new

   Don't assume that the C++'s new operator is sane.

   This option tells the compiler to do not assume that C++'s global
   new operator will always return a pointer that does not alias any
   other pointer when the function returns.

.. option:: -ftrap-function=[name]

   Instruct code generator to emit a function call to the specified
   function name for ``__builtin_trap()``.

   LLVM code generator translates ``__builtin_trap()`` to a trap
   instruction if it is supported by the target ISA. Otherwise, the
   builtin is translated into a call to ``abort``. If this option is
   set, then the code generator will always lower the builtin to a call
   to the specified function regardless of whether the target ISA has a
   trap instruction. This option is useful for environments (e.g.
   deeply embedded) where a trap cannot be properly handled, or when
   some custom behavior is desired.

.. option:: -ftls-model=[model]

   Select which TLS model to use.

   Valid values are: ``global-dynamic``, ``local-dynamic``,
   ``initial-exec`` and ``local-exec``. The default value is
   ``global-dynamic``. The compiler may use a different model if the
   selected model is not supported by the target, or if a more
   efficient model can be used. The TLS model can be overridden per
   variable using the ``tls_model`` attribute.

.. option:: -mhwdiv=[values]

   Select the ARM modes (arm or thumb) that support hardware division
   instructions.

   Valid values are: ``arm``, ``thumb`` and ``arm,thumb``.
   This option is used to indicate which mode (arm or thumb) supports
   hardware division instructions. This only applies to the ARM
   architecture.

.. option:: -m[no-]crc

   Enable or disable CRC instructions.

   This option is used to indicate whether CRC instructions are to
   be generated. This only applies to the ARM architecture.

   CRC instructions are enabled by default on ARMv8.

.. option:: -mgeneral-regs-only

   Generate code which only uses the general purpose registers.

   This option restricts the generated code to use general registers
   only. This only applies to the AArch64 architecture.


Profile Guided Optimization
---------------------------

Profile information enables better optimization. For example, knowing that a
branch is taken very frequently helps the compiler make better decisions when
ordering basic blocks. Knowing that a function ``foo`` is called more
frequently than another function ``bar`` helps the inliner.

Clang supports profile guided optimization with two different kinds of
profiling. A sampling profiler can generate a profile with very low runtime
overhead, or you can build an instrumented version of the code that collects
more detailed profile information. Both kinds of profiles can provide execution
counts for instructions in the code and information on branches taken and
function invocation.

Regardless of which kind of profiling you use, be careful to collect profiles
by running your code with inputs that are representative of the typical
behavior. Code that is not exercised in the profile will be optimized as if it
is unimportant, and the compiler may make poor optimization choices for code
that is disproportionately used while profiling.

Using Sampling Profilers
^^^^^^^^^^^^^^^^^^^^^^^^

Sampling profilers are used to collect runtime information, such as
hardware counters, while your application executes. They are typically
very efficient and do not incur a large runtime overhead. The
sample data collected by the profiler can be used during compilation
to determine what the most executed areas of the code are.

Using the data from a sample profiler requires some changes in the way
a program is built. Before the compiler can use profiling information,
the code needs to execute under the profiler. The following is the
usual build cycle when using sample profilers for optimization:

1. Build the code with source line table information. You can use all the
   usual build flags that you always build your application with. The only
   requirement is that you add ``-gline-tables-only`` or ``-g`` to the
   command line. This is important for the profiler to be able to map
   instructions back to source line locations.

   .. code-block:: console

     $ clang++ -O2 -gline-tables-only code.cc -o code

2. Run the executable under a sampling profiler. The specific profiler
   you use does not really matter, as long as its output can be converted
   into the format that the LLVM optimizer understands. Currently, there
   exists a conversion tool for the Linux Perf profiler
   (https://perf.wiki.kernel.org/), so these examples assume that you
   are using Linux Perf to profile your code.

   .. code-block:: console

     $ perf record -b ./code

   Note the use of the ``-b`` flag. This tells Perf to use the Last Branch
   Record (LBR) to record call chains. While this is not strictly required,
   it provides better call information, which improves the accuracy of
   the profile data.

3. Convert the collected profile data to LLVM's sample profile format.
   This is currently supported via the AutoFDO converter ``create_llvm_prof``.
   It is available at http://github.com/google/autofdo. Once built and
   installed, you can convert the ``perf.data`` file to LLVM using
   the command:

   .. code-block:: console

     $ create_llvm_prof --binary=./code --out=code.prof

   This will read ``perf.data`` and the binary file ``./code`` and emit
   the profile data in ``code.prof``. Note that if you ran ``perf``
   without the ``-b`` flag, you need to use ``--use_lbr=false`` when
   calling ``create_llvm_prof``.

4. Build the code again using the collected profile. This step feeds
   the profile back to the optimizers. This should result in a binary
   that executes faster than the original one. Note that you are not
   required to build the code with the exact same arguments that you
   used in the first step. The only requirement is that you build the code
   with ``-gline-tables-only`` and ``-fprofile-sample-use``.

   .. code-block:: console

     $ clang++ -O2 -gline-tables-only -fprofile-sample-use=code.prof code.cc -o code


Sample Profile Format
"""""""""""""""""""""

If you are not using Linux Perf to collect profiles, you will need to
write a conversion tool from your profiler to LLVM's format. This section
explains the file format expected by the backend.

Sample profiles are written as ASCII text. The file is divided into sections,
which correspond to each of the functions executed at runtime. Each
section has the following format (taken from
https://github.com/google/autofdo/blob/master/profile_writer.h):

.. code-block:: console

    function1:total_samples:total_head_samples
    offset1[.discriminator]: number_of_samples [fn1:num fn2:num ... ]
    offset2[.discriminator]: number_of_samples [fn3:num fn4:num ... ]
    ...
    offsetN[.discriminator]: number_of_samples [fn5:num fn6:num ... ]

The file may contain blank lines between sections and within a
section. However, the spacing within a single line is fixed. Additional
spaces will result in an error while reading the file.

Function names must be mangled in order for the profile loader to
match them in the current translation unit. The two numbers in the
function header specify how many total samples were accumulated in the
function (first number), and the total number of samples accumulated
in the prologue of the function (second number). This head sample
count provides an indicator of how frequently the function is invoked.

Each sampled line may contain several items. Some are optional (marked
below):

a. Source line offset. This number represents the line number
   in the function where the sample was collected. The line number is
   always relative to the line where symbol of the function is
   defined. So, if the function has its header at line 280, the offset
   13 is at line 293 in the file.

   Note that this offset should never be a negative number. This could
   happen in cases like macros. The debug machinery will register the
   line number at the point of macro expansion. So, if the macro was
   expanded in a line before the start of the function, the profile
   converter should emit a 0 as the offset (this means that the optimizers
   will not be able to associate a meaningful weight to the instructions
   in the macro).

b. [OPTIONAL] Discriminator. This is used if the sampled program
   was compiled with DWARF discriminator support
   (http://wiki.dwarfstd.org/index.php?title=Path_Discriminators).
   DWARF discriminators are unsigned integer values that allow the
   compiler to distinguish between multiple execution paths on the
   same source line location.

   For example, consider the line of code ``if (cond) foo(); else bar();``.
   If the predicate ``cond`` is true 80% of the time, then the edge
   into function ``foo`` should be considered to be taken most of the
   time. But both calls to ``foo`` and ``bar`` are at the same source
   line, so a sample count at that line is not sufficient. The
   compiler needs to know which part of that line is taken more
   frequently.

   This is what discriminators provide. In this case, the calls to
   ``foo`` and ``bar`` will be at the same line, but will have
   different discriminator values. This allows the compiler to correctly
   set edge weights into ``foo`` and ``bar``.

c. Number of samples. This is an integer quantity representing the
   number of samples collected by the profiler at this source
   location.

d. [OPTIONAL] Potential call targets and samples. If present, this
   line contains a call instruction. This models both direct and
   number of samples. For example,

   .. code-block:: console

     130: 7  foo:3  bar:2  baz:7

   The above means that at relative line offset 130 there is a call
   instruction that calls one of ``foo()``, ``bar()`` and ``baz()``,
   with ``baz()`` being the relatively more frequently called target.


Profiling with Instrumentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Clang also supports profiling via instrumentation. This requires building a
special instrumented version of the code and has some runtime
overhead during the profiling, but it provides more detailed results than a
sampling profiler. It also provides reproducible results, at least to the
extent that the code behaves consistently across runs.

Here are the steps for using profile guided optimization with
instrumentation:

1. Build an instrumented version of the code by compiling and linking with the
   ``-fprofile-instr-generate`` option.

   .. code-block:: console

     $ clang++ -O2 -fprofile-instr-generate code.cc -o code

2. Run the instrumented executable with inputs that reflect the typical usage.
   By default, the profile data will be written to a ``default.profraw`` file
   in the current directory. You can override that default by setting the
   ``LLVM_PROFILE_FILE`` environment variable to specify an alternate file.
   Any instance of ``%p`` in that file name will be replaced by the process
   ID, so that you can easily distinguish the profile output from multiple
   runs.

   .. code-block:: console

     $ LLVM_PROFILE_FILE="code-%p.profraw" ./code

3. Combine profiles from multiple runs and convert the "raw" profile format to
   the input expected by clang. Use the ``merge`` command of the llvm-profdata
   tool to do this.

   .. code-block:: console

     $ llvm-profdata merge -output=code.profdata code-*.profraw

   Note that this step is necessary even when there is only one "raw" profile,
   since the merge operation also changes the file format.

4. Build the code again using the ``-fprofile-instr-use`` option to specify the
   collected profile data.

   .. code-block:: console

     $ clang++ -O2 -fprofile-instr-use=code.profdata code.cc -o code

   You can repeat step 4 as often as you like without regenerating the
   profile. As you make changes to your code, clang may no longer be able to
   use the profile data. It will warn you when this happens.


Controlling Size of Debug Information
-------------------------------------

Debug info kind generated by Clang can be set by one of the flags listed
below. If multiple flags are present, the last one is used.

.. option:: -g0

  Don't generate any debug info (default).

.. option:: -gline-tables-only

  Generate line number tables only.

  This kind of debug info allows to obtain stack traces with function names,
  file names and line numbers (by such tools as ``gdb`` or ``addr2line``).  It
  doesn't contain any other data (e.g. description of local variables or
  function parameters).

.. option:: -fstandalone-debug

  Clang supports a number of optimizations to reduce the size of debug
  information in the binary. They work based on the assumption that
  the debug type information can be spread out over multiple
  compilation units.  For instance, Clang will not emit type
  definitions for types that are not needed by a module and could be
  replaced with a forward declaration.  Further, Clang will only emit
  type info for a dynamic C++ class in the module that contains the
  vtable for the class.

  The **-fstandalone-debug** option turns off these optimizations.
  This is useful when working with 3rd-party libraries that don't come
  with debug information.  Note that Clang will never emit type
  information for types that are not referenced at all by the program.

.. option:: -fno-standalone-debug

   On Darwin **-fstandalone-debug** is enabled by default. The
   **-fno-standalone-debug** option can be used to get to turn on the
   vtable-based optimization described above.

.. option:: -g

  Generate complete debug info.

Comment Parsing Options
-----------------------

Clang parses Doxygen and non-Doxygen style documentation comments and attaches
them to the appropriate declaration nodes.  By default, it only parses
Doxygen-style comments and ignores ordinary comments starting with ``//`` and
``/*``.

.. option:: -Wdocumentation

  Emit warnings about use of documentation comments.  This warning group is off
  by default.

  This includes checking that ``\param`` commands name parameters that actually
  present in the function signature, checking that ``\returns`` is used only on
  functions that actually return a value etc.

.. option:: -Wno-documentation-unknown-command

  Don't warn when encountering an unknown Doxygen command.

.. option:: -fparse-all-comments

  Parse all comments as documentation comments (including ordinary comments
  starting with ``//`` and ``/*``).

.. option:: -fcomment-block-commands=[commands]

  Define custom documentation commands as block commands.  This allows Clang to
  construct the correct AST for these custom commands, and silences warnings
  about unknown commands.  Several commands must be separated by a comma
  *without trailing space*; e.g. ``-fcomment-block-commands=foo,bar`` defines
  custom commands ``\foo`` and ``\bar``.

  It is also possible to use ``-fcomment-block-commands`` several times; e.g.
  ``-fcomment-block-commands=foo -fcomment-block-commands=bar`` does the same
  as above.

.. _c:

C Language Features
===================

The support for standard C in clang is feature-complete except for the
C99 floating-point pragmas.

Extensions supported by clang
-----------------------------

See :doc:`LanguageExtensions`.

Differences between various standard modes
------------------------------------------

clang supports the -std option, which changes what language mode clang
uses. The supported modes for C are c89, gnu89, c94, c99, gnu99 and
various aliases for those modes. If no -std option is specified, clang
defaults to gnu99 mode.

Differences between all ``c*`` and ``gnu*`` modes:

-  ``c*`` modes define "``__STRICT_ANSI__``".
-  Target-specific defines not prefixed by underscores, like "linux",
   are defined in ``gnu*`` modes.
-  Trigraphs default to being off in ``gnu*`` modes; they can be enabled by
   the -trigraphs option.
-  The parser recognizes "asm" and "typeof" as keywords in ``gnu*`` modes;
   the variants "``__asm__``" and "``__typeof__``" are recognized in all
   modes.
-  The Apple "blocks" extension is recognized by default in ``gnu*`` modes
   on some platforms; it can be enabled in any mode with the "-fblocks"
   option.
-  Arrays that are VLA's according to the standard, but which can be
   constant folded by the frontend are treated as fixed size arrays.
   This occurs for things like "int X[(1, 2)];", which is technically a
   VLA. ``c*`` modes are strictly compliant and treat these as VLAs.

Differences between ``*89`` and ``*99`` modes:

-  The ``*99`` modes default to implementing "inline" as specified in C99,
   while the ``*89`` modes implement the GNU version. This can be
   overridden for individual functions with the ``__gnu_inline__``
   attribute.
-  Digraphs are not recognized in c89 mode.
-  The scope of names defined inside a "for", "if", "switch", "while",
   or "do" statement is different. (example: "``if ((struct x {int
   x;}*)0) {}``".)
-  ``__STDC_VERSION__`` is not defined in ``*89`` modes.
-  "inline" is not recognized as a keyword in c89 mode.
-  "restrict" is not recognized as a keyword in ``*89`` modes.
-  Commas are allowed in integer constant expressions in ``*99`` modes.
-  Arrays which are not lvalues are not implicitly promoted to pointers
   in ``*89`` modes.
-  Some warnings are different.

c94 mode is identical to c89 mode except that digraphs are enabled in
c94 mode (FIXME: And ``__STDC_VERSION__`` should be defined!).

GCC extensions not implemented yet
----------------------------------

clang tries to be compatible with gcc as much as possible, but some gcc
extensions are not implemented yet:

-  clang does not support #pragma weak (`bug
   3679 <http://llvm.org/bugs/show_bug.cgi?id=3679>`_). Due to the uses
   described in the bug, this is likely to be implemented at some point,
   at least partially.
-  clang does not support decimal floating point types (``_Decimal32`` and
   friends) or fixed-point types (``_Fract`` and friends); nobody has
   expressed interest in these features yet, so it's hard to say when
   they will be implemented.
-  clang does not support nested functions; this is a complex feature
   which is infrequently used, so it is unlikely to be implemented
   anytime soon. In C++11 it can be emulated by assigning lambda
   functions to local variables, e.g:

   .. code-block:: cpp

     auto const local_function = [&](int parameter) {
       // Do something
     };
     ...
     local_function(1);

-  clang does not support global register variables; this is unlikely to
   be implemented soon because it requires additional LLVM backend
   support.
-  clang does not support static initialization of flexible array
   members. This appears to be a rarely used extension, but could be
   implemented pending user demand.
-  clang does not support
   ``__builtin_va_arg_pack``/``__builtin_va_arg_pack_len``. This is
   used rarely, but in some potentially interesting places, like the
   glibc headers, so it may be implemented pending user demand. Note
   that because clang pretends to be like GCC 4.2, and this extension
   was introduced in 4.3, the glibc headers will not try to use this
   extension with clang at the moment.
-  clang does not support the gcc extension for forward-declaring
   function parameters; this has not shown up in any real-world code
   yet, though, so it might never be implemented.

This is not a complete list; if you find an unsupported extension
missing from this list, please send an e-mail to cfe-dev. This list
currently excludes C++; see :ref:`C++ Language Features <cxx>`. Also, this
list does not include bugs in mostly-implemented features; please see
the `bug
tracker <http://llvm.org/bugs/buglist.cgi?quicksearch=product%3Aclang+component%3A-New%2BBugs%2CAST%2CBasic%2CDriver%2CHeaders%2CLLVM%2BCodeGen%2Cparser%2Cpreprocessor%2CSemantic%2BAnalyzer>`_
for known existing bugs (FIXME: Is there a section for bug-reporting
guidelines somewhere?).

Intentionally unsupported GCC extensions
----------------------------------------

-  clang does not support the gcc extension that allows variable-length
   arrays in structures. This is for a few reasons: one, it is tricky to
   implement, two, the extension is completely undocumented, and three,
   the extension appears to be rarely used. Note that clang *does*
   support flexible array members (arrays with a zero or unspecified
   size at the end of a structure).
-  clang does not have an equivalent to gcc's "fold"; this means that
   clang doesn't accept some constructs gcc might accept in contexts
   where a constant expression is required, like "x-x" where x is a
   variable.
-  clang does not support ``__builtin_apply`` and friends; this extension
   is extremely obscure and difficult to implement reliably.

.. _c_ms:

Microsoft extensions
--------------------

clang has some experimental support for extensions from Microsoft Visual
C++; to enable it, use the ``-fms-extensions`` command-line option. This is
the default for Windows targets. Note that the support is incomplete.
Some constructs such as ``dllexport`` on classes are ignored with a warning,
and others such as `Microsoft IDL annotations
<http://msdn.microsoft.com/en-us/library/8tesw2eh.aspx>`_ are silently
ignored.

clang has a ``-fms-compatibility`` flag that makes clang accept enough
invalid C++ to be able to parse most Microsoft headers. For example, it
allows `unqualified lookup of dependent base class members
<http://clang.llvm.org/compatibility.html#dep_lookup_bases>`_, which is
a common compatibility issue with clang. This flag is enabled by default
for Windows targets.

``-fdelayed-template-parsing`` lets clang delay parsing of function template
definitions until the end of a translation unit. This flag is enabled by
default for Windows targets.

-  clang allows setting ``_MSC_VER`` with ``-fmsc-version=``. It defaults to
   1700 which is the same as Visual C/C++ 2012. Any number is supported
   and can greatly affect what Windows SDK and c++stdlib headers clang
   can compile.
-  clang does not support the Microsoft extension where anonymous record
   members can be declared using user defined typedefs.
-  clang supports the Microsoft ``#pragma pack`` feature for controlling
   record layout. GCC also contains support for this feature, however
   where MSVC and GCC are incompatible clang follows the MSVC
   definition.
-  clang supports the Microsoft ``#pragma comment(lib, "foo.lib")`` feature for
   automatically linking against the specified library.  Currently this feature
   only works with the Visual C++ linker.
-  clang supports the Microsoft ``#pragma comment(linker, "/flag:foo")`` feature
   for adding linker flags to COFF object files.  The user is responsible for
   ensuring that the linker understands the flags.
-  clang defaults to C++11 for Windows targets.

.. _cxx:

C++ Language Features
=====================

clang fully implements all of standard C++98 except for exported
templates (which were removed in C++11), and all of standard C++11
and the current draft standard for C++1y.

Controlling implementation limits
---------------------------------

.. option:: -fbracket-depth=N

  Sets the limit for nested parentheses, brackets, and braces to N.  The
  default is 256.

.. option:: -fconstexpr-depth=N

  Sets the limit for recursive constexpr function invocations to N.  The
  default is 512.

.. option:: -ftemplate-depth=N

  Sets the limit for recursively nested template instantiations to N.  The
  default is 256.

.. option:: -foperator-arrow-depth=N

  Sets the limit for iterative calls to 'operator->' functions to N.  The
  default is 256.

.. _objc:

Objective-C Language Features
=============================

.. _objcxx:

Objective-C++ Language Features
===============================


.. _target_features:

Target-Specific Features and Limitations
========================================

CPU Architectures Features and Limitations
------------------------------------------

X86
^^^

The support for X86 (both 32-bit and 64-bit) is considered stable on
Darwin (Mac OS X), Linux, FreeBSD, and Dragonfly BSD: it has been tested
to correctly compile many large C, C++, Objective-C, and Objective-C++
codebases.

On ``x86_64-mingw32``, passing i128(by value) is incompatible with the
Microsoft x64 calling convention. You might need to tweak
``WinX86_64ABIInfo::classify()`` in lib/CodeGen/TargetInfo.cpp.

For the X86 target, clang supports the :option:`-m16` command line
argument which enables 16-bit code output. This is broadly similar to
using ``asm(".code16gcc")`` with the GNU toolchain. The generated code
and the ABI remains 32-bit but the assembler emits instructions
appropriate for a CPU running in 16-bit mode, with address-size and
operand-size prefixes to enable 32-bit addressing and operations.

ARM
^^^

The support for ARM (specifically ARMv6 and ARMv7) is considered stable
on Darwin (iOS): it has been tested to correctly compile many large C,
C++, Objective-C, and Objective-C++ codebases. Clang only supports a
limited number of ARM architectures. It does not yet fully support
ARMv5, for example.

PowerPC
^^^^^^^

The support for PowerPC (especially PowerPC64) is considered stable
on Linux and FreeBSD: it has been tested to correctly compile many
large C and C++ codebases. PowerPC (32bit) is still missing certain
features (e.g. PIC code on ELF platforms).

Other platforms
^^^^^^^^^^^^^^^

clang currently contains some support for other architectures (e.g. Sparc);
however, significant pieces of code generation are still missing, and they
haven't undergone significant testing.

clang contains limited support for the MSP430 embedded processor, but
both the clang support and the LLVM backend support are highly
experimental.

Other platforms are completely unsupported at the moment. Adding the
minimal support needed for parsing and semantic analysis on a new
platform is quite easy; see ``lib/Basic/Targets.cpp`` in the clang source
tree. This level of support is also sufficient for conversion to LLVM IR
for simple programs. Proper support for conversion to LLVM IR requires
adding code to ``lib/CodeGen/CGCall.cpp`` at the moment; this is likely to
change soon, though. Generating assembly requires a suitable LLVM
backend.

Operating System Features and Limitations
-----------------------------------------

Darwin (Mac OS X)
^^^^^^^^^^^^^^^^^

Thread Sanitizer is not supported.

Windows
^^^^^^^

Clang has experimental support for targeting "Cygming" (Cygwin / MinGW)
platforms.

See also :ref:`Microsoft Extensions <c_ms>`.

Cygwin
""""""

Clang works on Cygwin-1.7.

MinGW32
"""""""

Clang works on some mingw32 distributions. Clang assumes directories as
below;

-  ``C:/mingw/include``
-  ``C:/mingw/lib``
-  ``C:/mingw/lib/gcc/mingw32/4.[3-5].0/include/c++``

On MSYS, a few tests might fail.

MinGW-w64
"""""""""

For 32-bit (i686-w64-mingw32), and 64-bit (x86\_64-w64-mingw32), Clang
assumes as below;

-  ``GCC versions 4.5.0 to 4.5.3, 4.6.0 to 4.6.2, or 4.7.0 (for the C++ header search path)``
-  ``some_directory/bin/gcc.exe``
-  ``some_directory/bin/clang.exe``
-  ``some_directory/bin/clang++.exe``
-  ``some_directory/bin/../include/c++/GCC_version``
-  ``some_directory/bin/../include/c++/GCC_version/x86_64-w64-mingw32``
-  ``some_directory/bin/../include/c++/GCC_version/i686-w64-mingw32``
-  ``some_directory/bin/../include/c++/GCC_version/backward``
-  ``some_directory/bin/../x86_64-w64-mingw32/include``
-  ``some_directory/bin/../i686-w64-mingw32/include``
-  ``some_directory/bin/../include``

This directory layout is standard for any toolchain you will find on the
official `MinGW-w64 website <http://mingw-w64.sourceforge.net>`_.

Clang expects the GCC executable "gcc.exe" compiled for
``i686-w64-mingw32`` (or ``x86_64-w64-mingw32``) to be present on PATH.

`Some tests might fail <http://llvm.org/bugs/show_bug.cgi?id=9072>`_ on
``x86_64-w64-mingw32``.

.. _clang-cl:

clang-cl
========

clang-cl is an alternative command-line interface to Clang driver, designed for
compatibility with the Visual C++ compiler, cl.exe.

To enable clang-cl to find system headers, libraries, and the linker when run
from the command-line, it should be executed inside a Visual Studio Native Tools
Command Prompt or a regular Command Prompt where the environment has been set
up using e.g. `vcvars32.bat <http://msdn.microsoft.com/en-us/library/f2ccy3wt.aspx>`_.

clang-cl can also be used from inside Visual Studio  by using an LLVM Platform
Toolset.

Command-Line Options
--------------------

To be compatible with cl.exe, clang-cl supports most of the same command-line
options. Those options can start with either ``/`` or ``-``. It also supports
some of Clang's core options, such as the ``-W`` options.

Options that are known to clang-cl, but not currently supported, are ignored
with a warning. For example:

  ::

    clang-cl.exe: warning: argument unused during compilation: '/Zi'

To suppress warnings about unused arguments, use the ``-Qunused-arguments`` option.

Options that are not known to clang-cl will cause errors. If they are spelled with a
leading ``/``, they will be mistaken for a filename:

  ::

    clang-cl.exe: error: no such file or directory: '/foobar'

Please `file a bug <http://llvm.org/bugs/enter_bug.cgi?product=clang&component=Driver>`_
for any valid cl.exe flags that clang-cl does not understand.

Execute ``clang-cl /?`` to see a list of supported options:

  ::

    CL.EXE COMPATIBILITY OPTIONS:
      /?                     Display available options
      /arch:<value>          Set architecture for code generation
      /C                     Don't discard comments when preprocessing
      /c                     Compile only
      /D <macro[=value]>     Define macro
      /EH<value>             Exception handling model
      /EP                    Disable linemarker output and preprocess to stdout
      /E                     Preprocess to stdout
      /fallback              Fall back to cl.exe if clang-cl fails to compile
      /FA                    Output assembly code file during compilation
      /Fa<file or directory> Output assembly code to this file during compilation
      /Fe<file or directory> Set output executable file or directory (ends in / or \)
      /FI <value>            Include file before parsing
      /Fi<file>              Set preprocess output file name
      /Fo<file or directory> Set output object file, or directory (ends in / or \)
      /GF-                   Disable string pooling
      /GR-                   Disable emission of RTTI data
      /GR                    Enable emission of RTTI data
      /Gw-                   Don't put each data item in its own section
      /Gw                    Put each data item in its own section
      /Gy-                   Don't put each function in its own section
      /Gy                    Put each function in its own section
      /help                  Display available options
      /I <dir>               Add directory to include search path
      /J                     Make char type unsigned
      /LDd                   Create debug DLL
      /LD                    Create DLL
      /link <options>        Forward options to the linker
      /MDd                   Use DLL debug run-time
      /MD                    Use DLL run-time
      /MTd                   Use static debug run-time
      /MT                    Use static run-time
      /Ob0                   Disable inlining
      /Od                    Disable optimization
      /Oi-                   Disable use of builtin functions
      /Oi                    Enable use of builtin functions
      /Os                    Optimize for size
      /Ot                    Optimize for speed
      /Ox                    Maximum optimization
      /Oy-                   Disable frame pointer omission
      /Oy                    Enable frame pointer omission
      /O<n>                  Optimization level
      /P                     Preprocess to file
      /showIncludes          Print info about included files to stderr
      /TC                    Treat all source files as C
      /Tc <filename>         Specify a C source file
      /TP                    Treat all source files as C++
      /Tp <filename>         Specify a C++ source file
      /U <macro>             Undefine macro
      /vd<value>             Control vtordisp placement
      /vmb                   Use a best-case representation method for member pointers
      /vmg                   Use a most-general representation for member pointers
      /vmm                   Set the default most-general representation to multiple inheritance
      /vms                   Set the default most-general representation to single inheritance
      /vmv                   Set the default most-general representation to virtual inheritance
      /W0                    Disable all warnings
      /W1                    Enable -Wall
      /W2                    Enable -Wall
      /W3                    Enable -Wall
      /W4                    Enable -Wall
      /Wall                  Enable -Wall
      /WX-                   Do not treat warnings as errors
      /WX                    Treat warnings as errors
      /w                     Disable all warnings
      /Zi                    Enable debug information
      /Zs                    Syntax-check only

    OPTIONS:
      -###                  Print (but do not run) the commands to run for this compilation
      -fms-compatibility-version=<value>
                            Dot-separated value representing the Microsoft compiler version
                            number to report in _MSC_VER (0 = don't define it (default))
      -fmsc-version=<value> Microsoft compiler version number to report in _MSC_VER (0 = don't
                            define it (default))
      -fsanitize-blacklist=<value>
                            Path to blacklist file for sanitizers
      -fsanitize=<check>    Enable runtime instrumentation for bug detection: address (memory
                            errors) | thread (race detection) | undefined (miscellaneous
                            undefined behavior)
      -mllvm <value>        Additional arguments to forward to LLVM's option processing
      -Qunused-arguments    Don't emit warning for unused driver arguments
      --target=<value>      Generate code for the given target
      -v                    Show commands to run and use verbose output
      -W<warning>           Enable the specified warning
      -Xclang <arg>         Pass <arg> to the clang compiler

The /fallback Option
^^^^^^^^^^^^^^^^^^^^

When clang-cl is run with the ``/fallback`` option, it will first try to
compile files itself. For any file that it fails to compile, it will fall back
and try to compile the file by invoking cl.exe.

This option is intended to be used as a temporary means to build projects where
clang-cl cannot successfully compile all the files. clang-cl may fail to compile
a file either because it cannot generate code for some C++ feature, or because
it cannot parse some Microsoft language extension.
