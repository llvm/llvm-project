.. index:: modularize

==================================
Modularize User's Manual
==================================

.. toctree::
   :hidden:

   ModularizeUsage

:program:`modularize` is a standalone tool that checks whether a set of headers
provides the consistent definitions required to use modules. For example, it
detects whether the same entity (say, a NULL macro or size_t typedef) is
defined in multiple headers or whether a header produces different definitions
under different circumstances. These conditions cause modules built from the
headers to behave poorly, and should be fixed before introducing a module
map.

:program:`modularize` also has an assistant mode option for generating
a module map file based on the provided header list. The generated file
is a functional module map that can be used as a starting point for a
module.map file.

Getting Started
===============

To build from source:

1. Read `Getting Started with the LLVM System`_ and `Clang Tools
   Documentation`_ for information on getting sources for LLVM, Clang, and
   Clang Extra Tools.

2. `Getting Started with the LLVM System`_ and `Building LLVM with CMake`_ give
   directions for how to build. With sources all checked out into the
   right place the LLVM build will build Clang Extra Tools and their
   dependencies automatically.

   * If using CMake, you can also use the ``modularize`` target to build
     just the modularize tool and its dependencies.

Before continuing, take a look at :doc:`ModularizeUsage` to see how to invoke
modularize.

.. _Getting Started with the LLVM System: https://llvm.org/docs/GettingStarted.html
.. _Building LLVM with CMake: https://llvm.org/docs/CMake.html
.. _Clang Tools Documentation: https://clang.llvm.org/docs/ClangTools.html

What Modularize Checks
======================

Modularize will check for the following:

* Duplicate global type and variable definitions
* Duplicate macro definitions
* Macro instances, 'defined(macro)', or #if, #elif, #ifdef, #ifndef conditions
  that evaluate differently in a header
* #include directives inside 'extern "C/C++" {}' or 'namespace (name) {}' blocks
* Module map header coverage completeness (in the case of a module map input
  only)

Modularize will do normal C/C++ parsing, reporting normal errors and warnings,
but will also report special error messages like the following::

  error: '(symbol)' defined at multiple locations:
     (file):(row):(column)
     (file):(row):(column)

  error: header '(file)' has different contents depending on how it was included

The latter might be followed by messages like the following::

  note: '(symbol)' in (file) at (row):(column) not always provided

Checks will also be performed for macro expansions, defined(macro)
expressions, and preprocessor conditional directives that evaluate
inconsistently, and can produce error messages like the following::

   (...)/SubHeader.h:11:5:
  #if SYMBOL == 1
      ^
  error: Macro instance 'SYMBOL' has different values in this header,
         depending on how it was included.
    'SYMBOL' expanded to: '1' with respect to these inclusion paths:
      (...)/Header1.h
        (...)/SubHeader.h
  (...)/SubHeader.h:3:9:
  #define SYMBOL 1
          ^
  Macro defined here.
    'SYMBOL' expanded to: '2' with respect to these inclusion paths:
      (...)/Header2.h
          (...)/SubHeader.h
  (...)/SubHeader.h:7:9:
  #define SYMBOL 2
          ^
  Macro defined here.

Checks will also be performed for '#include' directives that are
nested inside 'extern "C/C++" {}' or 'namespace (name) {}' blocks,
and can produce error message like the following::

  IncludeInExtern.h:2:3:
  #include "Empty.h"
  ^
  error: Include directive within extern "C" {}.
  IncludeInExtern.h:1:1:
  extern "C" {
  ^
  The "extern "C" {}" block is here.

.. _module-map-coverage:

Module Map Coverage Check
=========================

The coverage check uses the Clang library to read and parse the
module map file. Starting at the module map file directory, or just the
include paths, if specified, it will collect the names of all the files it
considers headers (no extension, .h, or .inc--if you need more, modify the
isHeader function). It then compares the headers against those referenced
in the module map, either explicitly named, or implicitly named via an
umbrella directory or umbrella file, as parsed by the ModuleMap object.
If headers are found which are not referenced or covered by an umbrella
directory or file, warning messages will be produced, and this program
will return an error code of 1. If no problems are found, an error code of
0 is returned.

Note that in the case of umbrella headers, this tool invokes the compiler
to preprocess the file, and uses a callback to collect the header files
included by the umbrella header or any of its nested includes. If any
front end options are needed for these compiler invocations, these
can be included on the command line after the module map file argument.

Warning message have the form:

  warning: module.modulemap does not account for file: Level3A.h

Note that for the case of the module map referencing a file that does
not exist, the module map parser in Clang will (at the time of this
writing) display an error message.

To limit the checks :program:`modularize` does to just the module
map coverage check, use the ``-coverage-check-only option``.

For example::

  modularize -coverage-check-only module.modulemap

.. _module-map-generation:

Module Map Generation
=====================

If you specify the ``-module-map-path=<module map file>``,
:program:`modularize` will output a module map based on the input header list.
A module will be created for each header. Also, if the header in the header
list is a partial path, a nested module hierarchy will be created in which a
module will be created for each subdirectory component in the header path,
with the header itself represented by the innermost module. If other headers
use the same subdirectories, they will be enclosed in these same modules also.

For example, for the header list::

  SomeTypes.h
  SomeDecls.h
  SubModule1/Header1.h
  SubModule1/Header2.h
  SubModule2/Header3.h
  SubModule2/Header4.h
  SubModule2.h

The following module map will be generated::

  // Output/NoProblemsAssistant.txt
  // Generated by: modularize -module-map-path=Output/NoProblemsAssistant.txt \
       -root-module=Root NoProblemsAssistant.modularize
  
  module SomeTypes {
    header "SomeTypes.h"
    export *
  }
  module SomeDecls {
    header "SomeDecls.h"
    export *
  }
  module SubModule1 {
    module Header1 {
      header "SubModule1/Header1.h"
      export *
    }
    module Header2 {
      header "SubModule1/Header2.h"
      export *
    }
  }
  module SubModule2 {
    module Header3 {
      header "SubModule2/Header3.h"
      export *
    }
    module Header4 {
      header "SubModule2/Header4.h"
      export *
    }
    header "SubModule2.h"
    export *
  }

An optional ``-root-module=<root-name>`` option can be used to cause a root module
to be created which encloses all the modules.

An optional ``-problem-files-list=<problem-file-name>`` can be used to input
a list of files to be excluded, perhaps as a temporary stop-gap measure until
problem headers can be fixed.

For example, with the same header list from above::

  // Output/NoProblemsAssistant.txt
  // Generated by: modularize -module-map-path=Output/NoProblemsAssistant.txt \
       -root-module=Root NoProblemsAssistant.modularize
  
  module Root {
    module SomeTypes {
      header "SomeTypes.h"
      export *
    }
    module SomeDecls {
      header "SomeDecls.h"
      export *
    }
    module SubModule1 {
      module Header1 {
        header "SubModule1/Header1.h"
        export *
      }
      module Header2 {
        header "SubModule1/Header2.h"
        export *
      }
    }
    module SubModule2 {
      module Header3 {
        header "SubModule2/Header3.h"
        export *
      }
      module Header4 {
        header "SubModule2/Header4.h"
        export *
      }
      header "SubModule2.h"
      export *
    }
  }

Note that headers with dependents will be ignored with a warning, as the
Clang module mechanism doesn't support headers the rely on other headers
to be included first.

The module map format defines some keywords which can't be used in module
names. If a header has one of these names, an underscore ('_') will be
prepended to the name. For example, if the header name is ``header.h``,
because ``header`` is a keyword, the module name will be ``_header``.
For a list of the module map keywords, please see:
`Lexical structure <https://clang.llvm.org/docs/Modules.html#lexical-structure>`_
