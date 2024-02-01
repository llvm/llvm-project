====================
Standard C++ Modules
====================

.. contents::
   :local:

Introduction
============

The term ``modules`` has a lot of meanings. For the users of Clang, modules may
refer to ``Objective-C Modules``, ``Clang C++ Modules`` (or ``Clang Header Modules``,
etc.) or ``Standard C++ Modules``. The implementation of all these kinds of modules in Clang
has a lot of shared code, but from the perspective of users, their semantics and
command line interfaces are very different. This document focuses on
an introduction of how to use standard C++ modules in Clang.

There is already a detailed document about `Clang modules <Modules.html>`_, it
should be helpful to read `Clang modules <Modules.html>`_ if you want to know
more about the general idea of modules. Since standard C++ modules have different semantics
(and work flows) from `Clang modules`, this page describes the background and use of
Clang with standard C++ modules.

Modules exist in two forms in the C++ Language Specification. They can refer to
either "Named Modules" or to "Header Units". This document covers both forms.

Standard C++ Named modules
==========================

This document was intended to be a manual first and foremost, however, we consider it helpful to
introduce some language background here for readers who are not familiar with
the new language feature. This document is not intended to be a language
tutorial; it will only introduce necessary concepts about the
structure and building of the project.

Background and terminology
--------------------------

Modules
~~~~~~~

In this document, the term ``Modules``/``modules`` refers to standard C++ modules
feature if it is not decorated by ``Clang``.

Clang Modules
~~~~~~~~~~~~~

In this document, the term ``Clang Modules``/``Clang modules`` refer to Clang
c++ modules extension. These are also known as ``Clang header modules``,
``Clang module map modules`` or ``Clang c++ modules``.

Module and module unit
~~~~~~~~~~~~~~~~~~~~~~

A module consists of one or more module units. A module unit is a special
translation unit. Every module unit must have a module declaration. The syntax
of the module declaration is:

.. code-block:: c++

  [export] module module_name[:partition_name];

Terms enclosed in ``[]`` are optional. The syntax of ``module_name`` and ``partition_name``
in regex form corresponds to ``[a-zA-Z_][a-zA-Z_0-9\.]*``. In particular, a literal dot ``.``
in the name has no semantic meaning (e.g. implying a hierarchy).

In this document, module units are classified into:

* Primary module interface unit.

* Module implementation unit.

* Module interface partition unit.

* Internal module partition unit.

A primary module interface unit is a module unit whose module declaration is
``export module module_name;``. The ``module_name`` here denotes the name of the
module. A module should have one and only one primary module interface unit.

A module implementation unit is a module unit whose module declaration is
``module module_name;``. A module could have multiple module implementation
units with the same declaration.

A module interface partition unit is a module unit whose module declaration is
``export module module_name:partition_name;``. The ``partition_name`` should be
unique within any given module.

An internal module partition unit is a module unit whose module declaration
is ``module module_name:partition_name;``. The ``partition_name`` should be
unique within any given module.

In this document, we use the following umbrella terms:

* A ``module interface unit`` refers to either a ``primary module interface unit``
  or a ``module interface partition unit``.

* An ``importable module unit`` refers to either a ``module interface unit``
  or a ``internal module partition unit``.

* A ``module partition unit`` refers to either a ``module interface partition unit``
  or a ``internal module partition unit``.

Built Module Interface file
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A ``Built Module Interface file`` stands for the precompiled result of an importable module unit.
It is also called the acronym ``BMI`` generally.

Global module fragment
~~~~~~~~~~~~~~~~~~~~~~

In a module unit, the section from ``module;`` to the module declaration is called the global module fragment.


How to build projects using modules
-----------------------------------

Quick Start
~~~~~~~~~~~

Let's see a "hello world" example that uses modules.

.. code-block:: c++

  // Hello.cppm
  module;
  #include <iostream>
  export module Hello;
  export void hello() {
    std::cout << "Hello World!\n";
  }

  // use.cpp
  import Hello;
  int main() {
    hello();
    return 0;
  }

Then we type:

.. code-block:: console

  $ clang++ -std=c++20 Hello.cppm --precompile -o Hello.pcm
  $ clang++ -std=c++20 use.cpp -fmodule-file=Hello=Hello.pcm Hello.pcm -o Hello.out
  $ ./Hello.out
  Hello World!

In this example, we make and use a simple module ``Hello`` which contains only a
primary module interface unit ``Hello.cppm``.

Then let's see a little bit more complex "hello world" example which uses the 4 kinds of module units.

.. code-block:: c++

  // M.cppm
  export module M;
  export import :interface_part;
  import :impl_part;
  export void Hello();

  // interface_part.cppm
  export module M:interface_part;
  export void World();

  // impl_part.cppm
  module;
  #include <iostream>
  #include <string>
  module M:impl_part;
  import :interface_part;

  std::string W = "World.";
  void World() {
    std::cout << W << std::endl;
  }

  // Impl.cpp
  module;
  #include <iostream>
  module M;
  void Hello() {
    std::cout << "Hello ";
  }

  // User.cpp
  import M;
  int main() {
    Hello();
    World();
    return 0;
  }

Then we are able to compile the example by the following command:

.. code-block:: console

  # Precompiling the module
  $ clang++ -std=c++20 interface_part.cppm --precompile -o M-interface_part.pcm
  $ clang++ -std=c++20 impl_part.cppm --precompile -fprebuilt-module-path=. -o M-impl_part.pcm
  $ clang++ -std=c++20 M.cppm --precompile -fprebuilt-module-path=. -o M.pcm
  $ clang++ -std=c++20 Impl.cpp -fprebuilt-module-path=. -c -o Impl.o

  # Compiling the user
  $ clang++ -std=c++20 User.cpp -fprebuilt-module-path=. -c -o User.o

  # Compiling the module and linking it together
  $ clang++ -std=c++20 M-interface_part.pcm -fprebuilt-module-path=. -c -o M-interface_part.o
  $ clang++ -std=c++20 M-impl_part.pcm -fprebuilt-module-path=. -c -o M-impl_part.o
  $ clang++ -std=c++20 M.pcm -fprebuilt-module-path=. -c -o M.o
  $ clang++ User.o M-interface_part.o  M-impl_part.o M.o Impl.o -o a.out

We explain the options in the following sections.

How to enable standard C++ modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, standard C++ modules are enabled automatically
if the language standard is ``-std=c++20`` or newer.

How to produce a BMI
~~~~~~~~~~~~~~~~~~~~

We can generate a BMI for an importable module unit by either ``--precompile``
or ``-fmodule-output`` flags.

The ``--precompile`` option generates the BMI as the output of the compilation and the output path
can be specified using the ``-o`` option.

The ``-fmodule-output`` option generates the BMI as a by-product of the compilation.
If ``-fmodule-output=`` is specified, the BMI will be emitted the specified location. Then if
``-fmodule-output`` and ``-c`` are specified, the BMI will be emitted in the directory of the
output file with the name of the input file with the new extension ``.pcm``. Otherwise, the BMI
will be emitted in the working directory with the name of the input file with the new extension
``.pcm``.

The style to generate BMIs by ``--precompile`` is called two-phase compilation since it takes
2 steps to compile a source file to an object file. The style to generate BMIs by ``-fmodule-output``
is called one-phase compilation respectively. The one-phase compilation model is simpler
for build systems to implement and the two-phase compilation has the potential to compile faster due
to higher parallelism. As an example, if there are two module units A and B, and B depends on A, the
one-phase compilation model would need to compile them serially, whereas the two-phase compilation
model may be able to compile them simultaneously if the compilation from A.pcm to A.o takes a long
time.

File name requirement
~~~~~~~~~~~~~~~~~~~~~

The file name of an ``importable module unit`` should end with ``.cppm``
(or ``.ccm``, ``.cxxm``, ``.c++m``). The file name of a ``module implementation unit``
should end with ``.cpp`` (or ``.cc``, ``.cxx``, ``.c++``).

The file name of BMIs should end with ``.pcm``.
The file name of the BMI of a ``primary module interface unit`` should be ``module_name.pcm``.
The file name of BMIs of ``module partition unit`` should be ``module_name-partition_name.pcm``.

If the file names use different extensions, Clang may fail to build the module.
For example, if the filename of an ``importable module unit`` ends with ``.cpp`` instead of ``.cppm``,
then we can't generate a BMI for the ``importable module unit`` by ``--precompile`` option
since ``--precompile`` option now would only run preprocessor, which is equal to `-E` now.
If we want the filename of an ``importable module unit`` ends with other suffixes instead of ``.cppm``,
we could put ``-x c++-module`` in front of the file. For example,

.. code-block:: c++

  // Hello.cpp
  module;
  #include <iostream>
  export module Hello;
  export void hello() {
    std::cout << "Hello World!\n";
  }

  // use.cpp
  import Hello;
  int main() {
    hello();
    return 0;
  }

Now the filename of the ``module interface`` ends with ``.cpp`` instead of ``.cppm``,
we can't compile them by the original command lines. But we are still able to do it by:

.. code-block:: console

  $ clang++ -std=c++20 -x c++-module Hello.cpp --precompile -o Hello.pcm
  $ clang++ -std=c++20 use.cpp -fprebuilt-module-path=. Hello.pcm -o Hello.out
  $ ./Hello.out
  Hello World!

Module name requirement
~~~~~~~~~~~~~~~~~~~~~~~

[module.unit]p1 says:

.. code-block:: text

  All module-names either beginning with an identifier consisting of std followed by zero
  or more digits or containing a reserved identifier ([lex.name]) are reserved and shall not
  be specified in a module-declaration; no diagnostic is required. If any identifier in a reserved
  module-name is a reserved identifier, the module name is reserved for use by C++ implementations;
  otherwise it is reserved for future standardization.

So all of the following name is not valid by default:

.. code-block:: text

    std
    std1
    std.foo
    __test
    // and so on ...

If you still want to use the reserved module names for any reason, use
``-Wno-reserved-module-identifier`` to suppress the warning.

How to specify the dependent BMIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are 3 methods to specify the dependent BMIs:

* (1) ``-fprebuilt-module-path=<path/to/directory>``.
* (2) ``-fmodule-file=<path/to/BMI>`` (Deprecated).
* (3) ``-fmodule-file=<module-name>=<path/to/BMI>``.

The option ``-fprebuilt-module-path`` tells the compiler the path where to search for dependent BMIs.
It may be used multiple times just like ``-I`` for specifying paths for header files. The look up rule here is:

* (1) When we import module M. The compiler would look up M.pcm in the directories specified
  by ``-fprebuilt-module-path``.
* (2) When we import partition module unit M:P. The compiler would look up M-P.pcm in the
  directories specified by ``-fprebuilt-module-path``.

The option ``-fmodule-file=<path/to/BMI>`` tells the compiler to load the specified BMI directly.
The option ``-fmodule-file=<module-name>=<path/to/BMI>`` tells the compiler to load the specified BMI
for the module specified by ``<module-name>`` when necessary. The main difference is that
``-fmodule-file=<path/to/BMI>`` will load the BMI eagerly, whereas
``-fmodule-file=<module-name>=<path/to/BMI>`` will only load the BMI lazily, which is similar
with ``-fprebuilt-module-path``. The option ``-fmodule-file=<path/to/BMI>`` for named modules is deprecated
and is planning to be removed in future versions.

In case all ``-fprebuilt-module-path=<path/to/directory>``, ``-fmodule-file=<path/to/BMI>`` and
``-fmodule-file=<module-name>=<path/to/BMI>`` exist, the ``-fmodule-file=<path/to/BMI>`` option
takes highest precedence and ``-fmodule-file=<module-name>=<path/to/BMI>`` will take the second
highest precedence.

We need to specify all the dependent (directly and indirectly) BMIs.
See https://github.com/llvm/llvm-project/issues/62707 for detail.

When we compile a ``module implementation unit``, we must specify the BMI of the corresponding
``primary module interface unit``.
Since the language specification says a module implementation unit implicitly imports
the primary module interface unit.

  [module.unit]p8

  A module-declaration that contains neither an export-keyword nor a module-partition implicitly
  imports the primary module interface unit of the module as if by a module-import-declaration.

All of the 3 options ``-fprebuilt-module-path=<path/to/directory>``, ``-fmodule-file=<path/to/BMI>``
and ``-fmodule-file=<module-name>=<path/to/BMI>`` may occur multiple times.
For example, the command line to compile ``M.cppm`` in
the above example could be rewritten into:

.. code-block:: console

  $ clang++ -std=c++20 M.cppm --precompile -fmodule-file=M:interface_part=M-interface_part.pcm -fmodule-file=M:impl_part=M-impl_part.pcm -o M.pcm

When there are multiple ``-fmodule-file=<module-name>=`` options for the same
``<module-name>``, the last ``-fmodule-file=<module-name>=`` will override the previous
``-fmodule-file=<module-name>=`` options.

``-fprebuilt-module-path`` is more convenient and ``-fmodule-file`` is faster since
it saves time for file lookup.

Remember that module units still have an object counterpart to the BMI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is easy to forget to compile BMIs at first since we may envision module interfaces like headers.
However, this is not true.
Module units are translation units. We need to compile them to object files
and link the object files like the example shows.

For example, the traditional compilation processes for headers are like:

.. code-block:: text

  src1.cpp -+> clang++ src1.cpp --> src1.o ---,
  hdr1.h  --'                                 +-> clang++ src1.o src2.o ->  executable
  hdr2.h  --,                                 |
  src2.cpp -+> clang++ src2.cpp --> src2.o ---'

And the compilation process for module units are like:

.. code-block:: text

                src1.cpp ----------------------------------------+> clang++ src1.cpp -------> src1.o -,
  (header unit) hdr1.h    -> clang++ hdr1.h ...    -> hdr1.pcm --'                                    +-> clang++ src1.o mod1.o src2.o ->  executable
                mod1.cppm -> clang++ mod1.cppm ... -> mod1.pcm --,--> clang++ mod1.pcm ... -> mod1.o -+
                src2.cpp ----------------------------------------+> clang++ src2.cpp -------> src2.o -'

As the diagrams show, we need to compile the BMI from module units to object files and link the object files.
(But we can't do this for the BMI from header units. See the later section for the definition of header units)

If we want to create a module library, we can't just ship the BMIs in an archive.
We must compile these BMIs(``*.pcm``) into object files(``*.o``) and add those object files to the archive instead.

Consistency Requirement
~~~~~~~~~~~~~~~~~~~~~~~

If we envision modules as a cache to speed up compilation, then - as with other caching techniques -
it is important to keep cache consistency.
So **currently** Clang will do very strict check for consistency.

Options consistency
^^^^^^^^^^^^^^^^^^^

The language option of module units and their non-module-unit users should be consistent.
The following example is not allowed:

.. code-block:: c++

  // M.cppm
  export module M;

  // Use.cpp
  import M;

.. code-block:: console

  $ clang++ -std=c++20 M.cppm --precompile -o M.pcm
  $ clang++ -std=c++23 Use.cpp -fprebuilt-module-path=.

The compiler would reject the example due to the inconsistent language options.
Not all options are language options.
For example, the following example is allowed:

.. code-block:: console

  $ clang++ -std=c++20 M.cppm --precompile -o M.pcm
  # Inconsistent optimization level.
  $ clang++ -std=c++20 -O3 Use.cpp -fprebuilt-module-path=.
  # Inconsistent debugging level.
  $ clang++ -std=c++20 -g Use.cpp -fprebuilt-module-path=.

Although the two examples have inconsistent optimization and debugging level, both of them are accepted.

Note that **currently** the compiler doesn't consider inconsistent macro definition a problem. For example:

.. code-block:: console

  $ clang++ -std=c++20 M.cppm --precompile -o M.pcm
  # Inconsistent optimization level.
  $ clang++ -std=c++20 -O3 -DNDEBUG Use.cpp -fprebuilt-module-path=.

Currently Clang would accept the above example. But it may produce surprising results if the
debugging code depends on consistent use of ``NDEBUG`` also in other translation units.

Definitions consistency
^^^^^^^^^^^^^^^^^^^^^^^

The C++ language defines that same declarations in different translation units should have
the same definition, as known as ODR (One Definition Rule). Prior to modules, the translation
units don't dependent on each other and the compiler itself can't perform a strong
ODR violation check. With the introduction of modules, now the compiler have
the chance to perform ODR violations with language semantics across translation units.

However, in the practice, we found the existing ODR checking mechanism is not stable
enough. Many people suffers from the false positive ODR violation diagnostics, AKA,
the compiler are complaining two identical declarations have different definitions
incorrectly. Also the true positive ODR violations are rarely reported.
Also we learned that MSVC don't perform ODR check for declarations in the global module
fragment.

So in order to get better user experience, save the time checking ODR and keep consistent
behavior with MSVC, we disabled the ODR check for the declarations in the global module
fragment by default. Users who want more strict check can still use the
``-Xclang -fno-skip-odr-check-in-gmf`` flag to get the ODR check enabled. It is also
encouraged to report issues if users find false positive ODR violations or false negative ODR
violations with the flag enabled.

ABI Impacts
-----------

The declarations in a module unit which are not in the global module fragment have new linkage names.

For example,

.. code-block:: c++

  export module M;
  namespace NS {
    export int foo();
  }

The linkage name of ``NS::foo()`` would be ``_ZN2NSW1M3fooEv``.
This couldn't be demangled by previous versions of the debugger or demangler.
As of LLVM 15.x, users can utilize ``llvm-cxxfilt`` to demangle this:

.. code-block:: console

  $ llvm-cxxfilt _ZN2NSW1M3fooEv

The result would be ``NS::foo@M()``, which reads as ``NS::foo()`` in module ``M``.

The ABI implies that we can't declare something in a module unit and define it in a non-module unit (or vice-versa),
as this would result in linking errors.

If we still want to implement declarations within the compatible ABI in module unit,
we can use the language-linkage specifier. Since the declarations in the language-linkage specifier
is attached to the global module fragments. For example:

.. code-block:: c++

  export module M;
  namespace NS {
    export extern "C++" int foo();
  }

Now the linkage name of ``NS::foo()`` will be ``_ZN2NS3fooEv``.

Performance Tips
----------------

Reduce duplications
~~~~~~~~~~~~~~~~~~~

While it is legal to have duplicated declarations in the global module fragments
of different module units, it is not free for clang to deal with the duplicated
declarations. In other word, for a translation unit, it will compile slower if the
translation unit itself and its importing module units contains a lot duplicated
declarations.

For example,

.. code-block:: c++

  // M-partA.cppm
  module;
  #include "big.header.h"
  export module M:partA;
  ...

  // M-partB.cppm
  module;
  #include "big.header.h"
  export module M:partB;
  ...

  // other partitions
  ...

  // M-partZ.cppm
  module;
  #include "big.header.h"
  export module M:partZ;
  ...

  // M.cppm
  export module M;
  export import :partA;
  export import :partB;
  ...
  export import :partZ;

  // use.cpp
  import M;
  ... // use declarations from module M.

When ``big.header.h`` is big enough and there are a lot of partitions,
the compilation of ``use.cpp`` may be slower than
the following style significantly:

.. code-block:: c++

  module;
  #include "big.header.h"
  export module m:big.header.wrapper;
  export ... // export the needed declarations

  // M-partA.cppm
  export module M:partA;
  import :big.header.wrapper;
  ...

  // M-partB.cppm
  export module M:partB;
  import :big.header.wrapper;
  ...

  // other partitions
  ...

  // M-partZ.cppm
  export module M:partZ;
  import :big.header.wrapper;
  ...

  // M.cppm
  export module M;
  export import :partA;
  export import :partB;
  ...
  export import :partZ;

  // use.cpp
  import M;
  ... // use declarations from module M.

The key part of the tip is to reduce the duplications from the text includes.

Known Problems
--------------

The following describes issues in the current implementation of modules.
Please see https://github.com/llvm/llvm-project/labels/clang%3Amodules for more issues
or file a new issue if you don't find an existing one.
If you're going to create a new issue for standard C++ modules,
please start the title with ``[C++20] [Modules]`` (or ``[C++23] [Modules]``, etc)
and add the label ``clang:modules`` (if you have permissions for that).

For higher level support for proposals, you could visit https://clang.llvm.org/cxx_status.html.

Including headers after import is problematic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, the following example can be accept:

.. code-block:: c++

  #include <iostream>
  import foo; // assume module 'foo' contain the declarations from `<iostream>`

  int main(int argc, char *argv[])
  {
      std::cout << "Test\n";
      return 0;
  }

but it will get rejected if we reverse the order of ``#include <iostream>`` and
``import foo;``:

.. code-block:: c++

  import foo; // assume module 'foo' contain the declarations from `<iostream>`
  #include <iostream>

  int main(int argc, char *argv[])
  {
      std::cout << "Test\n";
      return 0;
  }

Both of the above examples should be accepted.

This is a limitation in the implementation. In the first example,
the compiler will see and parse <iostream> first then the compiler will see the import.
So the ODR Checking and declarations merging will happen in the deserializer.
In the second example, the compiler will see the import first and the include second.
As a result, the ODR Checking and declarations merging will happen in the semantic analyzer.

So there is divergence in the implementation path. It might be understandable that why
the orders matter here in the case.
(Note that "understandable" is different from "makes sense").

This is tracked in: https://github.com/llvm/llvm-project/issues/61465

Ignored PreferredName Attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Due to a tricky problem, when Clang writes BMIs, Clang will ignore the ``preferred_name`` attribute, if any.
This implies that the ``preferred_name`` wouldn't show in debugger or dumping.

This is tracked in: https://github.com/llvm/llvm-project/issues/56490

Don't emit macros about module declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is covered by P1857R3. We mention it again here since users may abuse it before we implement it.

Someone may want to write code which could be compiled both by modules or non-modules.
A direct idea would be use macros like:

.. code-block:: c++

  MODULE
  IMPORT header_name
  EXPORT_MODULE MODULE_NAME;
  IMPORT header_name
  EXPORT ...

So this file could be triggered like a module unit or a non-module unit depending on the definition
of some macros.
However, this kind of usage is forbidden by P1857R3 but we haven't implemented P1857R3 yet.
This means that is possible to write illegal modules code now, and obviously this will stop working
once P1857R3 is implemented.
A simple suggestion would be "Don't play macro tricks with module declarations".

This is tracked in: https://github.com/llvm/llvm-project/issues/56917

In consistent filename suffix requirement for importable module units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, clang requires the file name of an ``importable module unit`` should end with ``.cppm``
(or ``.ccm``, ``.cxxm``, ``.c++m``). However, the behavior is inconsistent with other compilers.

This is tracked in: https://github.com/llvm/llvm-project/issues/57416

clang-cl is not compatible with the standard C++ modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can't use the `/clang:-fmodule-file` or `/clang:-fprebuilt-module-path` to specify
the BMI within ``clang-cl.exe``.

This is tracked in: https://github.com/llvm/llvm-project/issues/64118

false positive ODR violation diagnostic due to using inconsistent qualified but the same type
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ODR violation is a pretty common issue when using modules.
Sometimes the program violated the One Definition Rule actually.
But sometimes it shows the compiler gives false positive diagnostics.

One often reported example is:

.. code-block:: c++

  // part.cc
  module;
  typedef long T;
  namespace ns {
  inline void fun() {
      (void)(T)0;
  }
  }
  export module repro:part;

  // repro.cc
  module;
  typedef long T;
  namespace ns {
      using ::T;
  }
  namespace ns {
  inline void fun() {
      (void)(T)0;
  }
  }
  export module repro;
  export import :part;

Currently the compiler complains about the inconsistent definition of `fun()` in
2 module units. This is incorrect. Since both definitions of `fun()` has the same
spelling and `T` refers to the same type entity finally. So the program should be
fine.

This is tracked in https://github.com/llvm/llvm-project/issues/78850.

Using TU-local entity in other units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Module units are translation units. So the entities which should only be local to the
module unit itself shouldn't be used by other units in any means.

In the language side, to address the idea formally, the language specification defines
the concept of ``TU-local`` and ``exposure`` in
`basic.link/p14 <https://eel.is/c++draft/basic.link#14>`_,
`basic.link/p15 <https://eel.is/c++draft/basic.link#15>`_,
`basic.link/p16 <https://eel.is/c++draft/basic.link#16>`_,
`basic.link/p17 <https://eel.is/c++draft/basic.link#17>`_ and
`basic.link/p18 <https://eel.is/c++draft/basic.link#18>`_.

However, the compiler doesn't support these 2 ideas formally.
This results in unclear and confusing diagnostic messages.
And it is worse that the compiler may import TU-local entities to other units without any
diagnostics.

This is tracked in https://github.com/llvm/llvm-project/issues/78173.

Header Units
============

How to build projects using header unit
---------------------------------------

.. warning::

   The user interfaces of header units is highly experimental. There are still
   many unanswered question about how tools should interact with header units.
   The user interfaces described here may change after we have progress on how
   tools should support for header units.

Quick Start
~~~~~~~~~~~

For the following example,

.. code-block:: c++

  import <iostream>;
  int main() {
    std::cout << "Hello World.\n";
  }

we could compile it as

.. code-block:: console

  $ clang++ -std=c++20 -xc++-system-header --precompile iostream -o iostream.pcm
  $ clang++ -std=c++20 -fmodule-file=iostream.pcm main.cpp

How to produce BMIs
~~~~~~~~~~~~~~~~~~~

Similar to named modules, we could use ``--precompile`` to produce the BMI.
But we need to specify that the input file is a header by ``-xc++-system-header`` or ``-xc++-user-header``.

Also we could use `-fmodule-header={user,system}` option to produce the BMI for header units
which has suffix like `.h` or `.hh`.
The value of `-fmodule-header` means the user search path or the system search path.
The default value for `-fmodule-header` is `user`.
For example,

.. code-block:: c++

  // foo.h
  #include <iostream>
  void Hello() {
    std::cout << "Hello World.\n";
  }

  // use.cpp
  import "foo.h";
  int main() {
    Hello();
  }

We could compile it as:

.. code-block:: console

  $ clang++ -std=c++20 -fmodule-header foo.h -o foo.pcm
  $ clang++ -std=c++20 -fmodule-file=foo.pcm use.cpp

For headers which don't have a suffix, we need to pass ``-xc++-header``
(or ``-xc++-system-header`` or ``-xc++-user-header``) to mark it as a header.
For example,

.. code-block:: c++

  // use.cpp
  import "foo.h";
  int main() {
    Hello();
  }

.. code-block:: console

  $ clang++ -std=c++20 -fmodule-header=system -xc++-header iostream -o iostream.pcm
  $ clang++ -std=c++20 -fmodule-file=iostream.pcm use.cpp

How to specify the dependent BMIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We could use ``-fmodule-file`` to specify the BMIs, and this option may occur multiple times as well.

With the existing implementation ``-fprebuilt-module-path`` cannot be used for header units
(since they are nominally anonymous).
For header units, use  ``-fmodule-file`` to include the relevant PCM file for each header unit.

This is expect to be solved in future editions of the compiler either by the tooling finding and specifying
the -fmodule-file or by the use of a module-mapper that understands how to map the header name to their PCMs.

Don't compile the BMI
~~~~~~~~~~~~~~~~~~~~~

Another difference with modules is that we can't compile the BMI from a header unit.
For example:

.. code-block:: console

  $ clang++ -std=c++20 -xc++-system-header --precompile iostream -o iostream.pcm
  # This is not allowed!
  $ clang++ iostream.pcm -c -o iostream.o

It makes sense due to the semantics of header units, which are just like headers.

Include translation
~~~~~~~~~~~~~~~~~~~

The C++ spec allows the vendors to convert ``#include header-name`` to ``import header-name;`` when possible.
Currently, Clang would do this translation for the ``#include`` in the global module fragment.

For example, the following two examples are the same:

.. code-block:: c++

  module;
  import <iostream>;
  export module M;
  export void Hello() {
    std::cout << "Hello.\n";
  }

with the following one:

.. code-block:: c++

  module;
  #include <iostream>
  export module M;
  export void Hello() {
      std::cout << "Hello.\n";
  }

.. code-block:: console

  $ clang++ -std=c++20 -xc++-system-header --precompile iostream -o iostream.pcm
  $ clang++ -std=c++20 -fmodule-file=iostream.pcm --precompile M.cppm -o M.cpp

In the latter example, the Clang could find the BMI for the ``<iostream>``
so it would try to replace the ``#include <iostream>`` to ``import <iostream>;`` automatically.


Relationships between Clang modules
-----------------------------------

Header units have pretty similar semantics with Clang modules.
The semantics of both of them are like headers.

In fact, we could even "mimic" the sytle of header units by Clang modules:

.. code-block:: c++

  module "iostream" {
    export *
    header "/path/to/libstdcxx/iostream"
  }

.. code-block:: console

  $ clang++ -std=c++20 -fimplicit-modules -fmodule-map-file=.modulemap main.cpp

It would be simpler if we are using libcxx:

.. code-block:: console

  $ clang++ -std=c++20 main.cpp -fimplicit-modules -fimplicit-module-maps

Since there is already one
`module map <https://github.com/llvm/llvm-project/blob/main/libcxx/include/module.modulemap.in>`_
in the source of libcxx.

Then immediately leads to the question: why don't we implement header units through Clang header modules?

The main reason for this is that Clang modules have more semantics like hierarchy or
wrapping multiple headers together as a big module.
However, these things are not part of Standard C++ Header units,
and we want to avoid the impression that these additional semantics get interpreted as Standard C++ behavior.

Another reason is that there are proposals to introduce module mappers to the C++ standard
(for example, https://wg21.link/p1184r2).
If we decide to reuse Clang's modulemap, we may get in trouble once we need to introduce another module mapper.

So the final answer for why we don't reuse the interface of Clang modules for header units is that
there are some differences between header units and Clang modules and that ignoring those
differences now would likely become a problem in the future.

Discover Dependencies
=====================

Prior to modules, all the translation units can be compiled parallelly.
But it is not true for the module units. The presence of module units requires
us to compile the translation units in a (topological) order.

The clang-scan-deps scanner implemented
`P1689 paper <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p1689r5.html>`_
to describe the order. Only named modules are supported now.

We need a compilation database to use clang-scan-deps. See
`JSON Compilation Database Format Specification <JSONCompilationDatabase.html>`_
for example. Note that the ``output`` entry is necessary for clang-scan-deps
to scan P1689 format. Here is an example:

.. code-block:: c++

  //--- M.cppm
  export module M;
  export import :interface_part;
  import :impl_part;
  export int Hello();

  //--- interface_part.cppm
  export module M:interface_part;
  export void World();

  //--- Impl.cpp
  module;
  #include <iostream>
  module M;
  void Hello() {
      std::cout << "Hello ";
  }

  //--- impl_part.cppm
  module;
  #include <string>
  #include <iostream>
  module M:impl_part;
  import :interface_part;

  std::string W = "World.";
  void World() {
      std::cout << W << std::endl;
  }

  //--- User.cpp
  import M;
  import third_party_module;
  int main() {
    Hello();
    World();
    return 0;
  }

And here is the compilation database:

.. code-block:: text

  [
  {
      "directory": ".",
      "command": "<path-to-compiler-executable>/clang++ -std=c++20 M.cppm -c -o M.o",
      "file": "M.cppm",
      "output": "M.o"
  },
  {
      "directory": ".",
      "command": "<path-to-compiler-executable>/clang++ -std=c++20 Impl.cpp -c -o Impl.o",
      "file": "Impl.cpp",
      "output": "Impl.o"
  },
  {
      "directory": ".",
      "command": "<path-to-compiler-executable>/clang++ -std=c++20 impl_part.cppm -c -o impl_part.o",
      "file": "impl_part.cppm",
      "output": "impl_part.o"
  },
  {
      "directory": ".",
      "command": "<path-to-compiler-executable>/clang++ -std=c++20 interface_part.cppm -c -o interface_part.o",
      "file": "interface_part.cppm",
      "output": "interface_part.o"
  },
  {
      "directory": ".",
      "command": "<path-to-compiler-executable>/clang++ -std=c++20 User.cpp -c -o User.o",
      "file": "User.cpp",
      "output": "User.o"
  }
  ]

And we can get the dependency information in P1689 format by:

.. code-block:: console

  $ clang-scan-deps -format=p1689 -compilation-database P1689.json

And we will get:

.. code-block:: text

  {
    "revision": 0,
    "rules": [
      {
        "primary-output": "Impl.o",
        "requires": [
          {
            "logical-name": "M",
            "source-path": "M.cppm"
          }
        ]
      },
      {
        "primary-output": "M.o",
        "provides": [
          {
            "is-interface": true,
            "logical-name": "M",
            "source-path": "M.cppm"
          }
        ],
        "requires": [
          {
            "logical-name": "M:interface_part",
            "source-path": "interface_part.cppm"
          },
          {
            "logical-name": "M:impl_part",
            "source-path": "impl_part.cppm"
          }
        ]
      },
      {
        "primary-output": "User.o",
        "requires": [
          {
            "logical-name": "M",
            "source-path": "M.cppm"
          },
          {
            "logical-name": "third_party_module"
          }
        ]
      },
      {
        "primary-output": "impl_part.o",
        "provides": [
          {
            "is-interface": false,
            "logical-name": "M:impl_part",
            "source-path": "impl_part.cppm"
          }
        ],
        "requires": [
          {
            "logical-name": "M:interface_part",
            "source-path": "interface_part.cppm"
          }
        ]
      },
      {
        "primary-output": "interface_part.o",
        "provides": [
          {
            "is-interface": true,
            "logical-name": "M:interface_part",
            "source-path": "interface_part.cppm"
          }
        ]
      }
    ],
    "version": 1
  }

See the P1689 paper for the meaning of the fields.

And if the user want a finer-grained control for any reason, e.g., to scan the generated source files,
the user can choose to get the dependency information per file. For example:

.. code-block:: console

  $ clang-scan-deps -format=p1689 -- <path-to-compiler-executable>/clang++ -std=c++20 impl_part.cppm -c -o impl_part.o

And we'll get:

.. code-block:: text

  {
    "revision": 0,
    "rules": [
      {
        "primary-output": "impl_part.o",
        "provides": [
          {
            "is-interface": false,
            "logical-name": "M:impl_part",
            "source-path": "impl_part.cppm"
          }
        ],
        "requires": [
          {
            "logical-name": "M:interface_part"
          }
        ]
      }
    ],
    "version": 1
  }

In this way, we can pass the single command line options after the ``--``.
Then clang-scan-deps will extract the necessary information from the options.
Note that we need to specify the path to the compiler executable instead of saying
``clang++`` simply.

The users may want the scanner to get the transitional dependency information for headers.
Otherwise, the users have to scan twice for the project, once for headers and once for modules.
To address the requirement, clang-scan-deps will recognize the specified preprocessor options
in the given command line and generate the corresponding dependency information. For example,

.. code-block:: console

  $ clang-scan-deps -format=p1689 -- ../bin/clang++ -std=c++20 impl_part.cppm -c -o impl_part.o -MD -MT impl_part.ddi -MF impl_part.dep
  $ cat impl_part.dep

We will get:

.. code-block:: text

  impl_part.ddi: \
    /usr/include/bits/wchar.h /usr/include/bits/types/wint_t.h \
    /usr/include/bits/types/mbstate_t.h \
    /usr/include/bits/types/__mbstate_t.h /usr/include/bits/types/__FILE.h \
    /usr/include/bits/types/FILE.h /usr/include/bits/types/locale_t.h \
    /usr/include/bits/types/__locale_t.h \
    ...

When clang-scan-deps detects ``-MF`` option, clang-scan-deps will try to write the
dependency information for headers to the file specified by ``-MF``.

Possible Issues: Failed to find system headers
----------------------------------------------

In case the users encounter errors like ``fatal error: 'stddef.h' file not found``,
probably the specified ``<path-to-compiler-executable>/clang++`` refers to a symlink
instead a real binary. There are 4 potential solutions to the problem:

* (1) End users can resolve the issue by pointing the specified compiler executable to
  the real binary instead of the symlink.
* (2) End users can invoke ``<path-to-compiler-executable>/clang++ -print-resource-dir``
  to get the corresponding resource directory for your compiler and add that directory
  to the include search paths manually in the build scripts.
* (3) Build systems that use a compilation database as the input for clang-scan-deps
  scanner, the build system can add the flag ``--resource-dir-recipe invoke-compiler`` to
  the clang-scan-deps scanner to calculate the resources directory dynamically.
  The calculation happens only once for a unique ``<path-to-compiler-executable>/clang++``.
* (4) For build systems that invokes the clang-scan-deps scanner per file, repeatedly
  calculating the resource directory may be inefficient. In such cases, the build
  system can cache the resource directory by itself and pass ``-resource-dir <resource-dir>``
  explicitly in the command line options:

.. code-block:: console

  $ clang-scan-deps -format=p1689 -- <path-to-compiler-executable>/clang++ -std=c++20 -resource-dir <resource-dir> mod.cppm -c -o mod.o


Possible Questions
==================

How modules speed up compilation
--------------------------------

A classic theory for the reason why modules speed up the compilation is:
if there are ``n`` headers and ``m`` source files and each header is included by each source file,
then the complexity of the compilation is ``O(n*m)``;
But if there are ``n`` module interfaces and ``m`` source files, the complexity of the compilation is
``O(n+m)``. So, using modules would be a big win when scaling.
In a simpler word, we could get rid of many redundant compilations by using modules.

Roughly, this theory is correct. But the problem is that it is too rough.
The behavior depends on the optimization level, as we will illustrate below.

First is ``O0``. The compilation process is described in the following graph.

.. code-block:: none

  ├-------------frontend----------┼-------------middle end----------------┼----backend----┤
  │                               │                                       │               │
  └---parsing----sema----codegen--┴----- transformations ---- codegen ----┴---- codegen --┘

  ┌---------------------------------------------------------------------------------------┐
  |                                                                                       │
  |                                     source file                                       │
  |                                                                                       │
  └---------------------------------------------------------------------------------------┘

              ┌--------┐
              │        │
              │imported│
              │        │
              │  code  │
              │        │
              └--------┘

Here we can see that the source file (could be a non-module unit or a module unit) would get processed by the
whole pipeline.
But the imported code would only get involved in semantic analysis, which is mainly about name lookup,
overload resolution and template instantiation.
All of these processes are fast relative to the whole compilation process.
More importantly, the imported code only needs to be processed once in frontend code generation,
as well as the whole middle end and backend.
So we could get a big win for the compilation time in O0.

But with optimizations, things are different:

(we omit ``code generation`` part for each end due to the limited space)

.. code-block:: none

  ├-------- frontend ---------┼--------------- middle end --------------------┼------ backend ----┤
  │                           │                                               │                   │
  └--- parsing ---- sema -----┴--- optimizations --- IPO ---- optimizations---┴--- optimizations -┘

  ┌-----------------------------------------------------------------------------------------------┐
  │                                                                                               │
  │                                         source file                                           │
  │                                                                                               │
  └-----------------------------------------------------------------------------------------------┘
                ┌---------------------------------------┐
                │                                       │
                │                                       │
                │            imported code              │
                │                                       │
                │                                       │
                └---------------------------------------┘

It would be very unfortunate if we end up with worse performance after using modules.
The main concern is that when we compile a source file, the compiler needs to see the function body
of imported module units so that it can perform IPO (InterProcedural Optimization, primarily inlining
in practice) to optimize functions in current source file with the help of the information provided by
the imported module units.
In other words, the imported code would be processed again and again in importee units
by optimizations (including IPO itself).
The optimizations before IPO and the IPO itself are the most time-consuming part in whole compilation process.
So from this perspective, we might not be able to get the improvements described in the theory.
But we could still save the time for optimizations after IPO and the whole backend.

Overall, at ``O0`` the implementations of functions defined in a module will not impact module users,
but at higher optimization levels the definitions of such functions are provided to user compilations for the
purposes of optimization (but definitions of these functions are still not included in the use's object file)-
this means the build speedup at higher optimization levels may be lower than expected given ``O0`` experience,
but does provide by more optimization opportunities.

Interoperability with Clang Modules
-----------------------------------

We **wish** to support clang modules and standard c++ modules at the same time,
but the mixed using form is not well used/tested yet.

Please file new github issues as you find interoperability problems.
