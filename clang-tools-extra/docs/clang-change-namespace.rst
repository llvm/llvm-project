======================
Clang-Change-Namespace
======================

.. contents::

.. toctree::
  :maxdepth: 1

:program:`clang-change-namespace` can be used to change the surrounding
namespaces of class/function definitions.

Classes/functions in the moved namespace will have new namespaces while
references to symbols (e.g. types, functions) which are not defined in the
changed namespace will be correctly qualified by prepending namespace specifiers
before them. This will try to add shortest namespace specifiers possible.

When a symbol reference needs to be fully-qualified, this adds a `::` prefix to
the namespace specifiers unless the new namespace is the global namespace. For
classes, only classes that are declared/defined in the given namespace in
specified files will be moved: forward declarations will remain in the old
namespace. The will be demonstrated in the next example.

Example usage
-------------

For example, consider this `test.cc` example here with the forward declared
class `FWD` and the defined class `A`, both in the namespace `a`.

.. code-block:: c++

  namespace a {
  class FWD;
  class A {
    FWD *fwd;
  };
  } // namespace a

And now let's change the namespace `a` to `x`.

.. code-block:: console

  clang-change-namespace \
    --old_namespace "a" \
    --new_namespace "x" \
    --file_pattern "test.cc" \
    --i \
    test.cc

Note that in the code below there's still the forward decalred class `FWD` that
stayed in the namespace `a`. It wasn't moved to the new namespace because it
wasn't defined/declared here in `a` but only forward declared.

.. code-block:: c++

  namespace a {
  class FWD;
  } // namespace a
  namespace x {

  class A {
    a::FWD *fwd;
  };
  } // namespace x


Another example
---------------

Consider this `test.cc` file:

.. code-block:: c++

  namespace na {
  class X {};
  namespace nb {
  class Y {
    X x;
  };
  } // namespace nb
  } // namespace na

To move the definition of class `Y` from namespace `na::nb` to `x::y`, run:

.. code-block:: console

  clang-change-namespace \
    --old_namespace "na::nb" \
    --new_namespace "x::y" \
    --file_pattern "test.cc" \
    --i \
    test.cc

This will overwrite `test.cc` to look like this:

.. code-block:: c++

  namespace na {
  class X {};

  } // namespace na
  namespace x {
  namespace y {
  class Y {
    na::X x;
  };
  } // namespace y
  } // namespace x

Note, that we've successfully moved the class `Y` from namespace `na::nb` to
namespace `x::y`.

Caveats
=======

Content already exists in new namespace
---------------------------------------

Consider this `test.cc` example that defines two `class A` one inside the
namespace `a` and one in namespace `b`:

.. code-block:: c++

  namespace a {
  class A {
      int classAFromWithinNamespace_a;
  };
  } // namespace a

  namespace b {
  class A {
      int classAFromWithinNamespace_b;
  };
  } //namespace b

Let's move everything from namespace `a` to namespace `b`:

.. code-block:: console

  clang-change-namespace \
    --old_namespace "a" \
    --new_namespace "b" \
    --file_pattern test.cc \
    test.cc

As expected we now have to definitions of `class A` inside the namespace `b`:

.. code-block:: c++

  namespace b {
  class A {
    int classAFromWithinNamespace_a;
  };
  } // namespace b

  namespace b {
  class A {
      int classAFromWithinNamespace_b;
  };
  } //namespace b

The re-factoring looks correct but the code will not compile due to the name
duplication. It is not up to the tool to ensure compilability in that sense.
But one has to be aware of that.

Inline namespace doesn't work
-----------------------------

Consider this usage of two versions of implementations for a `greet` function:

.. code-block:: c++

  #include <cstdio>

  namespace Greeter {
  inline namespace Version1 {
    const char* greet() { return "Hello from version 1!"; }
  } // namespace Version1
  namespace Version2 {
    const char* greet() { return "Hello from version 2!"; }
  } // namespace Version2
  } // namespace Greeter

  int main(int argc, char* argv[]) {
    printf("%s\n", Greeter::greet());
    return 0;
  }

Note, that currently `Greeter::greet()` will result in a call to
`Greeter::Version1::greet()` because that's the inlined namespace.

Let's say you want to move one and make `Version2` the default now and remove
the `inline` from the `Version1`. First let's try to turn `namespace Version2`
into `inline namespace Version2`:

.. code-block:: console

  clang-change-namespace \
    --old_namespace "Greeter::Version2" \
    --new_namespace "inline Version2" \
    --file_pattern main.cc main.cc

But this will put the `inline` keyword in the wrong place resulting in:

.. code-block:: c++

  #include <cstdio>

  namespace Greeter {
  inline namespace Version1 {
          const char* greet() { return "Hello from version 1!"; }
  } // namespace Version1

  } // namespace Greeter
  namespace inline Greeter {
  namespace Version2 {
  const char *greet() { return "Hello from version 2!"; }
  } // namespace Version2
  } // namespace inline Greeter

  int main(int argc, char* argv[]) {
          printf("%s\n", Greeter::greet());
          return 0;
  }

One cannot use `:program:`clang-change-namespace` to inline a namespace.

Symbol references not updated
-----------------------------

Consider this `test.cc` file:

.. code-block:: c++

  namespace old {
  struct foo {};
  }  // namespace old

  namespace b {
  old::foo g_foo;
  }  // namespace b

Notice that namespace `b` defines a global variable of type `old::foo`. If we
now change the name of the `old` namespace to `modern`, the reference will not
be updated:

.. code-block:: console

  clang-change-namespace \
    --old_namespace "old" \
    --new_namespace "modern" \
    --file_pattern test.cc \
    test.cc

.. code-block:: c++

  namespace modern {
  struct foo {};
  } // namespace modern

  namespace b {
  old::foo g_foo;
  }  // namespace b

`g_foo` is still of the no longer existing type `old::foo` while instead it
should use `modern::foo`.

Only symbol references in the moved namespace are updated, not outside of it.


:program:`clang-change-namespace` Command Line Options
======================================================

.. option:: --allowed_file=<string>

  A file containing regexes of symbol names that are not expected to be updated
  when changing namespaces around them.

.. option:: --dump_result

  Dump new file contents in YAML, if specified.

.. option:: --extra-arg=<string>

  Additional argument to append to the compiler command line

.. option:: --extra-arg-before=<string>

  Additional argument to prepend to the compiler command line

.. option:: --file_pattern=<string>

  Only rename namespaces in files that match the given regular expression
  pattern.

.. option:: -i

  Inplace edit <file>s, if specified.

.. option:: --new_namespace=<string>

  New namespace. Use `""` when you target the global namespace.

.. option:: --old_namespace=<string>

  Old namespace.

.. option:: -p <string>

  Build path

.. option:: --style=<string>

  The style name used for reformatting.
