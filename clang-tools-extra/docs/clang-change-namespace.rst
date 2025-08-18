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

Let's move everything from the namespace `a` to the global namespace
(`--new_namespace ""` means global namespace):

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
