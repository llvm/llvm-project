====================
Clang-Reorder-Fields
====================

.. contents::

.. toctree::
  :maxdepth: 1

:program:`clang-reorder-fields` is a refactoring tool to reorder fields in
C/C++ structs and classes. This tool automatically updates:

- Field declarations in the record definition
- Constructor initializer lists in C++ classes
- Aggregate initialization expressions (both C and C++)
- Designated initializer lists (C++20)

This can be useful for optimizing memory layout, improving cache performance,
or conforming to coding standards that require specific field orderings.

Example usage
-------------

Basic struct reordering
~~~~~~~~~~~~~~~~~~~~~~~

Consider this simple struct in `example.c`:

.. code-block:: c

  struct Foo {
    const int *x;
    int y;
    double z;
    int w;
  };

  int main() {
    const int val = 42;
    struct Foo foo = { &val, 0, 1.5, 17 };
    return 0;
  }

To reorder the fields to `z, w, y, x`, run:

.. code-block:: console

  clang-reorder-fields -record-name Foo -fields-order z,w,y,x example.c --

This will reorder both the struct definition and the initialization:

.. code-block:: c

  struct Foo {
    double z;
    int w;
    int y;
    const int *x;
  };

  int main() {
    const int val = 42;
    struct Foo foo = { 1.5, 17, 0, &val };
    return 0;
  }

Namespaced structs
~~~~~~~~~~~~~~~~~~

For C++ code with namespaces, use the fully-qualified name:

.. code-block:: c++

  namespace bar {
  struct Foo {
    const int *x;
    int y;
    double z;
    int w;
  };
  }

.. code-block:: console

  clang-reorder-fields -record-name ::bar::Foo -fields-order z,w,y,x example.cpp --

For classes defined in the global namespace (without any namespace), you can
use either the simple class name or prefix it with `::`:

.. code-block:: console

  clang-reorder-fields -record-name Foo -fields-order z,w,y,x example.cpp --
  # or
  clang-reorder-fields -record-name ::Foo -fields-order z,w,y,x example.cpp --

C++ constructor initializer lists
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tool also reorders constructor initializer lists. Given:

.. code-block:: c++

  class Foo {
  public:
    Foo();

  private:
    int x;
    const char *s1;
    const char *s2;
    double z;
  };

  Foo::Foo():
    x(12),
    s1("abc"),
    s2("def"),
    z(3.14)
  {}

Running:

.. code-block:: console

  clang-reorder-fields -record-name Foo -fields-order s1,x,z,s2 example.cpp --

Will reorder both the field declarations and the constructor initializers:

.. code-block:: c++

  class Foo {
  public:
    Foo();

  private:
    const char *s1;
    int x;
    double z;
    const char *s2;
  };

  Foo::Foo():
    s1("abc"),
    x(12),
    z(3.14),
    s2("def")
  {}

Designated initializers
~~~~~~~~~~~~~~~~~~~~~~~

For C++20 code using designated initializers:

.. code-block:: c++

  struct Bar {
    char a;
    int b;
    int c;
  };

  int main() {
    Bar bar1 = { 'a', 0, 123 };
    Bar bar2 = { .a = 'a', .b = 0, .c = 123 };
    return 0;
  }

.. code-block:: console

  clang-reorder-fields --extra-arg="-std=c++20" -record-name Bar \
    -fields-order c,a,b example.cpp --

Will produce:

.. code-block:: c++

  struct Bar {
    int c;
    char a;
    int b;
  };

  int main() {
    Bar bar1 = { 123, 'a', 0 };
    Bar bar2 = { .c = 123, .a = 'a', .b = 0 };
    return 0;
  }

In-place editing
~~~~~~~~~~~~~~~~

Use the `-i` flag to modify files in-place:

.. code-block:: console

  clang-reorder-fields -record-name Foo -fields-order z,w,y,x -i example.c --

Limitations and Caveats
-----------------------

Different access specifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tool cannot reorder fields with different access specifiers
(``public/private/protected``). All fields being reordered must have the same
access level.

.. code-block:: c++

  class Example {
  private:
    int x;
  public:
    int y;  // Cannot reorder x and y - different access levels
  };

Multiple field declarations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Declarations with multiple fields in one statement are not supported:

.. code-block:: c

  struct Example {
    int a, b;  // Not supported - multiple fields in one declaration
  };

Macro-expanded fields
~~~~~~~~~~~~~~~~~~~~~

Macros that expand to multiple field declarations are not supported. However,
macros that expand to a single field declaration work correctly:

.. code-block:: c

  #define INT_FIELD(NAME) int NAME     // Supported - expands to one field
  #define TWO_FIELDS int a; int b;     // Not supported - expands to two fields

  struct Supported {
    INT_FIELD(x);  // OK - this is a single field
    int y;
    INT_FIELD(z);  // OK - this is a single field
  };

  struct NotSupported {
    TWO_FIELDS     // Not OK - expands to multiple fields
    int c;
  };

The tool can reorder fields declared via macros as long as each macro invocation
expands to exactly one field declaration.

Preprocessor directives
~~~~~~~~~~~~~~~~~~~~~~~

Structs with preprocessor directives between fields cannot be reordered:

.. code-block:: c

  struct Example {
    int a;
  #ifdef FEATURE
    int b;
  #endif
    int c;  // Not supported - preprocessor directives present
  };

Flexible array members
~~~~~~~~~~~~~~~~~~~~~~

In C, a flexible array member is an incomplete array type that must be the last
member of a struct (as specified by C99 and later standards). This allows the
struct to have a variable-length array at the end. Since this is a language
requirement, the tool enforces that flexible array members remain in the last
position:

.. code-block:: c

  struct Example {
    int count;
    int data[];  // Flexible array member - must remain last
  };

Attempting to reorder fields such that the flexible array member is no longer
last will result in an error:

.. code-block:: console

  clang-reorder-fields -record-name Example -fields-order data,count example.c --

Will produce:

.. code-block:: text

  Flexible array member must remain the last field in the struct

This ensures the generated code remains valid C.

Field dependencies in initializers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tool will issue a warning if reordering causes a field to be used in an
initializer before it's initialized. Consider this example:

.. code-block:: c++

  class Foo {
  public:
    Foo(int x, char c);
    int x;
    char c;
    Dummy z;
  };

  Foo::Foo(int x, char c) :
    x(x),
    c(c),
    z(this->x, c)  // z's initializer uses x and c
  {}

If you reorder the fields to `z, c, x`:

.. code-block:: console

  clang-reorder-fields -record-name Foo -fields-order z,c,x example.cpp --

The tool will produce warnings:

.. code-block:: text

  example.cpp:10:3: warning: reordering field x after z makes x uninitialized when used in init expression
  example.cpp:10:3: warning: reordering field c after z makes c uninitialized when used in init expression

This warns you that in C++, member initializers are executed in the order that
fields are declared in the class, not the order they appear in the initializer
list. After reordering, `z` would be initialized first, but its initializer
tries to use `x` and `c` which haven't been initialized yet.

The tool will still perform the reordering but warns about the potential issue.
You should review these warnings and adjust your code accordingly.

:program:`clang-reorder-fields` Command Line Options
----------------------------------------------------

.. option:: --record-name=<string>

  The fully-qualified name of the struct or class to reorder. Required.

  For C structs, use the struct name directly (e.g., `Foo`).

  For C++ classes/structs in namespaces, use the fully-qualified name including
  namespaces (e.g., `::namespace::ClassName`).

  For C++ classes/structs in the global namespace, you can use either the simple
  name (e.g., `Foo`) or prefix with `::` (e.g., `::Foo`).

.. option:: --fields-order=<string>

  Comma-separated list of field names in the desired order. Required.

  All field names must exactly match the fields in the struct/class definition.
  The number of fields must match the number in the definition.

.. option:: -i

  Overwrite edited files in-place. If not specified, the rewritten code is
  printed to stdout.

.. option:: --extra-arg=<string>

  Additional argument to append to the compiler command line.

  Useful for specifying language standards (e.g., `--extra-arg="-std=c++20"`).

.. option:: --extra-arg-before=<string>

  Additional argument to prepend to the compiler command line.

.. option:: -p <string>

  Build path. Specifies the directory containing `compile_commands.json` for
  compilation database support.

Use Cases
---------

Memory layout optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Reorder fields to minimize padding and improve cache locality:

.. code-block:: c

  // Before: 24 bytes (with padding)
  struct Data {
    char a;     // 1 byte + 7 padding
    double b;   // 8 bytes
    char c;     // 1 byte + 7 padding
  };

.. code-block:: console

  clang-reorder-fields -record-name Data -fields-order b,a,c data.c --

.. code-block:: c

  // After: 16 bytes (less padding)
  struct Data {
    double b;   // 8 bytes
    char a;     // 1 byte
    char c;     // 1 byte + 6 padding
  };

Coding standard compliance
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure fields are ordered according to project conventions (e.g., alphabetically,
by type, or by access pattern).

Field grouping
~~~~~~~~~~~~~~

Group related fields together for better code organization and readability.
