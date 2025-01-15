.. title:: clang-tidy - misc-const-correctness

misc-const-correctness
======================

This check implements detection of local variables which could be declared as
``const`` but are not. Declaring variables as ``const`` is required or recommended by many
coding guidelines, such as:
`ES.25 <https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#es25-declare-an-object-const-or-constexpr-unless-you-want-to-modify-its-value-later-on>`_
from the C++ Core Guidelines.

Please note that this check's analysis is type-based only. Variables that are not modified
but used to create a non-const handle that might escape the scope are not diagnosed
as potential ``const``.

.. code-block:: c++

  // Declare a variable, which is not ``const`` ...
  int i = 42;
  // but use it as read-only. This means that `i` can be declared ``const``.
  int result = i * i;       // Before transformation
  int const result = i * i; // After transformation

The check can analyze values, pointers and references but not (yet) pointees:

.. code-block:: c++

  // Normal values like built-ins or objects.
  int potential_const_int = 42;       // Before transformation
  int const potential_const_int = 42; // After transformation
  int copy_of_value = potential_const_int;

  MyClass could_be_const;       // Before transformation
  MyClass const could_be_const; // After transformation
  could_be_const.const_qualified_method();

  // References can be declared const as well.
  int &reference_value = potential_const_int;       // Before transformation
  int const& reference_value = potential_const_int; // After transformation
  int another_copy = reference_value;

  // The similar semantics of pointers are not (yet) analyzed.
  int *pointer_variable = &potential_const_int; // _NO_ 'const int *pointer_variable' suggestion.
  int last_copy = *pointer_variable;

The automatic code transformation is only applied to variables that are declared in single
declarations. You may want to prepare your code base with
:doc:`readability-isolate-declaration <../readability/isolate-declaration>` first.

Note that there is the check
:doc:`cppcoreguidelines-avoid-non-const-global-variables <../cppcoreguidelines/avoid-non-const-global-variables>`
to enforce ``const`` correctness on all globals.

Known Limitations
-----------------

The check does not run on `C` code.

The check will not analyze templated variables or variables that are instantiation dependent.
Different instantiations can result in different ``const`` correctness properties and in general it
is not possible to find all instantiations of a template. The template might be used differently in
an independent translation unit.

Pointees can not be analyzed for constness yet. The following code shows this limitation.

.. code-block:: c++

  // Declare a variable that will not be modified.
  int constant_value = 42;

  // Declare a pointer to that variable, that does not modify either, but misses 'const'.
  // Could be 'const int *pointer_to_constant = &constant_value;'
  int *pointer_to_constant = &constant_value;

  // Usage:
  int result = 520 * 120 * (*pointer_to_constant);

This limitation affects the capability to add ``const`` to methods which is not possible, too.

Options
-------

.. option:: AnalyzeValues (default = true)

  Enable or disable the analysis of ordinary value variables, like ``int i = 42;``

  .. code-block:: c++

    // Warning
    int i = 42;
    // No warning
    int const i = 42;

    // Warning
    int a[] = {42, 42, 42};
    // No warning
    int const a[] = {42, 42, 42};

.. option:: AnalyzeReferences (default = true)

  Enable or disable the analysis of reference variables, like ``int &ref = i;``

  .. code-block:: c++

    int i = 42;
    // Warning
    int& ref = i;
    // No warning
    int const& ref = i;

.. option:: WarnPointersAsValues (default = false)

  This option enables the suggestion for ``const`` of the pointer itself.
  Pointer values have two possibilities to be ``const``, the pointer
  and the value pointing to.

  .. code-block:: c++

    int value = 42;

    // Warning
    const int * pointer_variable = &value;
    // No warning
    const int *const pointer_variable = &value;

.. option:: TransformValues (default = true)

  Provides fixit-hints for value types that automatically add ``const`` if its a single declaration.

  .. code-block:: c++

    // Before
    int value = 42;
    // After
    int const value = 42;

    // Before
    int a[] = {42, 42, 42};
    // After
    int const a[] = {42, 42, 42};

    // Result is modified later in its life-time. No diagnostic and fixit hint will be emitted.
    int result = value * 3;
    result -= 10;

.. option:: TransformReferences (default = true)

  Provides fixit-hints for reference types that automatically add ``const`` if its a single
  declaration.

  .. code-block:: c++

    // This variable could still be a constant. But because there is a non-const reference to
    // it, it can not be transformed (yet).
    int value = 42;
    // The reference 'ref_value' is not modified and can be made 'const int &ref_value = value;'
    // Before
    int &ref_value = value;
    // After
    int const &ref_value = value;

    // Result is modified later in its life-time. No diagnostic and fixit hint will be emitted.
    int result = ref_value * 3;
    result -= 10;

.. option:: TransformPointersAsValues (default = false)

  Provides fixit-hints for pointers if their pointee is not changed. This does not analyze if the
  value-pointed-to is unchanged!

  Requires 'WarnPointersAsValues' to be 'true'.

  .. code-block:: c++

    int value = 42;

    // Before
    const int * pointer_variable = &value;
    // After
    const int *const pointer_variable = &value;

    // Before
    const int * a[] = {&value, &value};
    // After
    const int *const a[] = {&value, &value};

    // Before
    int *ptr_value = &value;
    // After
    int *const ptr_value = &value;

    int result = 100 * (*ptr_value); // Does not modify the pointer itself.
    // This modification of the pointee is still allowed and not diagnosed.
    *ptr_value = 0;

    // The following pointer may not become a 'int *const'.
    int *changing_pointee = &value;
    changing_pointee = &result;

.. option:: AllowedTypes (default = '')

  A semicolon-separated list of names of types that
  will be excluded from const-correctness checking.
  Regular expressions are accepted, e.g. `[Rr]ef(erence)?$` matches every type
  with suffix `Ref`, `ref`, `Reference` and `reference`.
  If a name in the list contains the sequence `::`, it is matched against
  the qualified type name (i.e. `namespace::Type`), otherwise it is matched
  against only the type name (i.e. ``Type``).
