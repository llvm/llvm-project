.. title:: clang-tidy - cppcoreguidelines-const-correctness

cppcoreguidelines-const-correctness
===================================

This check implements detection of local variables which could be declared as
``const``, but are not. Declaring variables as ``const`` is required by many
coding guidelines, such as:
`CppCoreGuidelines ES.25 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#es25-declare-an-object-const-or-constexpr-unless-you-want-to-modify-its-value-later-on>`_
and `High Integrity C++ 7.1.2 <http://www.codingstandard.com/rule/7-1-2-use-const-whenever-possible/>`_.

Please note that this analysis is type-based only. Variables that are not modified
but non-const handles might escape out of the scope are not diagnosed as potential
``const``.

.. code-block:: c++
  
  // Declare a variable, which is not ``const`` ...
  int i = 42;
  // but use it as read-only. This means that `i` can be declared ``const``.
  int result = i * i;

The check analyzes values, pointers and references (if configured that way).
For better understanding some code samples:

.. code-block:: c++

  // Normal values like built-ins or objects.
  int potential_const_int = 42;
  int copy_of_value = potential_const_int;

  MyClass could_be_const;
  could_be_const.const_qualified_method();

  // References can be declared const as well.
  int &reference_value = potential_const_int;
  int another_copy = reference_value;

  // Similar behaviour for pointers.
  int *pointer_variable = &potential_const_int;
  int last_copy = *pointer_variable;


Options
-------

.. option:: AnalyzeValues (default = 1)

  Enable or disable the analysis of ordinary value variables, like ``int i = 42;``

.. option:: AnalyzeReferences (default = 1)

  Enable or disable the analysis of reference variables, like ``int &ref = i;``

.. option:: WarnPointersAsValues (default = 0)

  This option enables the suggestion for ``const`` of the pointer itself.
  Pointer values have two possibilities to be ``const``, the pointer itself
  and the value pointing to. 

  .. code-block:: c++

    const int value = 42;
    const int * const pointer_variable = &value;
    
    // The following operations are forbidden for `pointer_variable`.
    // *pointer_variable = 44;
    // pointer_variable = nullptr;
