.. title:: clang-tidy - modernize-use-auto

modernize-use-auto
==================

This check is responsible for using the ``auto`` type specifier for variable
declarations to *improve code readability and maintainability*. For example:

.. code-block:: c++

  std::vector<int>::iterator I = my_container.begin();

  // transforms to:

  auto I = my_container.begin();

The ``auto`` type specifier will only be introduced in situations where the
variable type matches the type of the initializer expression. In other words
``auto`` should deduce the same type that was originally spelled in the source.
However, not every situation should be transformed:

.. code-block:: c++

  int val = 42;
  InfoStruct &I = SomeObject.getInfo();

  // Should not become:

  auto val = 42;
  auto &I = SomeObject.getInfo();

In this example using ``auto`` for builtins doesn't improve readability. In
other situations it makes the code less self-documenting impairing readability
and maintainability. As a result, ``auto`` is used only introduced in specific
situations described below.

Iterators
---------

Iterator type specifiers tend to be long and used frequently, especially in
loop constructs. Since the functions generating iterators have a common format,
the type specifier can be replaced without obscuring the meaning of code while
improving readability and maintainability.

.. code-block:: c++

  for (std::vector<int>::iterator I = my_container.begin(),
                                  E = my_container.end();
       I != E; ++I) {
  }

  // becomes

  for (auto I = my_container.begin(), E = my_container.end(); I != E; ++I) {
  }

The check will only replace iterator type-specifiers when all of the following
conditions are satisfied:

* The iterator is for one of the standard container in ``std`` namespace:

  * ``array``
  * ``deque``
  * ``forward_list``
  * ``list``
  * ``vector``
  * ``map``
  * ``multimap``
  * ``set``
  * ``multiset``
  * ``unordered_map``
  * ``unordered_multimap``
  * ``unordered_set``
  * ``unordered_multiset``
  * ``queue``
  * ``priority_queue``
  * ``stack``

* The iterator is one of the possible iterator types for standard containers:

  * ``iterator``
  * ``reverse_iterator``
  * ``const_iterator``
  * ``const_reverse_iterator``

* In addition to using iterator types directly, typedefs or other ways of
  referring to those types are also allowed. However, implementation-specific
  types for which a type like ``std::vector<int>::iterator`` is itself a
  typedef will not be transformed. Consider the following examples:

.. code-block:: c++

  // The following direct uses of iterator types will be transformed.
  std::vector<int>::iterator I = MyVec.begin();
  {
    using namespace std;
    list<int>::iterator I = MyList.begin();
  }

  // The type specifier for J would transform to auto since it's a typedef
  // to a standard iterator type.
  typedef std::map<int, std::string>::const_iterator map_iterator;
  map_iterator J = MyMap.begin();

  // The following implementation-specific iterator type for which
  // std::vector<int>::iterator could be a typedef would not be transformed.
  __gnu_cxx::__normal_iterator<int*, std::vector> K = MyVec.begin();

* The initializer for the variable being declared is not a braced initializer
  list. Otherwise, use of ``auto`` would cause the type of the variable to be
  deduced as ``std::initializer_list``.

New expressions
---------------

Frequently, when a pointer is declared and initialized with ``new``, the
pointee type is written twice: in the declaration type and in the
``new`` expression. In this cases, the declaration type can be replaced with
``auto`` improving readability and maintainability.

.. code-block:: c++

  TypeName *my_pointer = new TypeName(my_param);

  // becomes

  auto *my_pointer = new TypeName(my_param);

The check will also replace the declaration type in multiple declarations, if
the following conditions are satisfied:

* All declared variables have the same type (i.e. all of them are pointers to
  the same type).
* All declared variables are initialized with a ``new`` expression.
* The types of all the new expressions are the same than the pointee of the
  declaration type.

.. code-block:: c++

  TypeName *my_first_pointer = new TypeName, *my_second_pointer = new TypeName;

  // becomes

  auto *my_first_pointer = new TypeName, *my_second_pointer = new TypeName;

Cast expressions
----------------

Frequently, when a variable is declared and initialized with a cast, the
variable type is written twice: in the declaration type and in the
cast expression. In this cases, the declaration type can be replaced with
``auto`` improving readability and maintainability.

.. code-block:: c++

  TypeName *my_pointer = static_cast<TypeName>(my_param);

  // becomes

  auto *my_pointer = static_cast<TypeName>(my_param);

The check handles ``static_cast``, ``dynamic_cast``, ``const_cast``,
``reinterpret_cast``, functional casts, C-style casts and function templates
that behave as casts, such as ``llvm::dyn_cast``, ``boost::lexical_cast`` and
``gsl::narrow_cast``.  Calls to function templates are considered to behave as
casts if the first template argument is explicit and is a type, and the function
returns that type, or a pointer or reference to it.

Known Limitations
-----------------

* If the initializer is an explicit conversion constructor, the check will not
  replace the type specifier even though it would be safe to do so.

* User-defined iterators are not handled at this time.

Options
-------

.. option:: RemoveStars

   If the option is set to non-zero (default is `0`), the check will remove
   stars from the non-typedef pointer types when replacing type names with
   ``auto``. Otherwise, the check will leave stars. For example:

.. code-block:: c++

  TypeName *my_first_pointer = new TypeName, *my_second_pointer = new TypeName;

  // RemoveStars = 0

  auto *my_first_pointer = new TypeName, *my_second_pointer = new TypeName;

  // RemoveStars = 1

  auto my_first_pointer = new TypeName, my_second_pointer = new TypeName;
