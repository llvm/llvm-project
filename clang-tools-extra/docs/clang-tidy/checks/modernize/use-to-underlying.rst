.. title:: clang-tidy - modernize-use-to-underlying

modernize-use-to-underlying
===========================

Finds casts from a scoped enumeration (``enum class``) to an integer type and
replaces them with a call to ``std::to_underlying`` (introduced in C++23).

Converting a scoped enumeration to a hard-coded integer type is error-prone: if
the enumeration's underlying type is later changed, every such cast silently
becomes a narrowing, widening or sign-changing conversion. ``std::to_underlying``
always yields exactly the underlying type and keeps the code correct.

.. code-block:: c++

  enum class Color : unsigned char { Red, Green, Blue };

  void f(Color c) {
    // Before:
    auto value = static_cast<unsigned char>(c);
    // After:
    auto value = std::to_underlying(c);
  }

The check matches ``static_cast``, C-style casts and functional-style casts.

Precise and imprecise casts
---------------------------

A cast is *precise* when its destination type is exactly the underlying type of
the enumeration. In that case the whole cast is replaced:

.. code-block:: c++

  enum class E : int {};

  int i = static_cast<int>(E{});
  // becomes:
  int i = std::to_underlying(E{});

A cast is *imprecise* when the destination type is an integer type other than
the underlying type (a different width or signedness), for example
``static_cast<long>`` or ``static_cast<unsigned>`` on an ``enum : int``. Such a
cast performs an additional integer conversion, so there is no single correct
rewrite; how imprecise casts are handled is controlled by the
:option:`ImpreciseCasts` option.

A cast to a non-integer type (floating point, pointer, another enumeration) is
never flagged. A cast to ``bool`` is only flagged when ``bool`` is the exact
underlying type of the enumeration; otherwise it is treated as a truthiness
test and left untouched.

Options
-------

.. option:: ImpreciseCasts

   Controls how imprecise casts (whose destination type differs from the
   underlying type) are handled. Precise casts are always diagnosed and fixed
   regardless of this option. Possible values:

   `Ignore`
     Do not diagnose imprecise casts.

   `Warn` *(default)*
     Diagnose imprecise casts but do not offer a fix-it. Neither automatic
     rewrite is applied because both change the meaning of the code in ways
     that may not be intended.

   `PreserveType`
     Wrap the operand in a call to the replacement function, keeping the
     original destination type:

     .. code-block:: c++

       long l = static_cast<long>(E{});
       // becomes:
       long l = static_cast<long>(std::to_underlying(E{}));

   `UseUnderlyingType`
     Replace the whole cast with a call to the replacement function. This
     **changes the type** of the expression from the destination type to the
     underlying type, so use it only when the wider or differently-signed
     destination type was itself unintended:

     .. code-block:: c++

       long l = static_cast<long>(E{});
       // becomes:
       long l = std::to_underlying(E{});

.. option:: ReplacementFunction

   The fully qualified name of the function used in the replacement. Defaults to
   ``std::to_underlying``. Set this to use a hand-rolled equivalent (for example
   ``llvm::to_underlying``) when targeting a language standard before C++23. When
   the value is ``std::to_underlying``, the check only runs in C++23 or later;
   with any other value it runs from C++11 onwards.

.. option:: ReplacementFunctionHeader

   The header to include when the replacement function is used. Defaults to
   ``<utility>`` when :option:`ReplacementFunction` is set to `std::to_underlying`,
   and is otherwise empty (no include is added). When the value is enclosed in
   angle brackets the include directive uses angle brackets, otherwise it uses
   quotes.

.. option:: IncludeStyle

   A string specifying which include-style is used, `llvm` or `google`.
   Default is `llvm`.
