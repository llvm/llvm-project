.. title:: clang-tidy - misc-override-with-different-visibility

misc-override-with-different-visibility
=======================================

Finds virtual function overrides with different visibility than the function
in the base class. This includes for example if a virtual function declared as
``private`` is overridden and declared as ``public`` in a subclass. The detected
change is the modification of visibility resulting from keywords ``public``,
``protected``, ``private`` at overridden virtual functions. The check applies to
any normal virtual function and optionally to destructors or operators. Use of
the ``using`` keyword is not considered as visibility change by this check.


.. code-block:: c++

  class A {
  public:
    virtual void f_pub();
  private:
    virtual void f_priv();
  };
  
  class B: public A {
  public:
    void f_priv(); // warning: changed visibility from private to public
  private:
    void f_pub(); // warning: changed visibility from public to private
  };

  class C: private A {
    // no warning: f_pub becomes private in this case but this is from the
    // private inheritance
  };

  class D: private A {
  public:
    void f_pub(); // warning: changed visibility from private to public
                  // 'f_pub' would have private access but is forced to be
                  // public
  };

If the visibility is changed in this way, it can indicate bad design or
programming error.

If a virtual function is private in a subclass but public in the base class, it
can still be accessed from a pointer to the subclass if the pointer is converted
to the base type. Probably private inheritance can be used instead.

A protected virtual function that is made public in a subclass may have valid
use cases but similar (not exactly same) effect can be achieved with the
``using`` keyword.

Options
-------

.. option:: DisallowedVisibilityChange

  Controls what kind of change to the visibility will be detected by the check.
  Possible values are `any`, `widening`, `narrowing`. For example the
  `widening` option will produce warning only if the visibility is changed
  from more restrictive (``private``) to less restrictive (``public``).
  Default value is `any`.

.. option:: CheckDestructors

  If `true`, the check does apply to destructors too. Otherwise destructors
  are ignored by the check.
  Default value is `false`.

.. option:: CheckOperators

  If `true`, the check does apply to overloaded C++ operators (as virtual
  member functions) too. This includes other special member functions (like
  conversions) too. This option is probably useful only in rare cases because
  operators and conversions are not often virtual functions.
  Default value is `false`.

.. option:: IgnoredFunctions

  This option can be used to ignore the check at specific functions.
  To configure this option, a semicolon-separated list of function names
  should be provided. The list can contain regular expressions, in this way it
  is possible to select all functions of a specific class (like `MyClass::.*`)
  or a specific function of any class (like `my_function` or
  `::.*::my_function`). The function names are matched at the base class.
  Default value is empty string.
