.. title:: clang-tidy - cppcoreguidelines-virtual-class-destructor

cppcoreguidelines-virtual-class-destructor
===============================================

Finds virtual classes whose destructor is neither public and virtual
nor protected and non-virtual. A virtual class's destructor should be specified
in one of these ways to prevent undefined behavior.

This check implements
`C.35 <http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-dtor-virtual>`_
from the C++ Core Guidelines.

Note that this check will diagnose a class with a virtual method regardless of
whether the class is used as a base class or not.

Fixes are available for user-declared and implicit destructors that are either
public and non-virtual or protected and virtual. No fixes are offered for
private destructors. There, the decision whether to make them private and
virtual or protected and non-virtual depends on the use case and is thus left
to the user.

Example
-------

For example, the following classes/structs get flagged by the check since they
violate guideline **C.35**:

.. code-block:: c++

  struct Foo {        // NOK, protected destructor should not be virtual
    virtual void f();
  protected:
    virtual ~Foo(){}
  };

  class Bar {         // NOK, public destructor should be virtual
    virtual void f();
  public:
    ~Bar(){}
  };

This would be rewritten to look like this:

.. code-block:: c++

  struct Foo {        // OK, destructor is not virtual anymore
    virtual void f();
  protected:
    ~Foo(){}
  };

  class Bar {         // OK, destructor is now virtual
    virtual void f();
  public:
    virtual ~Bar(){}
  };
