.. title:: clang-tidy - zircon-temporary-objects

zircon-temporary-objects
========================

Warns on construction of specific temporary objects in the Zircon kernel. 
If the object should be flagged, If the object should be flagged, the fully 
qualified type name must be explicitly passed to the check.

For example, given the list of classes "Foo" and "NS::Bar", all of the 
following will trigger the warning: 

.. code-block:: c++

  Foo();
  Foo F = Foo();
  func(Foo());

  namespace NS {

  Bar();

  }

With the same list, the following will not trigger the warning:

.. code-block:: c++

  Foo F;				         // Non-temporary construction okay
  Foo F(param);			     // Non-temporary construction okay
  Foo *F = new Foo();	   // New construction okay

  Bar(); 				         // Not NS::Bar, so okay
  NS::Bar B;			       // Non-temporary construction okay

Note that objects must be explicitly specified in order to be flagged, 
and so objects that inherit a specified object will not be flagged.

This check matches temporary objects without regard for inheritance and so a
prohibited base class type does not similarly prohibit derived class types.

.. code-block:: c++

  class Derived : Foo {} // Derived is not explicitly disallowed
  Derived();             // and so temporary construction is okay

Options
-------

.. option:: Names

   A semi-colon-separated list of fully-qualified names of C++ classes that 
   should not be constructed as temporaries. Default is empty.
