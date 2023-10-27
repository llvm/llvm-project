.. title:: clang-tidy - hicpp-undelegated-constructor
.. meta::
   :http-equiv=refresh: 5;URL=../bugprone/undelegated-constructor.html

hicpp-undelegated-constructor
=============================

This check is an alias for :doc:`bugprone-undelegated-constructor <../bugprone/undelegated-constructor>`.
Partially implements `rule 12.4.5 <https://www.perforce.com/resources/qac/high-integrity-cpp-coding-standard/special-member-functions>`_
to find misplaced constructor calls inside a constructor.

.. code-block:: c++

  struct Ctor {
    Ctor();
    Ctor(int);
    Ctor(int, int);
    Ctor(Ctor *i) {
      // All Ctor() calls result in a temporary object
      Ctor(); // did you intend to call a delegated constructor?
      Ctor(0); // did you intend to call a delegated constructor?
      Ctor(1, 2); // did you intend to call a delegated constructor?
      foo();
    }
  };
