.. title:: clang-tidy - hicpp-undelegated-construtor
.. meta::
   :http-equiv=refresh: 5;URL=misc-undelegated-constructor.html

hicpp-undelegated-constructor
=============================

This check is an alias for `misc-undelegated-constructor <misc-undelegated-constructor.html>`_.
Partially implements `rule 12.4.5 <http://www.codingstandard.com/rule/12-4-5-use-delegating-constructors-to-reduce-code-duplication/>`_ 
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
