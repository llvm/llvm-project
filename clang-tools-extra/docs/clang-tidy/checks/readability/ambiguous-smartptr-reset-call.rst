.. title:: clang-tidy - readability-ambiguous-smartptr-reset-call

readability-ambiguous-smartptr-reset-call
=========================================

Finds potentially erroneous calls to ``reset`` method on smart pointers when
the pointee type also has a ``reset`` method. Having a ``reset`` method in
both classes makes it easy to accidentally make the pointer null when
intending to reset the underlying object.

.. code-block:: c++

  struct Resettable {
    void reset() { /* Own reset logic */ }
  };

  auto ptr = std::make_unique<Resettable>();

  ptr->reset();  // Calls underlying reset method
  ptr.reset();   // Makes the pointer null

Both calls are valid C++ code, but the second one might not be what the
developer intended, as it destroys the pointed-to object rather than resetting
its state. It's easy to make such a typo because the difference between
``.`` and ``->`` is really small.

The recommended approach is to make the intent explicit by using either member
access or direct assignment:

.. code-block:: c++

  std::unique_ptr<Resettable> ptr = std::make_unique<Resettable>();

  (*ptr).reset();  // Clearly calls underlying reset method
  ptr = nullptr;   // Clearly makes the pointer null

The default smart pointers and classes that are considered are
``std::unique_ptr``, ``std::shared_ptr``, ``boost::shared_ptr``. To specify
other smart pointers or other classes use the :option:`SmartPointers` option.


.. note::
    
  The check may emit invalid fix-its and misleading warning messages when
  specifying custom smart pointers or other classes in the
  :option:`SmartPointers` option. For example, ``boost::scoped_ptr`` does not
  have an ``operator=`` which makes fix-its invalid.

.. note::
    
  Automatic fix-its are enabled only if :program:`clang-tidy` is invoked with
  the `--fix-notes` option.


Options
-------

.. option:: SmartPointers

    Semicolon-separated list of fully qualified class names of custom smart
    pointers. Default value is `::std::unique_ptr;::std::shared_ptr;
    ::boost::shared_ptr`.
