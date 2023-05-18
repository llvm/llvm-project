.. title:: clang-tidy - readability-redundant-smartptr-get

readability-redundant-smartptr-get
==================================

Find and remove redundant calls to smart pointer's ``.get()`` method.

Examples:

.. code-block:: c++

  ptr.get()->Foo()  ==>  ptr->Foo()
  *ptr.get()  ==>  *ptr
  *ptr->get()  ==>  **ptr
  if (ptr.get() == nullptr) ... => if (ptr == nullptr) ...


.. option:: IgnoreMacros

   If this option is set to `true` (default is `true`), the check will not warn
   about calls inside macros.
