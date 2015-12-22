.. title:: clang-tidy - google-readability-redundant-smartptr-get

google-readability-redundant-smartptr-get
=========================================


Find and remove redundant calls to smart pointer's ``.get()`` method.

Examples:

.. code:: c++

  ptr.get()->Foo()  ==>  ptr->Foo()
  *ptr.get()  ==>  *ptr
  *ptr->get()  ==>  **ptr

