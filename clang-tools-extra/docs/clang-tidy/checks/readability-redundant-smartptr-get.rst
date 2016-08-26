.. title:: clang-tidy - readability-redundant-smartptr-get

readability-redundant-smartptr-get
==================================

`google-readability-redundant-smartptr-get` redirects here as an alias for this
check.

Find and remove redundant calls to smart pointer's ``.get()`` method.

Examples:

.. code-block:: c++

  ptr.get()->Foo()  ==>  ptr->Foo()
  *ptr.get()  ==>  *ptr
  *ptr->get()  ==>  **ptr

