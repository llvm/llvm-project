.. title:: clang-tidy - misc-uniqueptr-reset-release

misc-uniqueptr-reset-release
============================

Find and replace ``unique_ptr::reset(release())`` with ``std::move()``.

Example:

.. code-block:: c++

  std::unique_ptr<Foo> x, y;
  x.reset(y.release()); -> x = std::move(y);

If ``y`` is already rvalue, ``std::move()`` is not added. ``x`` and ``y`` can
also be ``std::unique_ptr<Foo>*``.
