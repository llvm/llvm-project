.. title:: clang-tidy - modernize-use-scoped-lock

modernize-use-scoped-lock
=========================

Finds uses of ``std::lock_guard`` and suggests replacing them with C++17's
alternative ``std::scoped_lock``.

Fix-its are provided for single declarations of ``std::lock_guard`` and warning
is emitted for multiple declarations of ``std::lock_guard`` that can be
replaced with a single declaration of ``std::scoped_lock``.

Examples
--------

Single ``std::lock_guard`` declaration:

.. code-block:: c++

  std::mutex M;
  std::lock_guard<std::mutex> L(M);


Transforms to:

.. code-block:: c++

  std::mutex M;
  std::scoped_lock L(M);

Single ``std::lock_guard`` declaration with ``std::adopt_lock``:

.. code-block:: c++

  std::mutex M;
  std::lock(M);
  std::lock_guard<std::mutex> L(M, std::adopt_lock);


Transforms to:

.. code-block:: c++

  std::mutex M;
  std::lock(M);
  std::scoped_lock L(std::adopt_lock, M);

Multiple ``std::lock_guard`` declarations only emit warnings:

.. code-block:: c++

  std::mutex M1, M2;
  std::lock(M1, M2);
  std::lock_guard Lock1(M1, std::adopt_lock); // warning: use single 'std::scoped_lock' instead of multiple 'std::lock_guard'
  std::lock_guard Lock2(M2, std::adopt_lock); // note: additional 'std::lock_guard' declared here


Limitations
-----------

The check will not emit warnings if ``std::lock_guard`` is used implicitly via
``template`` parameter:

.. code-block:: c++

  template <template <typename> typename Lock>
  void TemplatedLock() {
    std::mutex M;
    Lock<std::mutex> L(M); // no warning
  }

  void instantiate() {
    TemplatedLock<std::lock_guard>();
  }


Options
-------

.. option:: WarnOnSingleLocks

  When `true`, the check will warn on single ``std::lock_guard`` declarations.
  Set this option to `false` if you want to get warnings only on multiple
  ``std::lock_guard`` declarations that can be replaced with a single
  ``std::scoped_lock``. Default is `true`.

.. option:: WarnOnUsingAndTypedef

  When `true`, the check will emit warnings if ``std::lock_guard`` is used
  in ``using`` or ``typedef`` context. Default is `true`.

  .. code-block:: c++

    template <typename T>
    using Lock = std::lock_guard<T>; // warning: use 'std::scoped_lock' instead of 'std::lock_guard'

    using LockMutex = std::lock_guard<std::mutex>; // warning: use 'std::scoped_lock' instead of 'std::lock_guard'

    typedef std::lock_guard<std::mutex> LockDef; // warning: use 'std::scoped_lock' instead of 'std::lock_guard'

    using std::lock_guard; // warning: use 'std::scoped_lock' instead of 'std::lock_guard'
