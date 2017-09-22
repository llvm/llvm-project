.. title:: clang-tidy - modernize-replace-random-shuffle

modernize-replace-random-shuffle
================================

This check will find occurrences of ``std::random_shuffle`` and replace it with ``std::shuffle``. In C++17 ``std::random_shuffle`` will no longer be available and thus we need to replace it.

Below are two examples of what kind of occurrences will be found and two examples of what it will be replaced with.

.. code-block:: c++

  std::vector<int> v;

  // First example
  std::random_shuffle(vec.begin(), vec.end());

  // Second example
  std::random_shuffle(vec.begin(), vec.end(), randomFunc);

Both of these examples will be replaced with:

.. code-block:: c++

  std::shuffle(vec.begin(), vec.end(), std::mt19937(std::random_device()()));

The second example will also receive a warning that ``randomFunc`` is no longer supported in the same way as before so if the user wants the same functionality, the user will need to change the implementation of the ``randomFunc``.

One thing to be aware of here is that ``std::random_device`` is quite expensive to initialize. So if you are using the code in a performance critical place, you probably want to initialize it elsewhere. 
