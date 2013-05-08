.. index:: Loop Convert Transform

======================
Loop Convert Transform
======================

The Loop Convert Transform is a transformation to convert ``for(...; ...;
...)`` loops to use the new range-based loops in C++11. The transform is enabled
with the :option:`-loop-convert` option of :program:`cpp11-migrate`.

Three kinds of loops can be converted:

-  Loops over statically allocated arrays
-  Loops over containers, using iterators
-  Loops over array-like containers, using ``operator[]`` and ``at()``

Risk
====

Risky
^^^^^

In loops where the container expression is more complex than just a
reference to a declared expression (a variable, function, enum, etc.),
and some part of it appears elsewhere in the loop, we lower our confidence
in the transformation due to the increased risk of changing semantics.
Transformations for these loops are marked as `risky`, and thus will only
be converted if the acceptable risk level is set to ``-risk=risky``.

.. code-block:: c++

  int arr[10][20];
  int l = 5;

  for (int j = 0; j < 20; ++j)
    int k = arr[l][j] + l; // using l outside arr[l] is considered risky

  for (int i = 0; i < obj.getVector().size(); ++i)
    obj.foo(10); // using 'obj' is considered risky

See
:ref:`Range-based loops evaluate end() only once<IncorrectRiskyTransformation>`
for an example of an incorrect transformation when the maximum acceptable risk
level is set to `risky`.

Reasonable (Default)
^^^^^^^^^^^^^^^^^^^^

If a loop calls ``.end()`` or ``.size()`` after each iteration, the
transformation for that loop is marked as `reasonable`, and thus will
be converted if the acceptable risk level is set to ``-risk=reasonable``
(default) or higher.

.. code-block:: c++

  // using size() is considered reasonable
  for (int i = 0; i < container.size(); ++i)
    cout << container[i];

Safe
^^^^

Any other loops that do not match the above criteria to be marked as
`risky` or `reasonable` are marked `safe`, and thus will be converted
if the acceptable risk level is set to ``-risk=safe`` or higher.

.. code-block:: c++

  int arr[] = {1,2,3};

  for (int i = 0; i < 3; ++i)
    cout << arr[i];

Example
=======

Original:

.. code-block:: c++

  const int N = 5;
  int arr[] = {1,2,3,4,5};
  vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  // safe transform
  for (int i = 0; i < N; ++i)
    cout << arr[i];

  // reasonable transform
  for (vector<int>::iterator it = v.begin(); it != v.end(); ++it)
    cout << *it;*

  // reasonable transform
  for (int i = 0; i < v.size(); ++i)
    cout << v[i];

After transformation with risk level set to ``-risk=reasonable`` (default):

.. code-block:: c++

  const int N = 5;
  int arr[] = {1,2,3,4,5};
  vector<int> v;
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);

  // safe transform
  for (auto & elem : arr)
    cout << elem;

  // reasonable transform
  for (auto & elem : v)
    cout << elem;

  // reasonable transform
  for (auto & elem : v)
    cout << elem;

Limitations
===========

There are certain situations where the tool may erroneously perform
transformations that remove information and change semantics. Users of the tool
should be aware of the behaviour and limitations of the transform outlined by
the cases below.

Comments inside loop headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comments inside the original loop header are ignored and deleted when
transformed.

.. code-block:: c++

  for (int i = 0; i < N; /* This will be deleted */ ++i) { }

Range-based loops evaluate end() only once
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The C++11 range-based for loop calls ``.end()`` only once during the
initialization of the loop. If in the original loop ``.end()`` is called after
each iteration the semantics of the transformed loop may differ.

.. code-block:: c++

  // The following is semantically equivalent to the C++11 range-based for loop,
  // therefore the semantics of the header will not change.
  for (iterator it = container.begin(), e = container.end(); it != e; ++it) { }

  // Instead of calling .end() after each iteration, this loop will be
  // transformed to call .end() only once during the initialization of the loop,
  // which may affect semantics.
  for (iterator it = container.begin(); it != container.end(); ++it) { }

.. _IncorrectRiskyTransformation:

As explained above, calling member functions of the container in the body
of the loop is considered `risky`. If the called member function modifies the
container the semantics of the converted loop will differ due to ``.end()``
being called only once.

.. code-block:: c++

  bool flag = false;
  for (vector<T>::iterator it = vec.begin(); it != vec.end(); ++it) {
    // Add a copy of the first element to the end of the vector.
    if (!flag) {
      // This line makes this transformation 'risky'.
      vec.push_back(*it);
      flag = true;
    }
    cout << *it;
  }

The original code above prints out the contents of the container including the
newly added element while the converted loop, shown below, will only print the
original contents and not the newly added element.

.. code-block:: c++

  bool flag = false;
  for (auto & elem : vec) {
    // Add a copy of the first element to the end of the vector.
    if (!flag) {
      // This line makes this transformation 'risky'
      vec.push_back(elem);
      flag = true;
    }
    cout << elem;
  }

Semantics will also be affected if ``.end()`` has side effects. For example, in
the case where calls to ``.end()`` are logged the semantics will change in the
transformed loop if ``.end()`` was originally called after each iteration.

.. code-block:: c++

  iterator end() {
    num_of_end_calls++;
    return container.end();
  }

Overloaded operator->() with side effects
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similarly, if ``operator->()`` was overloaded to have side effects, such as
logging, the semantics will change. If the iterator's ``operator->()`` was used
in the original loop it will be replaced with ``<container element>.<member>``
instead due to the implicit dereference as part of the range-based for loop.
Therefore any side effect of the overloaded ``operator->()`` will no longer be
performed.

.. code-block:: c++

  for (iterator it = c.begin(); it != c.end(); ++it) {
    it->func(); // Using operator->()
  }
  // Will be transformed to:
  for (auto & elem : c) {
    elem.func(); // No longer using operator->()
  }

Pointers and references to containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While most of the transform's risk analysis is dedicated to determining whether
the iterator or container was modified within the loop, it is possible to
circumvent the analysis by accessing and modifying the container through a
pointer or reference.

If the container were directly used instead of using the pointer or reference
the following transformation would have only been applied at the ``-risk=risky``
level since calling a member function of the container is considered `risky`.
The transform cannot identify expressions associated with the container that are
different than the one used in the loop header, therefore the transformation
below ends up being performed at the ``-risk=safe`` level.

.. code-block:: c++

  vector<int> vec;

  vector<int> *ptr = &vec;
  vector<int> &ref = vec;

  for (vector<int>::iterator it = vec.begin(), e = vec.end(); it != e; ++it) {
    if (!flag) {
      // Accessing and modifying the container is considered risky, but the risk
      // level is not raised here.
      ptr->push_back(*it);
      ref.push_back(*it);
      flag = true;
    }
  }
