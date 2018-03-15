.. title:: clang-tidy - bugprone-sizeof-expression

bugprone-sizeof-expression
==========================

The check finds usages of ``sizeof`` expressions which are most likely errors.

The ``sizeof`` operator yields the size (in bytes) of its operand, which may be
an expression or the parenthesized name of a type. Misuse of this operator may
be leading to errors and possible software vulnerabilities.

Suspicious usage of 'sizeof(K)'
-------------------------------

A common mistake is to query the ``sizeof`` of an integer literal. This is
equivalent to query the size of its type (probably ``int``). The intent of the
programmer was probably to simply get the integer and not its size.

.. code-block:: c++

  #define BUFLEN 42
  char buf[BUFLEN];
  memset(buf, 0, sizeof(BUFLEN));  // sizeof(42) ==> sizeof(int)

Suspicious usage of 'sizeof(this)'
----------------------------------

The ``this`` keyword is evaluated to a pointer to an object of a given type.
The expression ``sizeof(this)`` is returning the size of a pointer. The
programmer most likely wanted the size of the object and not the size of the
pointer.

.. code-block:: c++

  class Point {
    [...]
    size_t size() { return sizeof(this); }  // should probably be sizeof(*this)
    [...]
  };

Suspicious usage of 'sizeof(char*)'
-----------------------------------

There is a subtle difference between declaring a string literal with
``char* A = ""`` and ``char A[] = ""``. The first case has the type ``char*``
instead of the aggregate type ``char[]``. Using ``sizeof`` on an object declared
with ``char*`` type is returning the size of a pointer instead of the number of
characters (bytes) in the string literal.

.. code-block:: c++

  const char* kMessage = "Hello World!";      // const char kMessage[] = "...";
  void getMessage(char* buf) {
    memcpy(buf, kMessage, sizeof(kMessage));  // sizeof(char*)
  }

Suspicious usage of 'sizeof(A*)'
--------------------------------

A common mistake is to compute the size of a pointer instead of its pointee.
These cases may occur because of explicit cast or implicit conversion.

.. code-block:: c++

  int A[10];
  memset(A, 0, sizeof(A + 0));

  struct Point point;
  memset(point, 0, sizeof(&point));

Suspicious usage of 'sizeof(...)/sizeof(...)'
---------------------------------------------

Dividing ``sizeof`` expressions is typically used to retrieve the number of
elements of an aggregate. This check warns on incompatible or suspicious cases.

In the following example, the entity has 10-bytes and is incompatible with the
type ``int`` which has 4 bytes.

.. code-block:: c++

  char buf[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };  // sizeof(buf) => 10
  void getMessage(char* dst) {
    memcpy(dst, buf, sizeof(buf) / sizeof(int));  // sizeof(int) => 4  [incompatible sizes]
  }

In the following example, the expression ``sizeof(Values)`` is returning the
size of ``char*``. One can easily be fooled by its declaration, but in parameter
declaration the size '10' is ignored and the function is receiving a ``char*``.

.. code-block:: c++

  char OrderedValues[10] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  return CompareArray(char Values[10]) {
    return memcmp(OrderedValues, Values, sizeof(Values)) == 0;  // sizeof(Values) ==> sizeof(char*) [implicit cast to char*]
  }

Suspicious 'sizeof' by 'sizeof' expression
------------------------------------------

Multiplying ``sizeof`` expressions typically makes no sense and is probably a
logic error. In the following example, the programmer used ``*`` instead of
``/``.

.. code-block:: c++

  const char kMessage[] = "Hello World!";
  void getMessage(char* buf) {
    memcpy(buf, kMessage, sizeof(kMessage) * sizeof(char));  //  sizeof(kMessage) / sizeof(char)
  }

This check may trigger on code using the arraysize macro. The following code is
working correctly but should be simplified by using only the ``sizeof``
operator.

.. code-block:: c++

  extern Object objects[100];
  void InitializeObjects() {
    memset(objects, 0, arraysize(objects) * sizeof(Object));  // sizeof(objects)
  }

Suspicious usage of 'sizeof(sizeof(...))'
-----------------------------------------

Getting the ``sizeof`` of a ``sizeof`` makes no sense and is typically an error
hidden through macros.

.. code-block:: c++

  #define INT_SZ sizeof(int)
  int buf[] = { 42 };
  void getInt(int* dst) {
    memcpy(dst, buf, sizeof(INT_SZ));  // sizeof(sizeof(int)) is suspicious.
  }

Options
-------

.. option:: WarnOnSizeOfConstant

   When non-zero, the check will warn on an expression like
   ``sizeof(CONSTANT)``. Default is `1`.

.. option:: WarnOnSizeOfThis

   When non-zero, the check will warn on an expression like ``sizeof(this)``.
   Default is `1`.

.. option:: WarnOnSizeOfCompareToConstant

   When non-zero, the check will warn on an expression like
   ``sizeof(epxr) <= k`` for a suspicious constant `k` while `k` is `0` or
   greater than `0x8000`. Default is `1`.
