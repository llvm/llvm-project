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

Suspicious usage of 'sizeof(expr)'
----------------------------------

In cases, where there is an enum or integer to represent a type, a common
mistake is to query the ``sizeof`` on the integer or enum that represents the
type that should be used by ``sizeof``. This results in the size of the integer
and not of the type the integer represents:

.. code-block:: c++

  enum data_type {
    FLOAT_TYPE,
    DOUBLE_TYPE
  };

  struct data {
    data_type type;
    void* buffer;
    data_type get_type() {
      return type;
    }
  };

  void f(data d, int numElements) {
    // should be sizeof(float) or sizeof(double), depending on d.get_type()
    int numBytes = numElements * sizeof(d.get_type());
    ...
  }


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

Suspicious usages of 'sizeof(...)' in pointer arithmetic
--------------------------------------------------------

Arithmetic operators on pointers automatically scale the result with the size
of the pointed typed.
Further use of ``sizeof`` around pointer arithmetic will typically result in an
unintended result.

Scaling the result of pointer difference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Subtracting two pointers results in an integer expression (of type
``ptrdiff_t``) which expresses the distance between the two pointed objects in
"number of objects between".
A common mistake is to think that the result is "number of bytes between", and
scale the difference with ``sizeof``, such as ``P1 - P2 == N * sizeof(T)``
(instead of ``P1 - P2 == N``) or ``(P1 - P2) / sizeof(T)`` instead of
``P1 - P2``.

.. code-block:: c++

  void splitFour(const Obj* Objs, size_t N, Obj Delimiter) {
    const Obj *P = Objs;
    while (P < Objs + N) {
      if (*P == Delimiter) {
        break;
      }
    }

    if (P - Objs != 4 * sizeof(Obj)) { // Expecting a distance multiplied by sizeof is suspicious.
      error();
    }
  }

.. code-block:: c++

  void iterateIfEvenLength(int *Begin, int *End) {
    auto N = (Begin - End) / sizeof(int); // Dividing by sizeof() is suspicious.
    if (N % 2)
      return;

    // ...
  }

Stepping a pointer with a scaled integer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversely, when performing pointer arithmetics to add or subtract from a
pointer, the arithmetic operator implicitly scales the value actually added to
the pointer with the size of the pointee, as ``Ptr + N`` expects ``N`` to be
"number of objects to step", and not "number of bytes to step".

Seeing the calculation of a pointer where ``sizeof`` appears is suspicious,
and the result is typically unintended, often out of bounds.
``Ptr + sizeof(T)`` will offset the pointer by ``sizeof(T)`` elements,
effectively exponentiating the scaling factor to the power of 2.

Similarly, multiplying or dividing a numeric value with the ``sizeof`` of an
element or the whole buffer is suspicious, because the dimensional connection
between the numeric value and the actual ``sizeof`` result can not always be
deduced.
While scaling an integer up (multiplying) with ``sizeof`` is likely **always**
an issue, a scaling down (division) is not always inherently dangerous, in case
the developer is aware that the division happens between an appropriate number
of _bytes_ and a ``sizeof`` value.
Turning :option:`WarnOnOffsetDividedBySizeOf` off will restrict the
warnings to the multiplication case.

This case also checks suspicious ``alignof`` and ``offsetof`` usages in
pointer arithmetic, as both return the "size" in bytes and not elements,
potentially resulting in doubly-scaled offsets.

.. code-block:: c++

  void printEveryEvenIndexElement(int *Array, size_t N) {
    int *P = Array;
    while (P <= Array + N * sizeof(int)) { // Suspicious pointer arithmetic using sizeof()!
      printf("%d ", *P);

      P += 2 * sizeof(int); // Suspicious pointer arithmetic using sizeof()!
    }
  }

.. code-block:: c++

  struct Message { /* ... */; char Flags[8]; };
  void clearFlags(Message *Array, size_t N) {
    const Message *End = Array + N;
    while (Array < End) {
      memset(Array + offsetof(Message, Flags), // Suspicious pointer arithmetic using offsetof()!
             0, sizeof(Message::Flags));
      ++Array;
    }
  }

For this checked bogus pattern, `cert-arr39-c` redirects here as an alias of
this check.

This check corresponds to the CERT C Coding Standard rule
`ARR39-C. Do not add or subtract a scaled integer to a pointer
<http://wiki.sei.cmu.edu/confluence/display/c/ARR39-C.+Do+not+add+or+subtract+a+scaled+integer+to+a+pointer>`_.

Limitations
"""""""""""

Cases where the pointee type has a size of `1` byte (such as, and most
importantly, ``char``) are excluded.

Options
-------

.. option:: WarnOnSizeOfConstant

   When `true`, the check will warn on an expression like
   ``sizeof(CONSTANT)``. Default is `true`.

.. option:: WarnOnSizeOfIntegerExpression

   When `true`, the check will warn on an expression like ``sizeof(expr)``
   where the expression results in an integer. Default is `false`.

.. option:: WarnOnSizeOfThis

   When `true`, the check will warn on an expression like ``sizeof(this)``.
   Default is `true`.

.. option:: WarnOnSizeOfCompareToConstant

   When `true`, the check will warn on an expression like
   ``sizeof(expr) <= k`` for a suspicious constant `k` while `k` is `0` or
   greater than `0x8000`. Default is `true`.

.. option:: WarnOnSizeOfPointerToAggregate

   When `true`, the check will warn when the argument of ``sizeof`` is either a
   pointer-to-aggregate type, an expression returning a pointer-to-aggregate
   value or an expression that returns a pointer from an array-to-pointer
   conversion (that may be implicit or explicit, for example ``array + 2`` or
   ``(int *)array``). Default is `true`.

.. option:: WarnOnSizeOfPointer

   When `true`, the check will report all expressions where the argument of
   ``sizeof`` is an expression that produces a pointer (except for a few
   idiomatic expressions that are probably intentional and correct).
   This detects occurrences of CWE 467. Default is `false`.

.. option:: WarnOnOffsetDividedBySizeOf

   When `true`, the check will warn on pointer arithmetic where the
   element count is obtained from a division with ``sizeof(...)``,
   e.g., ``Ptr + Bytes / sizeof(*T)``. Default is `true`.
