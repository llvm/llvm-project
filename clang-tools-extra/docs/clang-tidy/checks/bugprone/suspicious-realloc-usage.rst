.. title:: clang-tidy - bugprone-suspicious-realloc-usage

bugprone-suspicious-realloc-usage
=================================

This check finds usages of ``realloc`` where the return value is assigned to the
same expression as passed to the first argument:
``p = realloc(p, size);``
The problem with this construct is that if ``realloc`` fails it returns a
null pointer but does not deallocate the original memory. If no other variable
is pointing to it, the original memory block is not available any more for the
program to use or free. In either case ``p = realloc(p, size);`` indicates bad
coding style and can be replaced by ``q = realloc(p, size);``. 

The pointer expression (used at ``realloc``) can be a variable or a field member
of a data structure, but can not contain function calls or unresolved types.

In obvious cases when the pointer used at realloc is assigned to another
variable before the ``realloc`` call, no warning is emitted. This happens only
if a simple expression in form of ``q = p`` or ``void *q = p`` is found in the
same function where ``p = realloc(p, ...)`` is found. The assignment has to be
before the call to realloc (but otherwise at any place) in the same function.
This suppression works only if ``p`` is a single variable.

Examples:

.. code-block:: c++

  struct A {
    void *p;
  };

  A &getA();

  void foo(void *p, A *a, int new_size) {
    p = realloc(p, new_size); // warning: 'p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer
    a->p = realloc(a->p, new_size); // warning: 'a->p' may be set to null if 'realloc' fails, which may result in a leak of the original buffer
    getA().p = realloc(getA().p, new_size); // no warning
  }

  void foo1(void *p, int new_size) {
    void *p1 = p;
    p = realloc(p, new_size); // no warning
  }
