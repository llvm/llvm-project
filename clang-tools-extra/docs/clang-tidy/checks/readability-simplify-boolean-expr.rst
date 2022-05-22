.. title:: clang-tidy - readability-simplify-boolean-expr

readability-simplify-boolean-expr
=================================

Looks for boolean expressions involving boolean constants and simplifies
them to use the appropriate boolean expression directly.  Simplifies
boolean expressions by application of DeMorgan's Theorem.

Examples:

===========================================  ================
Initial expression                           Result
-------------------------------------------  ----------------
``if (b == true)``                             ``if (b)``
``if (b == false)``                            ``if (!b)``
``if (b && true)``                             ``if (b)``
``if (b && false)``                            ``if (false)``
``if (b || true)``                             ``if (true)``
``if (b || false)``                            ``if (b)``
``e ? true : false``                           ``e``
``e ? false : true``                           ``!e``
``if (true) t(); else f();``                   ``t();``
``if (false) t(); else f();``                  ``f();``
``if (e) return true; else return false;``     ``return e;``
``if (e) return false; else return true;``     ``return !e;``
``if (e) b = true; else b = false;``           ``b = e;``
``if (e) b = false; else b = true;``           ``b = !e;``
``if (e) return true; return false;``          ``return e;``
``if (e) return false; return true;``          ``return !e;``
``!(!a || b)``                                 ``a && !b``
``!(a || !b)``                                 ``!a && b``
``!(!a || !b)``                                ``a && b``
``!(!a && b)``                                 ``a || !b``
``!(a && !b)``                                 ``!a || b``
``!(!a && !b)``                                ``a || b``
===========================================  ================

The resulting expression ``e`` is modified as follows:
  1. Unnecessary parentheses around the expression are removed.
  2. Negated applications of ``!`` are eliminated.
  3. Negated applications of comparison operators are changed to use the
     opposite condition.
  4. Implicit conversions of pointers, including pointers to members, to
     ``bool`` are replaced with explicit comparisons to ``nullptr`` in C++11
     or ``NULL`` in C++98/03.
  5. Implicit casts to ``bool`` are replaced with explicit casts to ``bool``.
  6. Object expressions with ``explicit operator bool`` conversion operators
     are replaced with explicit casts to ``bool``.
  7. Implicit conversions of integral types to ``bool`` are replaced with
     explicit comparisons to ``0``.

Examples:
  1. The ternary assignment ``bool b = (i < 0) ? true : false;`` has redundant
     parentheses and becomes ``bool b = i < 0;``.

  2. The conditional return ``if (!b) return false; return true;`` has an
     implied double negation and becomes ``return b;``.

  3. The conditional return ``if (i < 0) return false; return true;`` becomes
     ``return i >= 0;``.

     The conditional return ``if (i != 0) return false; return true;`` becomes
     ``return i == 0;``.

  4. The conditional return ``if (p) return true; return false;`` has an
     implicit conversion of a pointer to ``bool`` and becomes
     ``return p != nullptr;``.

     The ternary assignment ``bool b = (i & 1) ? true : false;`` has an
     implicit conversion of ``i & 1`` to ``bool`` and becomes
     ``bool b = (i & 1) != 0;``.

  5. The conditional return ``if (i & 1) return true; else return false;`` has
     an implicit conversion of an integer quantity ``i & 1`` to ``bool`` and
     becomes ``return (i & 1) != 0;``

  6. Given ``struct X { explicit operator bool(); };``, and an instance ``x`` of
     ``struct X``, the conditional return ``if (x) return true; return false;``
     becomes ``return static_cast<bool>(x);``

Options
-------

.. option:: ChainedConditionalReturn

   If `true`, conditional boolean return statements at the end of an
   ``if/else if`` chain will be transformed. Default is `false`.

.. option:: ChainedConditionalAssignment

   If `true`, conditional boolean assignments at the end of an ``if/else
   if`` chain will be transformed. Default is `false`.

.. option:: SimplifyDeMorgan

   If `true`, DeMorgan's Theorem will be applied to simplify negated
   conjunctions and disjunctions.  Default is `true`.
