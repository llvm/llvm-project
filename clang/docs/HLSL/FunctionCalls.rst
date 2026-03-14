===================
HLSL Function Calls
===================

.. contents::
   :local:

Introduction
============

This document describes the design and implementation of HLSL's function call
semantics in Clang. This includes details related to argument conversion and
parameter lifetimes.

This document does not seek to serve as official documentation for HLSL's
call semantics, but does provide an overview to assist a reader. The
authoritative documentation for HLSL's language semantics is the `draft language
specification <https://microsoft.github.io/hlsl-specs/specs/hlsl.pdf>`_.

Argument Semantics
==================

In HLSL, all function arguments are passed by value in and out of functions.
HLSL has 3 keywords which denote the parameter semantics (``in``, ``out`` and
``inout``). In a function declaration a parameter may be annotated any of the
following ways:

#. <no parameter annotation> - denotes input
#. ``in`` - denotes input
#. ``out`` - denotes output
#. ``in out`` - denotes input and output
#. ``out in`` - denotes input and output
#. ``inout`` - denotes input and output

Parameters that are exclusively input behave like C/C++ parameters that are
passed by value.

For parameters that are output (or input and output), a temporary value is
created in the caller. The temporary value is then passed by-address. For
output-only parameters, the temporary is uninitialized when passed (if the
parameter is not explicitly initialized inside the function an undefined value
is stored back to the argument expression). For parameters that are both input
and output, the temporary is initialized from the lvalue argument expression
through implicit  or explicit casting from the lvalue argument type to the
parameter type.

On return of the function, the values of any parameter temporaries are written
back to the argument expression through an inverted conversion sequence (if an
``out`` parameter was not initialized in the function, the uninitialized value
may be written back).

Parameters of constant-sized array type are also passed with value semantics.
This requires input parameters of arrays to construct temporaries and the
temporaries go through array-to-pointer decay when initializing parameters.

Implementations are allowed to avoid unnecessary temporaries, and HLSL's strict
no-alias rules can enable some trivial optimizations.

Array Temporaries
-----------------

Given the following example:

.. code-block:: c++

  void fn(float a[4]) {
    a[0] = a[1] + a[2] + a[3];
  }

  float4 main() : SV_Target {
    float arr[4] = {1, 1, 1, 1};
    fn(arr);
    return float4(arr[0], arr[1], arr[2], arr[3]);
  }

In C or C++, the array parameter decays to a pointer, so after the call to
``fn``, the value of ``arr[0]`` is ``3``. In HLSL, the array is passed by value,
so modifications inside ``fn`` do not propagate out.

.. note::

  DXC may pass unsized arrays directly as decayed pointers, which is an
  unfortunate behavior divergence.

Out Parameter Temporaries
-------------------------

.. code-block:: c++

  void Init(inout int X, inout int Y) {
    Y = 2;
    X = 1;
  }

  void main() {
    int V;
    Init(V, V); // MSVC (or clang-cl) V == 2, Clang V == 1
  }

In the above example the ``Init`` function's behavior depends on the C++
implementation. C++ does not define the order in which parameters are
initialized or destroyed. In MSVC and Clang's MSVC compatibility mode, arguments
are emitted right-to-left and destroyed left-to-right. This means that  the
parameter initialization and destruction occurs in the order: {``Y``, ``X``,
``~X``, ``~Y``}. This causes the write-back of the value of ``Y`` to occur last,
so the resulting value of ``V`` is ``2``. In the Itanium C++ ABI, the  parameter
ordering is reversed, so the initialization and destruction occurs in the order:
{``X``, ``Y``, ``~Y``, ``X``}. This causes the write-back of the value ``X`` to
occur last, resulting in the value of ``V`` being set to ``1``.

.. code-block:: c++

  void Trunc(inout int3 V) { }


  void main() {
    float3 F = {1.5, 2.6, 3.3};
    Trunc(F); // F == {1.0, 2.0, 3.0}
  }

In the above example, the argument expression ``F`` undergoes element-wise
conversion from a float vector to an integer vector to create a temporary
``int3``. On expiration the temporary undergoes elementwise conversion back to
the floating point vector type ``float3``. This results in an implicit
element-wise conversion of the vector even if the value is unused in the
function (effectively truncating the floating point values).


.. code-block:: c++

  void UB(out int X) {}

  void main() {
    int X = 7;
    UB(X); // X is undefined!
  }

In this example an initialized value is passed to an ``out`` parameter.
Parameters marked ``out`` are not initialized by the argument expression or
implicitly by the function. They must be explicitly initialized. In this case
the argument is not initialized in the function so the temporary is still
uninitialized when it is copied back to the argument expression. This is
undefined behavior in HLSL, and any use of the argument after the call is a use
of an undefined value which may be illegal in the target (DXIL programs with
used or potentially used ``undef`` or ``poison`` values fail validation).

Clang Implementation
====================

.. note::

  The implementation described here is a proposal. It has not yet been fully
  implemented, so the current state of Clang's sources may not reflect this
  design. A prototype implementation was built on DXC which is Clang-3.7 based.
  The prototype can be found
  `here <https://github.com/microsoft/DirectXShaderCompiler/pull/5249>`_. A lot
  of the changes in the prototype implementation are restoring Clang-3.7 code
  that was previously modified to its original state.

The implementation in clang adds a new non-decaying array type, a new AST node
to represent output parameters, and minor extensions to Clang's existing support
for Objective-C write-back arguments. The goal of this design is to capture the
semantic details of HLSL function calls in the AST, and minimize the amount of
magic that needs to occur during IR generation.

Array Temporaries
-----------------

The new ``ArrayParameterType`` is a sub-class of ``ConstantArrayType``
inheriting all the behaviors and methods of the parent except that it does not
decay to a pointer during overload resolution or template type deduction.

An argument of ``ConstantArrayType`` can be implicitly converted to an
equivalent non-decayed ``ArrayParameterType`` if the underlying canonical
``ConstantArrayType`` is the same. This occurs during overload resolution
instead of array to pointer decay.

.. code-block:: c++

  void SizedArray(float a[4]);
  void UnsizedArray(float a[]);

  void main() {
    float arr[4] = {1, 1, 1, 1};
    SizedArray(arr);
    UnsizedArray(arr);
  }

In the example above, the following AST is generated for the call to
``SizedArray``:

.. code-block:: text

  CallExpr 'void'
  |-ImplicitCastExpr 'void (*)(float [4])' <FunctionToPointerDecay>
  | `-DeclRefExpr 'void (float [4])' lvalue Function 'SizedArray' 'void (float [4])'
  `-ImplicitCastExpr 'float [4]' <HLSLArrayRValue>
    `-DeclRefExpr 'float [4]' lvalue Var 'arr' 'float [4]'

In the example above, the following AST is generated for the call to
``UnsizedArray``:

.. code-block:: text

  CallExpr 'void'
  |-ImplicitCastExpr 'void (*)(float [])' <FunctionToPointerDecay>
  | `-DeclRefExpr 'void (float [])' lvalue Function 'UnsizedArray' 'void (float [])'
  `-ImplicitCastExpr 'float [4]' <HLSLArrayRValue>
    `-DeclRefExpr 'float [4]' lvalue Var 'arr' 'float [4]'

In both of these cases the argument expression is of known array size so we can
initialize an appropriately sized temporary.

It is illegal in HLSL to convert an unsized array to a sized array:

.. code-block:: c++

  void SizedArray(float a[4]);
  void UnsizedArray(float a[]) {
    SizedArray(a); // Cannot convert float[] to float[4]
  }

When converting a sized array to an unsized array, an array temporary can also
be inserted. Given the following code:

.. code-block:: c++

  void UnsizedArray(float a[]);
  void SizedArray(float a[4]) {
    UnsizedArray(a);
  }

An expected AST should be something like:

.. code-block:: text

  CallExpr 'void'
  |-ImplicitCastExpr 'void (*)(float [])' <FunctionToPointerDecay>
  | `-DeclRefExpr 'void (float [])' lvalue Function 'UnsizedArray' 'void (float [])'
  `-ImplicitCastExpr 'float [4]' <HLSLArrayRValue>
    `-DeclRefExpr 'float [4]' lvalue Var 'arr' 'float [4]'

Out Parameter Temporaries
-------------------------

Output parameters are defined in HLSL as *casting expiring values* (cx-values),
which is a term made up for HLSL. A cx-value is a temporary value which may be
the result of a cast, and stores its value back to an lvalue when the value
expires.

To represent this concept in Clang we introduce a new ``HLSLOutArgExpr``. An
``HLSLOutArgExpr`` has three sub-expressions:

* An OpaqueValueExpr of the argument lvalue expression.
* An OpaqueValueExpr of the copy-initialized parameter temporary.
* A BinaryOpExpr assigning the first with the value of the second.

Given this example:

.. code-block:: c++

  void Init(inout int X) {
    X = 1;
  }

  void main() {
    int V;
    Init(V);
  }

The expected AST formulation for this code would be something like the example
below. Due to the nature of OpaqueValueExpr nodes, the nodes repeat in the AST
dump. The fake addresses ``0xSOURCE`` and ``0xTEMPORARY`` denote the source
lvalue and argument temporary lvalue expressions.

.. code-block:: text

  CallExpr 'void'
  |-ImplicitCastExpr 'void (*)(int &)' <FunctionToPointerDecay>
  | `-DeclRefExpr 'void (int &)' lvalue Function  'Init' 'void (int &)'
  `-HLSLOutArgExpr <col:10> 'int' lvalue inout
    |-OpaqueValueExpr 0xSOURCE <col:10> 'int' lvalue
    | `-DeclRefExpr <col:10> 'int' lvalue Var 'V' 'int'
    |-OpaqueValueExpr 0xTEMPORARY <col:10> 'int' lvalue
    | `-ImplicitCastExpr <col:10> 'int' <LValueToRValue>
    |   `-OpaqueValueExpr 0xSOURCE <col:10> 'int' lvalue
    |     `-DeclRefExpr <col:10> 'int' lvalue Var 'V' 'int'
    `-BinaryOperator <col:10> 'int' lvalue '='
      |-OpaqueValueExpr 0xSOURCE <col:10> 'int' lvalue
      | `-DeclRefExpr <col:10> 'int' lvalue Var 'V' 'int'
      `-ImplicitCastExpr <col:10> 'int' <LValueToRValue>
        `-OpaqueValueExpr 0xTEMPORARY <col:10> 'int' lvalue
          `-ImplicitCastExpr <col:10> 'int' <LValueToRValue>
            `-OpaqueValueExpr 0xSOURCE <col:10> 'int' lvalue
              `-DeclRefExpr <col:10> 'int' lvalue Var 'V' 'int'

The ``HLSLOutArgExpr`` captures that the value is ``inout`` vs ``out`` to
denote whether or not the temporary is initialized from the sub-expression.

The example below demonstrates argument casting:

.. code-block:: c++

  void Trunc(inout int3 V) { }


  void main() {
    float3 F = {1.5, 2.6, 3.3};
    Trunc(F);
  }

For this case the ``HLSLOutArgExpr`` will have sub-expressions to record both
casting expression sequences for the initialization and write back:

.. code-block:: text

  -CallExpr 'void'
    |-ImplicitCastExpr 'void (*)(int3 &)' <FunctionToPointerDecay>
    | `-DeclRefExpr 'void (int3 &)' lvalue Function 'inc_i32' 'void (int3 &)'
    `-HLSLOutArgExpr <col:11> 'int3':'vector<int, 3>' lvalue inout
      |-OpaqueValueExpr 0xSOURCE <col:11> 'float3':'vector<float, 3>' lvalue
      | `-DeclRefExpr <col:11> 'float3':'vector<float, 3>' lvalue Var 'F' 'float3':'vector<float, 3>'
      |-OpaqueValueExpr 0xTEMPORARY <col:11> 'int3':'vector<int, 3>' lvalue
      | `-ImplicitCastExpr <col:11> 'vector<int, 3>' <FloatingToIntegral>
      |   `-ImplicitCastExpr <col:11> 'float3':'vector<float, 3>' <LValueToRValue>
      |     `-OpaqueValueExpr 0xSOURCE <col:11> 'float3':'vector<float, 3>' lvalue
      |       `-DeclRefExpr <col:11> 'float3':'vector<float, 3>' lvalue Var 'F' 'float3':'vector<float, 3>'
      `-BinaryOperator <col:11> 'float3':'vector<float, 3>' lvalue '='
        |-OpaqueValueExpr 0xSOURCE <col:11> 'float3':'vector<float, 3>' lvalue
        | `-DeclRefExpr <col:11> 'float3':'vector<float, 3>' lvalue Var 'F' 'float3':'vector<float, 3>'
        `-ImplicitCastExpr <col:11> 'vector<float, 3>' <IntegralToFloating>
          `-ImplicitCastExpr <col:11> 'int3':'vector<int, 3>' <LValueToRValue>
            `-OpaqueValueExpr 0xTEMPORARY <col:11> 'int3':'vector<int, 3>' lvalue
              `-ImplicitCastExpr <col:11> 'vector<int, 3>' <FloatingToIntegral>
                `-ImplicitCastExpr <col:11> 'float3':'vector<float, 3>' <LValueToRValue>
                  `-OpaqueValueExpr 0xSOURCE <col:11> 'float3':'vector<float, 3>' lvalue
                    `-DeclRefExpr <col:11> 'float3':'vector<float, 3>' lvalue Var 'F' 'float3':'vector<float, 3>'

The AST representation is the same whether casting is required or not, which
simplifies the code generation. IR generation does the following:

* Emit the argument lvalue expression.
* Initialize the argument:
  * For ``inout`` arguments, emit the copy-initialization expression.
  * For ``out`` arguments, emit an uninitialized temporary.
* Emit the call
* Emit the write-back BinaryOperator expression.
