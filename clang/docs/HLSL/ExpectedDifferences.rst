===================================
Expected Differences vs DXC and FXC
===================================

.. contents::
   :local:

Introduction
============

HLSL currently has two reference compilers, the `DirectX Shader Compiler (DXC)
<https://github.com/microsoft/DirectXShaderCompiler/>`_ and the
`Effect-Compiler (FXC) <https://learn.microsoft.com/en-us/windows/win32/direct3dtools/fxc>`_.
The two reference compilers do not fully agree. Some known disagreements in the
references are tracked on
`DXC's GitHub
<https://github.com/microsoft/DirectXShaderCompiler/issues?q=is%3Aopen+is%3Aissue+label%3Afxc-disagrees>`_,
but many more are known to exist.

HLSL as implemented by Clang will also not fully match either of the reference
implementations, it is instead being written to match the `draft language
specification <https://microsoft.github.io/hlsl-specs/specs/hlsl.pdf>`_.

This document is a non-exhaustive collection the known differences between
Clang's implementation of HLSL and the existing reference compilers.

General Principles
------------------

Most of the intended differences between Clang and the earlier reference
compilers are focused on increased consistency and correctness. Both reference
compilers do not always apply language rules the same in all contexts.

Clang also deviates from the reference compilers by providing different
diagnostics, both in terms of the textual messages and the contexts in which
diagnostics are produced. While striving for a high level of source
compatibility with conforming HLSL code, Clang may produce earlier and more
robust diagnostics for incorrect code or reject code that a reference compiler
incorrectly accepted.

Language Version
================

Clang targets language compatibility for HLSL 2021 as implemented by DXC.
Language features that were removed in earlier versions of HLSL may be added on
a case-by-case basis, but are not planned for the initial implementation.

Overload Resolution
===================

Clang's HLSL implementation adopts C++ overload resolution rules as proposed for
HLSL 202x based on proposal
`0007 <https://github.com/microsoft/hlsl-specs/blob/main/proposals/0007-const-instance-methods.md>`_
and
`0008 <https://github.com/microsoft/hlsl-specs/blob/main/proposals/0008-non-member-operator-overloading.md>`_.

The largest difference between Clang and DXC's overload resolution is the
algorithm used for identifying best-match overloads. There are more details
about the algorithmic differences in the :ref:`multi_argument_overloads` section
below. There are three high level differences that should be highlighted:

* **There should be no cases** where DXC and Clang both successfully
  resolve an overload where the resolved overload is different between the two.
* There are cases where Clang will successfully resolve an overload that DXC
  wouldn't because we've trimmed the overload set in Clang to remove ambiguity.
* There are cases where DXC will successfully resolve an overload that Clang
  will not for two reasons: (1) DXC only generates partial overload sets for
  builtin functions and (2) DXC resolves cases that probably should be ambiguous.

Clang's implementation extends standard overload resolution rules to HLSL
library functionality. This causes subtle changes in overload resolution
behavior between Clang and DXC. Some examples include:

.. code-block:: c++

  void halfOrInt16(half H);
  void halfOrInt16(uint16_t U);
  void halfOrInt16(int16_t I);

  void takesDoubles(double, double, double);

  cbuffer CB {
    bool B;
    uint U;
    int I;
    float X, Y, Z;
    double3 R, G;
  }

  void takesSingleDouble(double);
  void takesSingleDouble(vector<double, 1>);

  void scalarOrVector(double);
  void scalarOrVector(vector<double, 2>);

  export void call() {
    half H;
    halfOrInt16(I); // All: Resolves to halfOrInt16(int16_t).

  #ifndef IGNORE_ERRORS
    halfOrInt16(U); // All: Fails with call ambiguous between int16_t and uint16_t
                    // overloads

    // asfloat16 is a builtin with overloads for half, int16_t, and uint16_t.
    H = asfloat16(I); // DXC: Fails to resolve overload for int.
                      // Clang: Resolves to asfloat16(int16_t).
    H = asfloat16(U); // DXC: Fails to resolve overload for int.
                      // Clang: Resolves to asfloat16(uint16_t).
  #endif
    H = asfloat16(0x01); // DXC: Resolves to asfloat16(half).
                         // Clang: Resolves to asfloat16(uint16_t).

    takesDoubles(X, Y, Z); // Works on all compilers
  #ifndef IGNORE_ERRORS
    fma(X, Y, Z); // DXC: Fails to resolve no known conversion from float to
                  //   double.
                  // Clang: Resolves to fma(double,double,double).

    double D = dot(R, G); // DXC: Resolves to dot(double3, double3), fails DXIL Validation.
                          // FXC: Expands to compute double dot product with fmul/fadd
                          // Clang: Fails to resolve as ambiguous against
                          //   dot(half, half) or dot(float, float)
  #endif

  #ifndef IGNORE_ERRORS
    tan(B); // DXC: resolves to tan(float).
            // Clang: Fails to resolve, ambiguous between integer types.

  #endif

    double D;
    takesSingleDouble(D); // All: Fails to resolve ambiguous conversions.
    takesSingleDouble(R); // All: Fails to resolve ambiguous conversions.

    scalarOrVector(D); // All: Resolves to scalarOrVector(double).
    scalarOrVector(R); // All: Fails to resolve ambiguous conversions.
  }

.. note::

  In Clang, a conscious decision was made to exclude the ``dot(vector<double,N>, vector<double,N>)``
  overload and allow overload resolution to resolve the
  ``vector<float,N>`` overload. This approach provides ``-Wconversion``
  diagnostic notifying the user of the conversion rather than silently altering
  precision relative to the other overloads (as FXC does) or generating code
  that will fail validation (as DXC does).

.. _multi_argument_overloads:

Multi-Argument Overloads
------------------------

In addition to the differences in single-element conversions, Clang and DXC
differ dramatically in multi-argument overload resolution. C++ multi-argument
overload resolution behavior (or something very similar) is required to
implement
`non-member operator overloading <https://github.com/microsoft/hlsl-specs/blob/main/proposals/0008-non-member-operator-overloading.md>`_.

Clang adopts the C++ inspired language from the
`draft HLSL specification <https://microsoft.github.io/hlsl-specs/specs/hlsl.pdf>`_,
where an overload ``f1`` is a better candidate than ``f2`` if for all arguments the
conversion sequences is not worse than the corresponding conversion sequence and
for at least one argument it is better.

.. code-block:: c++

  cbuffer CB {
    int I;
    float X;
    float4 V;
  }

  void twoParams(int, int);
  void twoParams(float, float);
  void threeParams(float, float, float);
  void threeParams(float4, float4, float4);

  export void call() {
    twoParams(I, X); // DXC: resolves twoParams(int, int).
                     // Clang: Fails to resolve ambiguous conversions.

    threeParams(X, V, V); // DXC: resolves threeParams(float4, float4, float4).
                          // Clang: Fails to resolve ambiguous conversions.
  }

For the examples above since ``twoParams`` called with mixed parameters produces
implicit conversion sequences that are { ExactMatch, FloatingIntegral }  and {
FloatingIntegral, ExactMatch }. In both cases an argument has a worse conversion
in the other sequence, so the overload is ambiguous.

In the ``threeParams`` example the sequences are { ExactMatch, VectorTruncation,
VectorTruncation } or { VectorSplat, ExactMatch, ExactMatch }, again in both
cases at least one parameter has a worse conversion in the other sequence, so
the overload is ambiguous.

.. note::

  The behavior of DXC documented below is undocumented so this is gleaned from
  observation and a bit of reading the source.

DXC's approach for determining the best overload produces an integer score value
for each implicit conversion sequence for each argument expression. Scores for
casts are based on a bitmask construction that is complicated to reverse
engineer. It seems that:

* Exact match is 0
* Dimension increase is 1
* Promotion is 2
* Integral -> Float conversion is 4
* Float -> Integral conversion is 8
* Cast is 16

The masks are or'd against each other to produce a score for the cast.

The scores of each conversion sequence are then summed to generate a score for
the overload candidate. The overload candidate with the lowest score is the best
candidate. If more than one overload are matched for the lowest score the call
is ambiguous.
