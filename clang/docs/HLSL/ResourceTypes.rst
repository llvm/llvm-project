===================
HLSL Resource Types
===================

.. contents::
   :local:

Introduction
============

HLSL Resources are runtime-bound data that is provided as input, output or both
to shader programs written in HLSL. Resource Types in HLSL provide key user
abstractions for reading and writing resource data.

Implementation Details
======================

In Clang resource types are forward declared by the ``HLSLExternalSemaSource``
on initialization. They are then lazily completed when ``requiresCompleteType``
is called later in Sema.

Resource types are templated class declarations. The template parameter
specifies the expected return type of resource loads, and the expected parameter
type for stores.

In Clang's AST and code generation, resource types are classes that store a
pointer of the template parameter type. The pointer is populated from a call to
``__builtin_hlsl_create_handle``, and treated as a pointer to an array of typed
data through until lowering in the backend.

Resource types are annotated with the ``HLSLResource`` attribute, which drives
code generation for resource binding metadata. The ``hlsl`` metadata nodes are
transformed in the backend to the binding information expected by the target
runtime.
