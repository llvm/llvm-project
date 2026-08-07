===================
HLSL Resource Types
===================

.. contents::
   :local:

Introduction
============

HLSL Resources are runtime-bound data that is provided as input, output or both
to shader programs written in HLSL. These appear in HLSL source as instances of
special classes like `RWByteAddressBuffer`_, `ConstantBuffer`_, `Texture3D`_,
and `RasterizerOrderedTexture2D`_. They provide key user abstractions for
reading and writing resource data.

.. _RWByteAddressBuffer: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
.. _ConstantBuffer: https://learn.microsoft.com/en-us/windows/win32/direct3d12/resource-binding-in-hlsl#constant-buffers
.. _Texture3D: https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-texture3d
.. _RasterizerOrderedTexture2D: https://learn.microsoft.com/en-us/windows/win32/direct3d11/rasterizer-order-views

Implementation Details
======================

Clang's implementation of the HLSL resource types is designed to allow for a
future version where the individual classes are implemented directly as HLSL in
a library. However, this isn't possible today as they rely on some features,
such as constructors, that are explicitly disallowed in HLSL user code. Because
of this, these types are forward declared by the ``HLSLExternalSemaSource`` on
initialization. They are then lazily completed when ``requiresCompleteType`` is
called later in Sema.

A class is a resource if it contains a member of the ``__hlsl_resource_t``
type, which represents a "intangible resource handle". This resource handle
type is annotated with various attributes to describe what type of resource it
is and what can be done with it.

- ``hlsl::resource_class(C)``: Given ``C`` in ``"SRV"``, ``"UAV"``,
  ``"CBuffer"``, or ``"Sampler"``, mark the resource as a shader resource view,
  unordered access view, constant buffer, or sampler, respectively.
- ``hlsl::contained_type(T)``: Given a type ``T``, specify the type of objects
  contained in the resource.
- ``hlsl::dimension(K)``: Given ``K`` in ``"Unknown"``, ``"1D"``, ``"2D"``,
  ``"3D"``, or ``"Cube"``, mark the resource as having the given dimensions.
- ``hlsl::is_array``: Specify that the resource is an array of objects of the
  given dimensions.
- ``hlsl::raw_buffer``: Specify that the resource is accessed as a raw buffer,
  rather than following the typed buffer alignment and offset rules.
- ``hlsl::is_ms``: Specify that the resource is multisampled.
- ``hlsl::is_rov``: Specify that the resource is a rasterizer ordered view.
- ``hlsl::is_counter``: Specify that this is a counter associated with another
  resource.

Member functions of a resource class are generally fairly simple wrappers
around builtins that operate on the handle member.

During code generation resource types are lowered to target extension types in
IR. These types are target specific and differ between DXIL and SPIR-V
generation, providing the necessary information for the targets to generate
binding metadata for their respective target runtimes.
