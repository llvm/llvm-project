====================
HLSL Root Signatures
====================

.. contents::
   :local:

Usage
=====

In HLSL, the `root signature
<https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures>`_
defines what types of resources are bound to the graphics pipeline.

A root signature can be specified in HLSL as a `string
<https://learn.microsoft.com/en-us/windows/win32/direct3d12/specifying-root-signatures-in-hlsl#an-example-hlsl-root-signature>`_.
The string contains a collection of comma-separated clauses that describe root
signature constituent components.

There are three mechanisms to compile an HLSL root signature. First, it is
possible to attach a root signature string to a particular shader via the
RootSignature attribute (in the following example, using the main entry
point):

.. code-block::

    #define RS "RootFlags( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
              "DENY_VERTEX_SHADER_ROOT_ACCESS), " \
              "CBV(b0, space = 1, flags = DATA_STATIC), " \
              "SRV(t0), " \
              "UAV(u0), " \
              "DescriptorTable( CBV(b1), " \
              "                 SRV(t1, numDescriptors = 8, " \
              "                     flags = DESCRIPTORS_VOLATILE), " \
              "                 UAV(u1, numDescriptors = unbounded, " \
              "                     flags = DESCRIPTORS_VOLATILE)), " \
              "DescriptorTable(Sampler(s0, space=1, numDescriptors = 4)), " \
              "RootConstants(num32BitConstants=3, b10), " \
              "StaticSampler(s1)," \
              "StaticSampler(s2, " \
              "              addressU = TEXTURE_ADDRESS_CLAMP, " \
              "              filter = FILTER_MIN_MAG_MIP_LINEAR )"

    [RootSignature(RS)]
    float4 main(float4 coord : COORD) : SV_Target
    {
    …
    }

The compiler will create and verify the root signature blob for the shader and
embed it alongside the shader byte code into the shader blob.

The second mechanism is using -rootsig-define option to specify macro define of
the root signature string.

.. code-block::

    #define RS "…"

    float4 main(float4 coord : COORD) : SV_Target
    {
    …
    }

Instead of adding the root signature attribute, the root signature macro define
is passed to the compiler with -rootsign-define.

.. code-block:: sh

  dxc.exe -T ps_6_0 MyRS1.hlsl -rootsig-define RS -Fo MyRS1.fxo


The third mechanism is to create a standalone root signature blob, perhaps to
reuse it with a large set of shaders, saving space. The name of the define
string is specified via the usual -E argument. For example:

.. code-block:: sh

  dxc.exe -T rootsig_1_1 MyRS1.hlsl -E MyRS1 -Fo MyRS1.fxo

Note that the root signature string define can also be passed on the command
line, e.g, -D MyRS1=”…”.

Root signature could also used in form of StateObject like this:

.. code-block::

  GlobalRootSignature MyGlobalRootSignature =
  {
      "DescriptorTable(UAV(u0)),"                     // Output texture
      "SRV(t0),"                                      // Acceleration structure
      "CBV(b0),"                                      // Scene constants
      "DescriptorTable(SRV(t1, numDescriptors = 2))"  // Static index and vertex buffers.
  };

  LocalRootSignature MyLocalRootSignature =
  {
      "RootConstants(num32BitConstants = 4, b1)"  // Cube constants
  };



GlobalRootSignature
===================

A GlobalRootSignature corresponds to a D3D12_GLOBAL_ROOT_SIGNATURE structure.

The fields consist of some number of strings describing the parts of the root signature.
The string should follow Root Signature Grammar.

.. code-block::

  GlobalRootSignature MyGlobalRootSignature =
  {
      "DescriptorTable(UAV(u0)),"                     // Output texture
      "SRV(t0),"                                      // Acceleration structure
      "CBV(b0),"                                      // Scene constants
      "DescriptorTable(SRV(t1, numDescriptors = 2))"  // Static index and vertex buffers
  };


LocalRootSignature
==================

A LocalRootSignature corresponds to a D3D12_LOCAL_ROOT_SIGNATURE structure.

Just like the global root signature subobject, the fields consist of some
number of strings describing the parts of the root signature.
The string should follow Root Signature Grammar.

.. code-block::

  LocalRootSignature MyLocalRootSignature =
  {
      "RootConstants(num32BitConstants = 4, b1)"  // Cube constants
  };


SubobjectToExportsAssociation
=============================

By default, a subobject merely declared in the same library as an export is
able to apply to that export.
However, applications have the ability to override that and get specific about
what subobject goes with which export. In HLSL, this "explicit association" is
done using SubobjectToExportsAssociation.

A SubobjectToExportsAssociation corresponds to a
D3D12_DXIL_SUBOBJECT_TO_EXPORTS_ASSOCIATION structure.

This subobject is declared with the syntax
.. code-block::

  SubobjectToExportsAssociation Name =
  {
      SubobjectName,
      Exports
  };

The local/global root signature in above example could be used like this:

.. code-block::

  SubobjectToExportsAssociation MyLocalRootSignatureAssociation =
  {
      "MyLocalRootSignature",    // Subobject name
      "MyHitGroup;MyMissShader"  // Exports association
  };
  SubobjectToExportsAssociation MyGlobalRootSignatureAssociation =
  {
      "MyGlobalRootSignature",    // Subobject name
      "MyHitGroup;MyMissShader"  // Exports association
  };


Root Signature Grammar
======================

.. code-block:: peg

    RootSignature : (RootElement(,RootElement)?)?

    RootElement : RootFlags | RootConstants | RootCBV | RootSRV | RootUAV |
                  DescriptorTable | StaticSampler

    RootFlags : 'RootFlags' '(' (RootFlag(|RootFlag)?)? ')'

    RootFlag : 'ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT' |
               'DENY_VERTEX_SHADER_ROOT_ACCESS'

    RootConstants : 'RootConstants' '(' 'num32BitConstants' '=' NUMBER ','
           bReg (',' 'space' '=' NUMBER)?
           (',' 'visibility' '=' SHADER_VISIBILITY)? ')'

    RootCBV : 'CBV' '(' bReg (',' 'space' '=' NUMBER)?
          (',' 'visibility' '=' SHADER_VISIBILITY)?
          (',' 'flags' '=' DATA_FLAGS)? ')'

    RootSRV : 'SRV' '(' tReg (',' 'space' '=' NUMBER)?
          (',' 'visibility' '=' SHADER_VISIBILITY)?
          (',' 'flags' '=' DATA_FLAGS)? ')'

    RootUAV : 'UAV' '(' uReg (',' 'space' '=' NUMBER)?
          (',' 'visibility' '=' SHADER_VISIBILITY)?
          (',' 'flags' '=' DATA_FLAGS)? ')'

    DescriptorTable : 'DescriptorTable' '(' (DTClause(|DTClause)?)?
          (',' 'visibility' '=' SHADER_VISIBILITY)? ')'

    DTClause : CBV | SRV | UAV | Sampler

    CBV : 'CBV' '(' bReg (',' 'numDescriptors' '=' NUMBER)?
          (',' 'space' '=' NUMBER)?
          (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)?
          (',' 'flags' '=' DATA_FLAGS)? ')'

    SRV : 'SRV' '(' tReg (',' 'numDescriptors' '=' NUMBER)?
    (',' 'space' '=' NUMBER)?
          (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)?
          (',' 'flags' '=' DATA_FLAGS)? ')'

    UAV : 'UAV' '(' uReg (',' 'numDescriptors' '=' NUMBER)?
          (',' 'space' '=' NUMBER)?
          (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)?
          (',' 'flags' '=' DATA_FLAGS)? ')'

    Sampler : 'Sampler' '(' sReg (',' 'numDescriptors' '=' NUMBER)?
          (',' 'space' '=' NUMBER)?
          (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)? (',' 'flags' '=' NUMBER)? ')'


    SHADER_VISIBILITY : 'SHADER_VISIBILITY_ALL' | 'SHADER_VISIBILITY_VERTEX' |
                        'SHADER_VISIBILITY_HULL' |
                        'SHADER_VISIBILITY_DOMAIN' |
                        'SHADER_VISIBILITY_GEOMETRY' |
                        'SHADER_VISIBILITY_PIXEL' |
                        'SHADER_VISIBILITY_AMPLIFICATION' |
                        'SHADER_VISIBILITY_MESH'

    DATA_FLAGS : 'DATA_STATIC_WHILE_SET_AT_EXECUTE' | 'DATA_VOLATILE'

    DESCRIPTOR_RANGE_OFFSET : 'DESCRIPTOR_RANGE_OFFSET_APPEND' | NUMBER

    StaticSampler : 'StaticSampler' '(' sReg (',' 'filter' '=' FILTER)?
             (',' 'addressU' '=' TEXTURE_ADDRESS)?
             (',' 'addressV' '=' TEXTURE_ADDRESS)?
             (',' 'addressW' '=' TEXTURE_ADDRESS)?
             (',' 'mipLODBias' '=' NUMBER)?
             (',' 'maxAnisotropy' '=' NUMBER)?
             (',' 'comparisonFunc' '=' COMPARISON_FUNC)?
             (',' 'borderColor' '=' STATIC_BORDER_COLOR)?
             (',' 'minLOD' '=' NUMBER)?
             (',' 'maxLOD' '=' NUMBER)? (',' 'space' '=' NUMBER)?
             (',' 'visibility' '=' SHADER_VISIBILITY)? ')'

    bReg : 'b' NUMBER

    tReg : 't' NUMBER

    uReg : 'u' NUMBER

    sReg : 's' NUMBER

    FILTER : 'FILTER_MIN_MAG_MIP_POINT' |
             'FILTER_MIN_MAG_POINT_MIP_LINEAR' |
             'FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT' |
             'FILTER_MIN_POINT_MAG_MIP_LINEAR' |
             'FILTER_MIN_LINEAR_MAG_MIP_POINT' |
             'FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR' |
             'FILTER_MIN_MAG_LINEAR_MIP_POINT' |
             'FILTER_MIN_MAG_MIP_LINEAR' |
             'FILTER_ANISOTROPIC' |
             'FILTER_COMPARISON_MIN_MAG_MIP_POINT' |
             'FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR' |
             'FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT' |
             'FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR' |
             'FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT' |
             'FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR' |
             'FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT' |
             'FILTER_COMPARISON_MIN_MAG_MIP_LINEAR' |
             'FILTER_COMPARISON_ANISOTROPIC' |
             'FILTER_MINIMUM_MIN_MAG_MIP_POINT' |
             'FILTER_MINIMUM_MIN_MAG_POINT_MIP_LINEAR' |
             'FILTER_MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT' |
             'FILTER_MINIMUM_MIN_POINT_MAG_MIP_LINEAR' |
             'FILTER_MINIMUM_MIN_LINEAR_MAG_MIP_POINT' |
             'FILTER_MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR' |
             'FILTER_MINIMUM_MIN_MAG_LINEAR_MIP_POINT' |
             'FILTER_MINIMUM_MIN_MAG_MIP_LINEAR' |
             'FILTER_MINIMUM_ANISOTROPIC' |
             'FILTER_MAXIMUM_MIN_MAG_MIP_POINT' |
             'FILTER_MAXIMUM_MIN_MAG_POINT_MIP_LINEAR' |
             'FILTER_MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT' |
             'FILTER_MAXIMUM_MIN_POINT_MAG_MIP_LINEAR' |
             'FILTER_MAXIMUM_MIN_LINEAR_MAG_MIP_POINT' |
             'FILTER_MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR' |
             'FILTER_MAXIMUM_MIN_MAG_LINEAR_MIP_POINT' |
             'FILTER_MAXIMUM_MIN_MAG_MIP_LINEAR' |
             'FILTER_MAXIMUM_ANISOTROPIC'

    TEXTURE_ADDRESS : 'TEXTURE_ADDRESS_WRAP' |
                      'TEXTURE_ADDRESS_MIRROR' | 'TEXTURE_ADDRESS_CLAMP' |
                      'TEXTURE_ADDRESS_BORDER' | 'TEXTURE_ADDRESS_MIRROR_ONCE'

    COMPARISON_FUNC : 'COMPARISON_NEVER' | 'COMPARISON_LESS' |
                      'COMPARISON_EQUAL' | 'COMPARISON_LESS_EQUAL' |
                      'COMPARISON_GREATER' | 'COMPARISON_NOT_EQUAL' |
                      'COMPARISON_GREATER_EQUAL' | 'COMPARISON_ALWAYS'

    STATIC_BORDER_COLOR : 'STATIC_BORDER_COLOR_TRANSPARENT_BLACK' |
                          'STATIC_BORDER_COLOR_OPAQUE_BLACK' |
                          'STATIC_BORDER_COLOR_OPAQUE_WHITE'
