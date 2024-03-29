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

There are two mechanisms to compile an HLSL root signature. First, it is 
possible to attach a root signature string to a particular shader via the 
RootSignature attribute (in the following example, using the MyRS1 entry 
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

The other mechanism is to create a standalone root signature blob, perhaps to 
reuse it with a large set of shaders, saving space. The name of the define 
string is specified via the usual -E argument. For example:

.. code-block:: sh

  dxc.exe -T rootsig_1_1 MyRS1.hlsl -E MyRS1 -Fo MyRS1.fxo

Note that the root signature string define can also be passed on the command 
line, e.g, -D MyRS1=”…”.


Life of a Root Signature
========================

The root signature in a compiler begins with a string in the root signature 
attribute and ends with a serialized root signature in the DXContainer.

To report errors as early as possible, root signature string parsing should 
occur in Sema. 
To prevent redundant work, this parsing should only be performed in Sema. 
Consequently, the parsed root signature information must be stored in the AST.

To ensure that a used resource has a binding in the root signature, this 
information should be accessible in the backend to facilitate early error 
reporting. 

Since only the binding information is necessary for validation, one 
approach could be to store the binding information separately for backend 
verification and serialize the root signature information promptly, as it 
constitutes the final output and is not utilized by any compiler passes.


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


Serialized format
======================
The root signature string is parsed and serialized into a binary format. The
binary format is a sequence of bytes that can be used to create a root signature
object in the Direct3D 12 API. The binary format is defined by the
`D3D12_ROOT_SIGNATURE_DESC (for rootsig_1_0)
<https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_signature_desc>`_
or `D3D12_ROOT_SIGNATURE_DESC1 (for rootsig_1_1)
<https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_signature_desc1>`_ 
structure in the Direct3D 12 API. (With the pointers translated to offsets.) 

It will be look like this:

.. code-block:: c++

  struct DxilContainerRootDescriptor1 {
    uint32_t ShaderRegister;
    uint32_t RegisterSpace;
    uint32_t Flags;
  };

  struct DxilContainerDescriptorRange {
    uint32_t RangeType;
    uint32_t NumDescriptors;
    uint32_t BaseShaderRegister;
    uint32_t RegisterSpace;
    uint32_t OffsetInDescriptorsFromTableStart;
  };

  struct DxilContainerDescriptorRange1 {
    uint32_t RangeType;
    uint32_t NumDescriptors;
    uint32_t BaseShaderRegister;
    uint32_t RegisterSpace;
    uint32_t Flags;
    uint32_t OffsetInDescriptorsFromTableStart;
  };

  struct DxilContainerRootDescriptorTable {
    uint32_t NumDescriptorRanges;
    uint32_t DescriptorRangesOffset;
  };

  struct DxilContainerRootParameter {
    uint32_t ParameterType;
    uint32_t ShaderVisibility;
    uint32_t PayloadOffset;
  };

  struct DxilContainerRootSignatureDesc {
    uint32_t Version;
    uint32_t NumParameters;
    uint32_t RootParametersOffset;
    uint32_t NumStaticSamplers;
    uint32_t StaticSamplersOffset;
    uint32_t Flags;
  };


The binary representation begins with a **DxilContainerRootSignatureDesc** 
object. 

The object will be followed by an array of 
**DxilContainerRootParameter/Parameter1** objects located at 
**DxilContainerRootSignatureDesc::RootParametersOffset**, which corresponds to 
the size of **DxilContainerRootSignatureDesc**.

Subsequently, there will be detailed object (**DxilRootConstants**, 
**DxilContainerRootDescriptorTable**, or 
**DxilRootDescriptor/DxilContainerRootDescriptor1**, depending on the parameter 
type) for each **DxilContainerRootParameter** in the array. With 
**DxilContainerRootParameter.PayloadOffset** pointing to the detailed object.

In cases where the detailed object is a **DxilContainerRootDescriptorTable**, 
it is succeeded by an array of 
**DxilContainerDescriptorRange/DxilContainerDescriptorRange1** at 
**DxilContainerRootDescriptorTable.DescriptorRangesOffset**.

The binary representation is finalized with an array of 
**DxilStaticSamplerDesc** at 
**DxilContainerRootSignatureDesc::StaticSamplersOffset**.

Implementation Details
======================

The root signature string will be parsed in Clang. 
The parsing 
will happened when build HLSLRootSignatureAttr or when build standalone root 
signature blob. 

The root signature parsing will generate a HLSLRootSignatureAttr with member 
represents the root signature string and the parsed information for each 
resource in the root signature. It will bind to the entry function in the AST. 
HLSLRootSignatureAttr will be something like this:
Note, VersionedRootSignatureDesc is not D3D12_ROOT_SIGNATURE_DESC, it is just 
a simple struct to collect all the information in the root signature string.

.. code-block:: c++

    struct DescriptorRange {
      DescriptorRangeType RangeType;
      uint32_t NumDescriptors = 1;
      uint32_t BaseShaderRegister;
      uint32_t RegisterSpace = 0;
      DescriptorRangeFlags Flags = DescriptorRangeFlags::None;
      uint32_t OffsetInDescriptorsFromTableStart = DescriptorRangeOffsetAppend;
    };

    struct RootDescriptorTable {
      std::vector<DescriptorRange> DescriptorRanges;
    };
    struct RootConstants {
      uint32_t ShaderRegister;
      uint32_t RegisterSpace = 0;
      uint32_t Num32BitValues;
    };

    struct RootDescriptor {
      uint32_t ShaderRegister;
      uint32_t RegisterSpace = 0;
      RootDescriptorFlags Flags = RootDescriptorFlags::None;
    };
    struct RootParameter {
      RootParameterType ParameterType;
      std::variant<RootDescriptorTable, RootConstants, RootDescriptor>
            Parameter;
      ShaderVisibility ShaderVisibility = ShaderVisibility::All;
    };

    struct StaticSamplerDesc {
      Filter Filter = Filter::ANISOTROPIC;
      TextureAddressMode AddressU = TextureAddressMode::Wrap;
      TextureAddressMode AddressV = TextureAddressMode::Wrap;
      TextureAddressMode AddressW = TextureAddressMode::Wrap;
      float MipLODBias = 0.f;
      uint32_t MaxAnisotropy = 16;
      ComparisonFunc ComparisonFunc = ComparisonFunc::LessEqual;
      StaticBorderColor BorderColor = StaticBorderColor::OpaqueWhite;
      float MinLOD = 0.f;
      float MaxLOD = MaxLOD;
      uint32_t ShaderRegister;
      uint32_t RegisterSpace = 0;
      ShaderVisibility ShaderVisibility = ShaderVisibility::All;
    };

    struct RootSignatureDesc {
      std::vector<RootParameter> Parameters;
      std::vector<StaticSamplerDesc> StaticSamplers;
      RootSignatureFlags Flags;
    };

    enum class RootSignatureVersion {
      Version_1 = 1,
      Version_1_0 = 1,
      Version_1_1 = 2
    };

    struct VersionedRootSignatureDesc {
      RootSignatureVersion Version;
      RootSignatureDesc Desc;
    };

    class HLSLRootSignatureAttr : public InheritableAttr {
    protected:
      std::string RootSignatureStr;
      VersionedRootSignatureDesc RootSignature;
    };


.. code-block::

    def HLSLEntryRootSignature: HLSLRootSignatureAttr {
      let Spellings = [GNU<"RootSignature">];
      let Subjects = Subjects<[HLSLEntry]>;
      let LangOpts = [HLSL];
      let Args = [StringArgument<"InputString">];
    }

For case compile to a standalone root signature blob, the 
HLSLRootSignatureAttr will be bind to a fake empty entry.

In clang code generation, the HLSLRootSignatureAttr in AST will be translated 
into a global variable with struct type to express the layout and metadata to 
save things like static sampler, root flags, space and NumDescriptors in LLVM IR. 
The struct type will be look like this:

.. code-block:: llvm

  %struct.TABLE0 = type { target("dx.rs.desc"),
                          target("dx.rs.sampler")}

  %struct.RS = type { target("dx.rs.rootconstant", 4), 
                      %struct.TABLE0,
                      target("dx.rs.rootdescriptor") }

The metadata will be look like this:

.. code-block::

  !1 = !{ data for static sampler } ; Save informations for single static 
                                    ; sampler
  !2 = !{!1} ; All static samplers
  !3 = !{ data for descriptors } ; Save informations for single descriptor 
  !4 = !{ !3 } ; All descriptors
  !5 = !{void ()* @main, %struct.RS undef, i32 rootFlags, !2, !4}


CGHLSLRuntime will generate metadata to link the global variable as root 
signature for given entry function. 

In LLVM DirectX backend, the global variable will be serialized and saved as 
another global variable with section 'RTS0' with the serialized root signature 
as initializer in DXContainerGlobals pass. The serialized root signature is in 
exactly the format it will be written out to the DXContainer object.
The MC ObjectWriter for DXContainer will take the global and write it to the 
correct part based on the section name given to the global.

In DXIL validation for DXC, the root signature part will be deserialized and 
check if resource used in the shader (the information is in pipeline state 
validation part) exists in the root signature. 
For LLVM DirectX backend, this could be done in IR pass before emit DXIL 
instead of validation.

Same check could be done in Sema as well. But at AST level, it is impossible 
to identify unused resource which will be removed later. And the resource 
binding allocation is not done. 
So the only case could be caught in Sema is for resources that are known to be 
used for sure (like resources used in entry function and not under any control 
flow) and binded by user. 
If the resource is not in root signature, error should be reported in Sema.
