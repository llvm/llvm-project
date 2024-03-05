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

.. code-block:: c++

    [RootSignature(MyRS1)]
    float4 main(float4 coord : COORD) : SV_Target
    {
    …
    }

The compiler will create and verify the root signature blob for the shader and 
embed it alongside the shader byte code into the shader blob. 

The other mechanism is to create a standalone root signature blob, perhaps to 
reuse it with a large set of shaders, saving space. The name of the define 
string is specified via the usual -E argument. For example:

.. code-block:: c++
  dxc.exe -T rootsig_1_1 MyRS1.hlsl -E MyRS1 -Fo MyRS1.fxo

Note that the root signature string define can also be passed on the command 
line, e.g, -D MyRS1=”…”.

Root Signature Grammar
======================

.. code-block:: c++

    RootSignature : (RootElement(,RootElement)?)?

    RootElement : RootFlags | RootConstants | RootCBV | RootSRV | RootUAV | DescriptorTable | StaticSampler

    RootFlags : 'RootFlags' '(' (RootFlag(|RootFlag)?)? ')'

    RootFlag : 'ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT' | 'DENY_VERTEX_SHADER_ROOT_ACCESS'

    RootConstants : 'RootConstants' '(' 'num32BitConstants' '=' NUMBER ',' bReg (',' 'space' '=' NUMBER)? (',' 'visibility' '=' SHADER_VISIBILITY)? ')'

    RootCBV : 'CBV' '(' bReg (',' 'space' '=' NUMBER)? (',' 'visibility' '=' SHADER_VISIBILITY)? (',' 'flags' '=' DATA_FLAGS)? ')'

    RootSRV : 'SRV' '(' tReg (',' 'space' '=' NUMBER)? (',' 'visibility' '=' SHADER_VISIBILITY)? (',' 'flags' '=' DATA_FLAGS)? ')'

    RootUAV : 'UAV' '(' uReg (',' 'space' '=' NUMBER)? (',' 'visibility' '=' SHADER_VISIBILITY)? (',' 'flags' '=' DATA_FLAGS)? ')'

    DescriptorTable : 'DescriptorTable' '(' (DTClause(|DTClause)?)? (',' 'visibility' '=' SHADER_VISIBILITY)? ')'

    DTClause : CBV | SRV | UAV | Sampler

    CBV : 'CBV' '(' bReg (',' 'numDescriptors' '=' NUMBER)? (',' 'space' '=' NUMBER)? (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)? (',' 'flags' '=' DATA_FLAGS)? ')'

    SRV : 'SRV' '(' tReg (',' 'numDescriptors' '=' NUMBER)? (',' 'space' '=' NUMBER)? (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)? (',' 'flags' '=' DATA_FLAGS)? ')'

    UAV : 'UAV' '(' uReg (',' 'numDescriptors' '=' NUMBER)? (',' 'space' '=' NUMBER)? (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)? (',' 'flags' '=' DATA_FLAGS)? ')'

    Sampler : 'Sampler' '(' sReg (',' 'numDescriptors' '=' NUMBER)? (',' 'space' '=' NUMBER)? (',' 'offset' '=' DESCRIPTOR_RANGE_OFFSET)? (',' 'flags' '=' NUMBER)? ')'


    SHADER_VISIBILITY : 'SHADER_VISIBILITY_ALL' | 'SHADER_VISIBILITY_VERTEX' | 'SHADER_VISIBILITY_HULL' | 'SHADER_VISIBILITY_DOMAIN' | 'SHADER_VISIBILITY_GEOMETRY' | 'SHADER_VISIBILITY_PIXEL' | 'SHADER_VISIBILITY_AMPLIFICATION' | 'SHADER_VISIBILITY_MESH'

    DATA_FLAGS : 'DATA_STATIC_WHILE_SET_AT_EXECUTE' | 'DATA_VOLATILE'

    DESCRIPTOR_RANGE_OFFSET : 'DESCRIPTOR_RANGE_OFFSET_APPEND' | NUMBER

    StaticSampler : 'StaticSampler' '(' sReg (',' 'filter' '=' FILTER)? (',' 'addressU' '=' TEXTURE_ADDRESS)? (',' 'addressV' '=' TEXTURE_ADDRESS)? (',' 'addressW' '=' TEXTURE_ADDRESS)? (',' 'mipLODBias' '=' NUMBER)? (',' 'maxAnisotropy' '=' NUMBER)? (',' 'comparisonFunc' '=' COMPARISON_FUNC)? (',' 'borderColor' '=' STATIC_BORDER_COLOR)? (',' 'minLOD' '=' NUMBER)? (',' 'maxLOD' '=' NUMBER)? (',' 'space' '=' NUMBER)? (',' 'visibility' '=' SHADER_VISIBILITY)? ')'

    bReg : 'b' NUMBER 

    tReg : 't' NUMBER 

    uReg : 'u' NUMBER 

    sReg : 's' NUMBER 

    FILTER : 'FILTER_MIN_MAG_MIP_POINT' | 'FILTER_MIN_MAG_POINT_MIP_LINEAR' | 'FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT' | 'FILTER_MIN_POINT_MAG_MIP_LINEAR' | 'FILTER_MIN_LINEAR_MAG_MIP_POINT' | 'FILTER_MIN_LINEAR_MAG_POINT_MIP_LINEAR' | 'FILTER_MIN_MAG_LINEAR_MIP_POINT' | 'FILTER_MIN_MAG_MIP_LINEAR' | 'FILTER_ANISOTROPIC' | 'FILTER_COMPARISON_MIN_MAG_MIP_POINT' | 'FILTER_COMPARISON_MIN_MAG_POINT_MIP_LINEAR' | 'FILTER_COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT' | 'FILTER_COMPARISON_MIN_POINT_MAG_MIP_LINEAR' | 'FILTER_COMPARISON_MIN_LINEAR_MAG_MIP_POINT' | 'FILTER_COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR' | 'FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT' | 'FILTER_COMPARISON_MIN_MAG_MIP_LINEAR' | 'FILTER_COMPARISON_ANISOTROPIC' | 'FILTER_MINIMUM_MIN_MAG_MIP_POINT' | 'FILTER_MINIMUM_MIN_MAG_POINT_MIP_LINEAR' | 'FILTER_MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT' | 'FILTER_MINIMUM_MIN_POINT_MAG_MIP_LINEAR' | 'FILTER_MINIMUM_MIN_LINEAR_MAG_MIP_POINT' | 'FILTER_MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR' | 'FILTER_MINIMUM_MIN_MAG_LINEAR_MIP_POINT' | 'FILTER_MINIMUM_MIN_MAG_MIP_LINEAR' | 'FILTER_MINIMUM_ANISOTROPIC' | 'FILTER_MAXIMUM_MIN_MAG_MIP_POINT' | 'FILTER_MAXIMUM_MIN_MAG_POINT_MIP_LINEAR' | 'FILTER_MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT' | 'FILTER_MAXIMUM_MIN_POINT_MAG_MIP_LINEAR' | 'FILTER_MAXIMUM_MIN_LINEAR_MAG_MIP_POINT' | 'FILTER_MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR' | 'FILTER_MAXIMUM_MIN_MAG_LINEAR_MIP_POINT' | 'FILTER_MAXIMUM_MIN_MAG_MIP_LINEAR' | 'FILTER_MAXIMUM_ANISOTROPIC'

    TEXTURE_ADDRESS : 'TEXTURE_ADDRESS_WRAP' | 'TEXTURE_ADDRESS_MIRROR' | 'TEXTURE_ADDRESS_CLAMP' | 'TEXTURE_ADDRESS_BORDER' | 'TEXTURE_ADDRESS_MIRROR_ONCE'

    COMPARISON_FUNC : 'COMPARISON_NEVER' | 'COMPARISON_LESS' | 'COMPARISON_EQUAL' | 'COMPARISON_LESS_EQUAL' | 'COMPARISON_GREATER' | 'COMPARISON_NOT_EQUAL' | 'COMPARISON_GREATER_EQUAL' | 'COMPARISON_ALWAYS'

    STATIC_BORDER_COLOR : 'STATIC_BORDER_COLOR_TRANSPARENT_BLACK' | 'STATIC_BORDER_COLOR_OPAQUE_BLACK' | 'STATIC_BORDER_COLOR_OPAQUE_WHITE'


Serialized format
======================
The root signature string is parsed and serialized into a binary format. The
binary format is a sequence of bytes that can be used to create a root signature
object in the Direct3D 12 API. The binary format is defined by the
`D3D12_ROOT_SIGNATURE_DESC (for rootsig_1_0)
<https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_signature_desc>`_
or `D3D12_ROOT_SIGNATURE_DESC1 (for rootsig_1_1)
<https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_root_signature_desc1>`_ 
structure in the Direct3D 12 API.



Implementation Details
======================

The root signature string will be parsed in the HLSL frontend. 
The parsing 
will happened when build HLSLRootSignatureAttr or when build standalone root 
signature blob. 

The root signature parsing will generate a VersionedRootSignatureDesc object 
that represents the root signature string. 
VersionedRootSignatureDesc is a struct that contains a RootSignatureVersion 
and a RootSignatureDesc.

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

    struct VersionedRootSignatureDesc {
    RootSignatureVersion Version;
    RootSignatureDesc Desc;
    };

Things like DescriptorRangeType and RootDescriptorFlags will be enums.

After parsing, the VersionedRootSignatureDesc will be translated into a 
constant global variable in the clang AST and save to the 
HLSLRootSignatureAttr. 

For case compile to a standalone root signature blob, the global variable will 
be saved in the ASTContext.

The global variable in AST will have a struct type that represents the root signature
layout and a initializer that contains the values like space and 
numDescriptors of the root signature.

In clang code generation, the global variable in AST will be translated into a global 
variable with cosntant initializer in LLVM IR. 

CGHLSLRuntime will generate metadata to link the global variable as root 
signature for given entry function or just nullptr for the standalone root 
signature blob case. 

In LLVM DirectX backend, the global variable will be serialized and save into the root
signature part of dx container when emit DXIL.
