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

    [RootSignature(MyRS)]
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

  namespace dxbc {
    namespace SerializedRootSignature {
      namespace v_1_0 {
      
        struct DxilContainerDescriptorRange {
          uint32_t RangeType;
          uint32_t NumDescriptors;
          uint32_t BaseShaderRegister;
          uint32_t RegisterSpace;
          uint32_t OffsetInDescriptorsFromTableStart;
        };

        struct ContainerRootDescriptor {
          uint32_t ShaderRegister;
          uint32_t RegisterSpace;
        };
      }
      namespace v_1_1 {

        struct ContainerDescriptorRange {
          uint32_t RangeType;
          uint32_t NumDescriptors;
          uint32_t BaseShaderRegister;
          uint32_t RegisterSpace;
          uint32_t Flags;
          uint32_t OffsetInDescriptorsFromTableStart;
        };

        struct ContainerRootDescriptor {
          uint32_t ShaderRegister;
          uint32_t RegisterSpace;
          uint32_t Flags;
        };
      }

      struct ContainerRootDescriptorTable {
        uint32_t NumDescriptorRanges;
        uint32_t DescriptorRangesOffset;
      };

      struct RootConstants {
        uint32_t ShaderRegister;
        uint32_t RegisterSpace = 0;
        uint32_t Num32BitValues;
      };

      struct ContainerRootParameter {
        uint32_t ParameterType;
        uint32_t ShaderVisibility;
        uint32_t PayloadOffset;
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

      struct ContainerRootSignatureDesc {
        uint32_t Version;
        uint32_t NumParameters;
        uint32_t RootParametersOffset;
        uint32_t NumStaticSamplers;
        uint32_t StaticSamplersOffset;
        uint32_t Flags;
      };
    }
  }


The binary representation begins with a **DxilContainerRootSignatureDesc** 
object. 

The object will be followed by an array of 
**DxilContainerRootParameter/Parameter1** objects located at 
**DxilContainerRootSignatureDesc::RootParametersOffset**, which corresponds to 
the size of **DxilContainerRootSignatureDesc**.

Then it will be an array of 
**DxilStaticSamplerDesc** at 
**DxilContainerRootSignatureDesc::StaticSamplersOffset**.

Subsequently, there will be detailed object (**DxilRootConstants**, 
**DxilContainerRootDescriptorTable**, or 
**DxilRootDescriptor/DxilContainerRootDescriptor1**, depending on the parameter 
type) for each **DxilContainerRootParameter** in the array. With 
**DxilContainerRootParameter.PayloadOffset** pointing to the detailed object.

In cases where the detailed object is a **DxilContainerRootDescriptorTable**, 
it is succeeded by an array of 
**DxilContainerDescriptorRange/DxilContainerDescriptorRange1** at 
**DxilContainerRootDescriptorTable.DescriptorRangesOffset**.

The layout could be look like this for the MyRS in above example:
.. code-block:: c++

  struct SerializedRS {
    ContainerRootSignatureDesc RSDesc;
    ContainerRootParameter  rootParameters[6];
    StaticSamplerDesc  samplers[2];

    // Extra part for each RootParameter in rootParameters.

    // RootConstants/RootDescriptorTable/RootDescriptor dependent on ParameterType.
    // For RootDescriptorTable, the extra part will be like

    // struct {

    //   RootDescriptorTable table;

    //   ContainerDescriptorRange ranges[NumDescriptorRangesForTheTable];

    // };

    struct {

      RootConstants b0;

      ContainerRootDescriptor t1;

      ContainerRootDescriptor u1;

      struct {

        ContainerRootDescriptor tab0;

        ContainerDescriptorRange tab0Ranges[2];

      } table0;

      struct {

        ContainerRootDescriptor tab1;

        ContainerDescriptorRange tab1Ranges[1];

      } table1;

      RootConstants b10;

    };

  };


Life of a Root Signature
========================

The root signature in a compiler begins with a string in the root signature 
attribute and ends with a serialized root signature in the DXContainer.

To report errors as early as possible, root signature string parsing should 
occur in Sema.

To ensure that a used resource has a binding in the root signature, this 
information should be accessible in the backend to make sure legal dxil is 
generated.


Implementation Details
======================

The root signature string will be parsed in Clang. 
The parsing 
will happened when build HLSLRootSignatureAttr, standalone root signature blob 
or local/global root signature.

A new AST node HLSLRootSignatureDecl will be added to represent the root 
signature in the AST.

Assuming the cost to parse root signature string is cheap,
HLSLRootSignatureDecl will follow common Clang approach which only save the 
string in the AST.
A root signature will be parsed twice, once in Sema for diagnostic and once in 
clang code generation for generating the serialized root signature.
The first parsing will check register overlap and run on all root signatures 
in the translation unit.
The second parsing will calculate the offset for serialized root signature and 
only run on the root signature for the entry function or local/global root 
signature which used in SubobjectToExportsAssociation.

HLSLRootSignatureDecl will have method to parse the string for diagnostic and 
generate the SerializedRS mentioned above.

A HLSLRootSignatureAttr will be created when meet RootSignature attribute in 
HLSL. 
It will defined as below.

.. code-block::

    def HLSLEntryRootSignature: InheriableAttr {
      let Spellings = [GNU<"RootSignature">];
      let Subjects = Subjects<[HLSLEntry]>;
      let LangOpts = [HLSL];
      let Args = [StringArgument<"InputString">, DeclArgument<HLSLRootSignature, "RootSignature", 0, /*fake*/ 1>];
    }

Because the RootSignature attribute in hlsl only have the string, it is easier
to add a StringArgument to save the string first.
A HLSLRootSignatureDecl will be created for diagnostic and saved to the 
HLSLEntryRootSignatureAttr for clang code generation. 
The HLSLRootSignatureDecl will save StringLiteral instead StringRef for diagnostic.

The AST for the attribute will be like this:

.. code-block::

    HLSLEntryRootSignatureAttr 
      "RootFlags( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | DENY_VERTEX_SHADER_ROOT_ACCESS), 
       CBV(b0, space = 1, flags = DATA_STATIC), SRV(t0), UAV(u0), 
       DescriptorTable( CBV(b1), SRV(t1, numDescriptors = 8,flags = DESCRIPTORS_VOLATILE), 
       UAV(u1, numDescriptors = unbounded, flags = DESCRIPTORS_VOLATILE)), 
       DescriptorTable(Sampler(s0, space=1, numDescriptors = 4)), 
       RootConstants(num32BitConstants=3, b10), 
       StaticSampler(s1),
       StaticSampler(s2, addressU = TEXTURE_ADDRESS_CLAMP, filter = FILTER_MIN_MAG_MIP_LINEAR )" 
       HLSLRootSignature 'main.RS'

For case compile to a standalone root signature blob, the 
HLSLRootSignatureAttr will be bind to a fake empty entry.

In clang code generation, the HLSLRootSignatureAttr in AST will be translated 
into a global variable with struct type to express the layout and metadata to 
save things like static sampler, root flags, space and NumDescriptors in LLVM IR. 
The global variable will be look like this for the example above:

.. code-block:: llvm

  ; %0 is the type for the whole serialzed root signature.
  %0 = type { %"dxbc::RootSignature::ContainerRootSignatureDesc", 
              [6 x %"dxbc::RootSignature::ContainerRootParameter"], 
              [2 x %"dxbc::RootSignature::StaticSamplerDesc"], 
              %1 }
  ; %1 is the type for extra parameter information for each RootParameter in rootParameters.
  %1 = type { %"dxbc::RootSignature::ContainerRootDescriptor", 
              %"dxbc::RootSignature::ContainerRootDescriptor", 
              %"dxbc::RootSignature::ContainerRootDescriptor", 
              %2, 
              %3, 
              %"dxbc::RootSignature::ContainerRootConstants" }
  ; %2 is the type for first DescriptorTable in RootParameter.
  %2 = type { %"dxbc::RootSignature::ContainerRootDescriptorTable", [3 x %"dxbc::RootSignature::ContainerDescriptorRange"] }
  ; %3 is the type for second DescriptorTable in RootParameter.
  %3 = type { %"dxbc::RootSignature::ContainerRootDescriptorTable", [1 x %"dxbc::RootSignature::ContainerDescriptorRange"] }

  %"dxbc::RootSignature::ContainerRootSignatureDesc" = type { i32, i32, i32, i32, i32, i32 }
  %"dxbc::RootSignature::ContainerRootParameter" = type { i32, i32, i32 }
  %"dxbc::RootSignature::StaticSamplerDesc" = type { i32, i32, i32, i32, float, i32, i32, i32, float, float, i32, i32, i32 }
  %"dxbc::RootSignature::ContainerRootDescriptor" = type { i32, i32, i32 }
  %"dxbc::RootSignature::ContainerRootDescriptorTable" = type { i32, i32 }
  %"dxbc::RootSignature::ContainerDescriptorRange" = type { i32, i32, i32, i32, i32, i32 }
  %"dxbc::RootSignature::ContainerRootConstants" = type { i32, i32, i32 }
  ; The global variable which save the serialized root signature ConstantStruct as init.
  @RootSig = internal constant %0 { 
    %"dxbc::RootSignature::ContainerRootSignatureDesc" { i32 2, i32 0, i32 24, i32 0, i32 96, i32 3 }, 
    [6 x %"dxbc::RootSignature::ContainerRootParameter"] 
      [%"dxbc::RootSignature::ContainerRootParameter" { i32 2, i32 0, i32 200 }, 
       %"dxbc::RootSignature::ContainerRootParameter" { i32 3, i32 0, i32 212 }, 
       %"dxbc::RootSignature::ContainerRootParameter" { i32 4, i32 0, i32 224 }, 
       %"dxbc::RootSignature::ContainerRootParameter" { i32 0, i32 0, i32 236 }, 
       %"dxbc::RootSignature::ContainerRootParameter" { i32 0, i32 0, i32 316 }, 
       %"dxbc::RootSignature::ContainerRootParameter" { i32 1, i32 0, i32 348 }], 
    [2 x %"dxbc::RootSignature::StaticSamplerDesc"] 
      [%"dxbc::RootSignature::StaticSamplerDesc" { i32 85, i32 1, i32 1, i32 1, float 0.000000e+00, i32 16, i32 4, i32 2, float 0.000000e+00, float 0x47EFFFFFE0000000, i32 1, i32 0, i32 0 }, 
       %"dxbc::RootSignature::StaticSamplerDesc" { i32 21, i32 3, i32 1, i32 1, float 0.000000e+00, i32 16, i32 4, i32 2, float 0.000000e+00, float 0x47EFFFFFE0000000, i32 2, i32 0, i32 0 }], 
    %1 { 
      %"dxbc::RootSignature::ContainerRootDescriptor" { i32 0, i32 1, i32 8 }, 
      %"dxbc::RootSignature::ContainerRootDescriptor" zeroinitializer, 
      %"dxbc::RootSignature::ContainerRootDescriptor" zeroinitializer, 
      %2 { %"dxbc::RootSignature::ContainerRootDescriptorTable" { i32 3, i32 244 }, 
        [3 x %"dxbc::RootSignature::ContainerDescriptorRange"] 
          [%"dxbc::RootSignature::ContainerDescriptorRange" { i32 2, i32 1, i32 1, i32 0, i32 0, i32 -1 }, 
          %"dxbc::RootSignature::ContainerDescriptorRange" { i32 0, i32 8, i32 1, i32 0, i32 1, i32 -1 }, 
          %"dxbc::RootSignature::ContainerDescriptorRange" { i32 1, i32 -1, i32 1, i32 0, i32 1, i32 -1 }] }, 
      %3 { %"dxbc::RootSignature::ContainerRootDescriptorTable" { i32 1, i32 324 }, 
        [1 x %"dxbc::RootSignature::ContainerDescriptorRange"] 
          [%"dxbc::RootSignature::ContainerDescriptorRange" { i32 3, i32 4, i32 0, i32 1, i32 0, i32 -1 }] }, 
      %"dxbc::RootSignature::ContainerRootConstants" { i32 10, i32 0, i32 3 } } }


CGHLSLRuntime will generate metadata to link the global variable as root 
signature for given entry function. 

.. code-block::

  ; named metadata for entry root signature
  !hlsl.entry.rootsignatures = !{!2}
  …
  ; link the global variable to entry function
  !2 = !{ptr @main, ptr @RootSig}

To make sure the emitted DXIL is legal, the root signature information will be 
validated after all optimizations in the DXIL backend. 
For all resources used in the shader, the root signature will be checked to 
ensure that the resource is in the root signature. 
The binding information will be collected from the root signature ConstantStruct.

In LLVM DirectX backend, the global variable will be serialized and saved as 
another global variable with section 'RTS0' with the serialized root signature 
as initializer in DXContainerGlobals pass. The serialized root signature is in 
exactly the format it will be written out to the DXContainer object.
The MC ObjectWriter for DXContainer will take the global and write it to the 
correct part based on the section name given to the global.

In DXIL validation for DXC, the root signature part will be deserialized and 
check if resource used in the shader (the information is in pipeline state 
validation part) exists in the root signature. 
For LLVM DirectX backend, this will be done in IR pass before emit DXIL 
instead of validation.

Same check could be done in Sema as well. But at AST level, it is impossible 
to identify unused resource which will be removed later. And the resource 
binding allocation is not done. 
So the only case could be caught in Sema is for resources that are known to be 
used for sure (like resources used in entry function and not under any control 
flow) and binded by user. 
If the resource is not in root signature, error should be reported in Sema.
