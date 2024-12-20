#include "clang/Sema/ParseHLSLRootSignature.h"

namespace llvm {
namespace hlsl {
namespace root_signature {

// TODO: Hook up with Sema to properly report semantic/validation errors
bool Parser::ReportError() { return true; }

bool Parser::ParseRootFlags() {
  // Set to RootFlags::None and skip whitespace to catch when we have RootFlags(
  // )
  RootFlags Flags = RootFlags::None;
  Buffer = Buffer.drop_while(isspace);
  bool First = true;

  // Loop until we reach the end of the rootflags
  while (!Buffer.starts_with(")")) {
    // Trim expected | when more than 1 flag
    if (!First && !Buffer.consume_front("|"))
      return ReportError();
    First = false;

    // Remove any whitespace
    Buffer = Buffer.drop_while(isspace);

    RootFlags CurFlag;
    if (ParseRootFlag(CurFlag))
      return ReportError();
    Flags |= CurFlag;

    // Remove any whitespace
    Buffer = Buffer.drop_while(isspace);
  }

  // Create and push the root element on the parsed elements
  Elements->push_back(RootElement(Flags));
  return false;
}

bool Parser::ParseRootParameter() {
  RootParameter Parameter;
  Parameter.Type = llvm::StringSwitch<RootType>(Token)
                       .Case("CBV", RootType::CBV)
                       .Case("SRV", RootType::SRV)
                       .Case("UAV", RootType::UAV)
                       .Case("RootConstants", RootType::Constants);
  // Should never reach as Token was just verified in dispatch
  // Remove any whitespace
  Buffer = Buffer.drop_while(isspace);

  // Retreive mandatory num32BitConstant arg for RootConstants
  if (Parameter.Type == RootType::Constants) {
    if (!Buffer.consume_front("num32BitConstants"))
      return ReportError();

    if (ParseAssign())
      return ReportError();

    if (ParseUnsignedInt(Parameter.Num32BitConstants))
      return ReportError();

    if (ParseOptComma())
      return ReportError();
  }

  // Retrieve mandatory register
  if (ParseRegister(Parameter.Register))
    return ReportError();

  if (ParseOptComma())
    return ReportError();

  // Parse common optional space arg
  if (Buffer.consume_front("space")) {
    if (ParseAssign())
      return ReportError();

    if (ParseUnsignedInt(Parameter.Space))
      return ReportError();

    if (ParseOptComma())
      return ReportError();
  }

  // Parse common optional visibility arg
  if (Buffer.consume_front("visibility")) {
    if (ParseAssign())
      return ReportError();

    if (ParseVisibility(Parameter.Visibility))
      return ReportError();

    if (ParseOptComma())
      return ReportError();
  }

  // Retreive optional flags arg for non-RootConstants
  if (Parameter.Type != RootType::Constants && Buffer.consume_front("flags")) {
    if (ParseAssign())
      return ReportError();

    if (ParseRootDescriptorFlag(Parameter.Flags))
      return ReportError();

    // Remove trailing whitespace
    Buffer = Buffer.drop_while(isspace);
  }

  // Create and push the root element on the parsed elements
  Elements->push_back(RootElement(Parameter));
  return false;
}

// Helper Parser methods

// Parses " = " with varying whitespace
bool Parser::ParseAssign() {
  Buffer = Buffer.drop_while(isspace);
  if (!Buffer.starts_with('='))
    return true;
  Buffer = Buffer.drop_front();
  Buffer = Buffer.drop_while(isspace);
  return false;
}

// Parses ", " with varying whitespace
bool Parser::ParseComma() {
  if (!Buffer.starts_with(','))
    return true;
  Buffer = Buffer.drop_front();
  Buffer = Buffer.drop_while(isspace);
  return false;
}

// Parses ", " if possible. When successful we expect another parameter, and
// return no error, otherwise we expect that we should be at the end of the
// root element and return an error if this isn't the case
bool Parser::ParseOptComma() {
  if (!ParseComma())
    return false;
  Buffer = Buffer.drop_while(isspace);
  return !Buffer.starts_with(')');
}

bool Parser::ParseRegister(Register &Register) {
  // Parse expected register type ('b', 't', 'u', 's')
  if (Buffer.empty())
    return ReportError();

  // Get type character
  Token = Buffer.take_front();
  Buffer = Buffer.drop_front();

  auto MaybeType = llvm::StringSwitch<std::optional<RegisterType>>(Token)
                       .Case("b", RegisterType::BReg)
                       .Case("t", RegisterType::TReg)
                       .Case("u", RegisterType::UReg)
                       .Case("s", RegisterType::SReg)
                       .Default(std::nullopt);
  if (!MaybeType)
    return ReportError();
  Register.ViewType = *MaybeType;

  if (ParseUnsignedInt(Register.Number))
    return ReportError();

  return false;
}

// Parses "[0-9+]" as an unsigned int
bool Parser::ParseUnsignedInt(uint32_t &Number) {
  StringRef NumString = Buffer.take_while(isdigit);
  APInt X = APInt(32, 0);
  if (NumString.getAsInteger(/*radix=*/10, X))
    return true;
  Number = X.getZExtValue();
  Buffer = Buffer.drop_front(NumString.size());
  return false;
}

template <typename EnumType>
bool Parser::ParseEnum(SmallVector<std::pair<StringLiteral, EnumType>> Mapping,
                       EnumType &Enum) {
  // Retrieve enum
  Token = Buffer.take_while([](char C) { return isalnum(C) || C == '_'; });
  Buffer = Buffer.drop_front(Token.size());

  // Try to get the case-insensitive enum
  auto Switch = llvm::StringSwitch<std::optional<EnumType>>(Token);
  for (auto Pair : Mapping)
    Switch.CaseLower(Pair.first, Pair.second);
  auto MaybeEnum = Switch.Default(std::nullopt);
  if (!MaybeEnum)
    return true;
  Enum = *MaybeEnum;

  return false;
}

bool Parser::ParseRootDescriptorFlag(RootDescriptorFlags &Flag) {
  SmallVector<std::pair<StringLiteral, RootDescriptorFlags>> Mapping = {
      {"0", RootDescriptorFlags::None},
      {"DATA_VOLATILE", RootDescriptorFlags::DataVolatile},
      {"DATA_STATIC_WHILE_SET_AT_EXECUTE",
       RootDescriptorFlags::DataStaticWhileSetAtExecute},
      {"DATA_STATIC", RootDescriptorFlags::DataStatic},
  };

  return ParseEnum<RootDescriptorFlags>(Mapping, Flag);
}

bool Parser::ParseRootFlag(RootFlags &Flag) {
  SmallVector<std::pair<StringLiteral, RootFlags>> Mapping = {
      {"0", RootFlags::None},
      {"ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT",
       RootFlags::AllowInputAssemblerInputLayout},
      {"DENY_VERTEX_SHADER_ROOT_ACCESS", RootFlags::DenyVertexShaderRootAccess},
      {"DENY_HULL_SHADER_ROOT_ACCESS", RootFlags::DenyHullShaderRootAccess},
      {"DENY_DOMAIN_SHADER_ROOT_ACCESS", RootFlags::DenyDomainShaderRootAccess},
      {"DENY_GEOMETRY_SHADER_ROOT_ACCESS",
       RootFlags::DenyGeometryShaderRootAccess},
      {"DENY_PIXEL_SHADER_ROOT_ACCESS", RootFlags::DenyPixelShaderRootAccess},
      {"ALLOW_STREAM_OUTPUT", RootFlags::AllowStreamOutput},
      {"LOCAL_ROOT_SIGNATURE", RootFlags::LocalRootSignature},
      {"DENY_AMPLIFICATION_SHADER_ROOT_ACCESS",
       RootFlags::DenyAmplificationShaderRootAccess},
      {"DENY_MESH_SHADER_ROOT_ACCESS", RootFlags::DenyMeshShaderRootAccess},
      {"CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED",
       RootFlags::CBVSRVUAVHeapDirectlyIndexed},
      {"SAMPLER_HEAP_DIRECTLY_INDEXED", RootFlags::SamplerHeapDirectlyIndexed},
      {"AllowLowTierReservedHwCbLimit",
       RootFlags::AllowLowTierReservedHwCbLimit},
  };

  return ParseEnum<RootFlags>(Mapping, Flag);
}

bool Parser::ParseVisibility(ShaderVisibility &Visibility) {
  SmallVector<std::pair<StringLiteral, ShaderVisibility>> Mapping = {
      {"SHADER_VISIBILITY_ALL", ShaderVisibility::All},
      {"SHADER_VISIBILITY_VERTEX", ShaderVisibility::Vertex},
      {"SHADER_VISIBILITY_HULL", ShaderVisibility::Hull},
      {"SHADER_VISIBILITY_DOMAIN", ShaderVisibility::Domain},
      {"SHADER_VISIBILITY_GEOMETRY", ShaderVisibility::Geometry},
      {"SHADER_VISIBILITY_PIXEL", ShaderVisibility::Pixel},
      {"SHADER_VISIBILITY_AMPLIFICATION", ShaderVisibility::Amplification},
      {"SHADER_VISIBILITY_MESH", ShaderVisibility::Mesh},
  };

  return ParseEnum<ShaderVisibility>(Mapping, Visibility);
}

bool Parser::ParseRootElement() {
  // Define different ParserMethods to use StringSwitch for dispatch
  enum class ParserMethod {
    ReportError,
    ParseRootFlags,
    ParseRootParameter,
  };

  // Retreive which method should be used
  auto Method = llvm::StringSwitch<ParserMethod>(Token)
                    .Case("RootFlags", ParserMethod::ParseRootFlags)
                    .Case("RootConstants", ParserMethod::ParseRootParameter)
                    .Case("CBV", ParserMethod::ParseRootParameter)
                    .Case("SRV", ParserMethod::ParseRootParameter)
                    .Case("UAV", ParserMethod::ParseRootParameter)
                    .Default(ParserMethod::ReportError);

  // Dispatch on the correct method
  bool Error = false;
  switch (Method) {
  case ParserMethod::ReportError:
    Error = true;
    break;
  case ParserMethod::ParseRootFlags:
    Error = ParseRootFlags();
    break;
  case ParserMethod::ParseRootParameter:
    Error = ParseRootParameter();
    break;
  }

  if (Error)
    return ReportError();

  return false;
}

// Parser entry point function
bool Parser::Parse() {
  bool First = true;
  while (!Buffer.empty()) {
    // Trim expected comma when more than 1 root element
    if (!First && !Buffer.consume_front(","))
      return ReportError();
    First = false;

    // Remove any whitespace
    Buffer = Buffer.drop_while(isspace);

    // Retrieve the root element identifier
    auto Split = Buffer.split('(');
    Token = Split.first;
    Buffer = Split.second;

    // Dispatch to the applicable root element parser
    if (ParseRootElement())
      return ReportError();

    // Then we can clean up the remaining ")"
    if (!Buffer.consume_front(")"))
      return ReportError();
  }

  // All input has been correctly parsed
  return false;
}

} // namespace root_signature
} // namespace hlsl
} // namespace llvm
