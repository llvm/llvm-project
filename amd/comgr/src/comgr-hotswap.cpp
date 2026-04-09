//===- comgr-hotswap.cpp - HotSwap ISA stepping rewrite (stub) -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"

using namespace COMGR;

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    amd_comgr_data_t input,
    const char *source_isa_name, const char *target_isa_name,
    amd_comgr_data_t *output) {
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE ||
      !source_isa_name || !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate and parse both ISA names.
  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Currently only GFX1250 B0-to-A0 is supported.
  if (SourceIdent.Processor != "gfx1250" || TargetIdent.Processor != "gfx1250")
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Stub: return a copy of the input unchanged.
  // Full B0-to-A0 patching implementation follows in subsequent commits.
  DataObject *OutputP = DataObject::allocate(AMD_COMGR_DATA_KIND_EXECUTABLE);
  if (!OutputP)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (auto Status =
          OutputP->setData(llvm::StringRef(InputP->Data, InputP->Size))) {
    OutputP->release();
    return Status;
  }

  *output = DataObject::convert(OutputP);
  return AMD_COMGR_STATUS_SUCCESS;
}
