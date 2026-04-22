//===- comgr-hotswap.cpp - HotSwap ISA rewriting: public API bridge -------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr-hotswap-internal.h"
#include "comgr.h"

using namespace COMGR;

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_rewrite(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, amd_comgr_data_t *output) {
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE || !source_isa_name ||
      !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  if (SourceIdent.Processor != "gfx1250" || TargetIdent.Processor != "gfx1250")
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  std::unique_ptr<llvm::MemoryBuffer> OutBuffer;
  amd_comgr_status_t Status = hotswap::retargetCodeObjectB0A0(
      InputP->Data, InputP->Size, TargetIdent, OutBuffer);
  if (Status != AMD_COMGR_STATUS_SUCCESS)
    return Status;
  if (!OutBuffer)
    return AMD_COMGR_STATUS_ERROR;

  DataObject *OutputP = DataObject::allocate(AMD_COMGR_DATA_KIND_EXECUTABLE);
  if (!OutputP)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (amd_comgr_status_t SetStatus = OutputP->setData(std::move(OutBuffer))) {
    OutputP->release();
    return SetStatus;
  }

  *output = DataObject::convert(OutputP);
  return AMD_COMGR_STATUS_SUCCESS;
}
