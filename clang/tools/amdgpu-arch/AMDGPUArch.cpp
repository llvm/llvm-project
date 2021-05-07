//===- AMDGPUArch.cpp - list AMDGPU installed ----------*- C++ -*---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a tool for detecting name of AMDGPU installed in system
// using HSA. This tool is used by AMDGPU OpenMP driver.
//
//===----------------------------------------------------------------------===//

#if defined(__has_include)
#if __has_include("hsa.h")
#define HSA_HEADER_FOUND 1
#include "hsa.h"
#elif __has_include("hsa/hsa.h")
#define HSA_HEADER_FOUND 1
#include "hsa/hsa.h"
#else
#define HSA_HEADER_FOUND 0
#endif
#else
#define HSA_HEADER_FOUND 0
#endif

#if !HSA_HEADER_FOUND
int main() { return 1; }
#else

#include <string>
#include <vector>

static hsa_status_t iterateAgentsCallback(hsa_agent_t Agent, void *Data) {
  hsa_device_type_t DeviceType;
  hsa_status_t Status =
      hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);

  // continue only if device type if GPU
  if (Status != HSA_STATUS_SUCCESS || DeviceType != HSA_DEVICE_TYPE_GPU) {
    return Status;
  }

  std::vector<std::string> *GPUs =
      static_cast<std::vector<std::string> *>(Data);
  char GPUName[64];
  Status = hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, GPUName);
  if (Status != HSA_STATUS_SUCCESS) {
    return Status;
  }
  GPUs->push_back(GPUName);
  return HSA_STATUS_SUCCESS;
}

int main() {
  hsa_status_t Status = hsa_init();
  if (Status != HSA_STATUS_SUCCESS) {
    return 1;
  }

  std::vector<std::string> GPUs;
  Status = hsa_iterate_agents(iterateAgentsCallback, &GPUs);
  if (Status != HSA_STATUS_SUCCESS) {
    return 1;
  }

  for (const auto &GPU : GPUs)
    printf("%s\n", GPU.c_str());

  if (GPUs.size() < 1)
    return 1;

  hsa_shut_down();
  return 0;
}

#endif
