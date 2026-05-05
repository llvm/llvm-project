#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_HPP
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct KernelArgMeta {
  std::string name;
  int offset = 0;
  int size = 0;
  std::string valueKind;
  int addressSpace = -1;
};

// Per-kernel metadata extracted from the AMDGPU code object's MsgPack notes
// + kernel descriptor (`<name>.kd`).
struct KernelMeta {
  std::string name;
  int kernargSegmentSize = 0;
  int groupSegmentFixedSize = 0;
  int privateSegmentFixedSize = 0;
  int maxFlatWorkgroupSize = 256;
  std::vector<KernelArgMeta> args;

  bool hasKernelDescriptor = false;
  uint32_t computePgmRsrc1 = 0;
  uint32_t computePgmRsrc2 = 0;
  uint16_t kernelCodeProperties = 0;
  uint16_t kernargPreload = 0;

  int implicitArgsBase() const {
    int maxEnd = 0;
    for (auto &a : args) {
      if (a.valueKind.rfind("hidden_", 0) == 0)
        continue;
      int end = a.offset + a.size;
      if (end > maxEnd) maxEnd = end;
    }
    return (maxEnd + 7) & ~7;
  }
};

} // namespace transpiler

#endif
