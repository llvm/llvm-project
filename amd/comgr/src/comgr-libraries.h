#include "libraries.inc"
#include "opencl1.2-c.inc"
#include "opencl2.0-c.inc"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/ArrayRef.h"

static std::tuple<const char*, const void*, size_t> get_oclc_isa_version(llvm::StringRef gfxip) {
#define AMD_DEVICE_LIBS_GFXIP(target, target_gfxip) \
  if (gfxip == target_gfxip) return std::make_tuple(#target ".bc", target##_lib, target##_lib_size);
#include "libraries_defs.inc"

  return std::make_tuple(nullptr, nullptr, 0);
}

#define AMD_DEVICE_LIBS_FUNCTION(target, function) \
  static std::tuple<const char*, const void*, size_t> get_oclc_##function(bool on) { \
    return std::make_tuple( \
      on ? "oclc_" #function "_on_lib.bc" : "oclc_" #function "_off_lib.bc", \
      on ? oclc_##function##_on_lib : oclc_##function##_off_lib, \
      on ? oclc_##function##_on_lib_size : oclc_##function##_off_lib_size \
    ); \
  }
#include "libraries_defs.inc"

llvm::ArrayRef<std::tuple<llvm::StringRef, llvm::StringRef>> COMGR::getDeviceLibraries() {
  static std::tuple<llvm::StringRef, llvm::StringRef> DeviceLibs[] = {
#define AMD_DEVICE_LIBS_TARGET(target) \
    {#target ".bc", llvm::StringRef(reinterpret_cast<const char *>(target##_lib), target##_lib_size)},
#include "libraries_defs.inc"
  };
  return DeviceLibs;
}


