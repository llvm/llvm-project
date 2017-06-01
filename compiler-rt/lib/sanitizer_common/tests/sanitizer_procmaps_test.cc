//===-- sanitizer_procmaps_test.cc ----------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//
#if !defined(_WIN32)  // There are no /proc/maps on Windows.

#include "sanitizer_common/sanitizer_procmaps.h"
#include "gtest/gtest.h"

#include <stdlib.h>

static void noop() {}
extern const char *argv0;

namespace __sanitizer {

#if SANITIZER_LINUX && !SANITIZER_ANDROID
TEST(MemoryMappingLayout, CodeRange) {
  uptr start, end;
  bool res = GetCodeRangeForFile("[vdso]", &start, &end);
  EXPECT_EQ(res, true);
  EXPECT_GT(start, 0U);
  EXPECT_LT(start, end);
}
#endif

TEST(MemoryMappingLayout, DumpListOfModules) {
  const char *last_slash = strrchr(argv0, '/');
  const char *binary_name = last_slash ? last_slash + 1 : argv0;
  MemoryMappingLayout memory_mapping(false);
  const uptr kMaxModules = 100;
  InternalMmapVector<LoadedModule> modules(kMaxModules);
  memory_mapping.DumpListOfModules(&modules);
  EXPECT_GT(modules.size(), 0U);
  bool found = false;
  for (uptr i = 0; i < modules.size(); ++i) {
    if (modules[i].containsAddress((uptr)&noop)) {
      // Verify that the module name is sane.
      if (strstr(modules[i].full_name(), binary_name) != 0)
        found = true;
    }
    modules[i].clear();
  }
  EXPECT_TRUE(found);
}

TEST(MemoryMapping, LoadedModuleArchAndUUID) {
  if (SANITIZER_MAC) {
    MemoryMappingLayout memory_mapping(false);
    const uptr kMaxModules = 100;
    InternalMmapVector<LoadedModule> modules(kMaxModules);
    memory_mapping.DumpListOfModules(&modules);
    for (uptr i = 0; i < modules.size(); ++i) {
      ModuleArch arch = modules[i].arch();
      // Darwin unit tests are only run on i386/x86_64/x86_64h.
      if (SANITIZER_WORDSIZE == 32) {
        EXPECT_EQ(arch, kModuleArchI386);
      } else if (SANITIZER_WORDSIZE == 64) {
        EXPECT_TRUE(arch == kModuleArchX86_64 || arch == kModuleArchX86_64H);
      }
      const u8 *uuid = modules[i].uuid();
      u8 null_uuid[kModuleUUIDSize] = {0};
      EXPECT_NE(memcmp(null_uuid, uuid, kModuleUUIDSize), 0);
    }
  }
}

}  // namespace __sanitizer
#endif  // !defined(_WIN32)
