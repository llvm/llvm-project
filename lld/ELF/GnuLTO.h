//===- LTO.h ----------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a C++ interface to the GCC LTO format.
//
// This file must be kept in sync with plugin-api.h.
//
//===----------------------------------------------------------------------===//

#ifndef GNULTO_H
#define GNULTO_H

#include "lld/Common/LLVM.h"

namespace lld::elf {

enum PluginStatus {
  LDPS_OK = 0,
  LDPS_NO_SYMS,
  LDPS_BAD_HANDLE,
  LDPS_ERR,
};

enum PluginAPIVersion {
  LD_PLUGIN_API_VERSION = 1,
};

enum PluginFileType {
  LDPO_REL,
  LDPO_EXEC,
  LDPO_DYN,
  LDPO_PIE,
};

struct PluginInputFile {
  const char *name;
  int fd;
  off_t offset;
  off_t filesize;
  void *handle;
};

struct PluginSymbol {
  char *name;
  char *version;
  /* This is for compatibility with older ABIs.  The older ABI defined
     only 'def' field.  */
#if __BIG_ENDIAN__ == 1
  char unused;
  char section_kind;
  char symbol_type;
  char def;
#elif __LITTLE_ENDIAN__ == 1
  char def;
  char symbol_type;
  char section_kind;
  char unused;
#else
#error "Could not detect architecture endianess"
#endif
  int visibility;
  uint64_t size;
  char *comdat_key;
  int resolution;
};

enum PluginSymbolResolution {
  LDPR_UNKNOWN = 0,
  LDPR_UNDEF,
  LDPR_PREVAILING_DEF,
  LDPR_PREVAILING_DEF_IRONLY,
  LDPR_PREEMPTED_REG,
  LDPR_PREEMPTED_IR,
  LDPR_RESOLVED_IR,
  LDPR_RESOLVED_EXEC,
  LDPR_RESOLVED_DYN,
  LDPR_PREVAILING_DEF_IRONLY_EXP,
};

enum PluginTag {
  LDPT_NULL,
  LDPT_API_VERSION,
  LDPT_GOLD_VERSION,
  LDPT_LINKER_OUTPUT,
  LDPT_OPTION,
  LDPT_REGISTER_CLAIM_FILE_HOOK,
  LDPT_REGISTER_ALL_SYMBOLS_READ_HOOK,
  LDPT_REGISTER_CLEANUP_HOOK,
  LDPT_ADD_SYMBOLS,
  LDPT_GET_SYMBOLS,
  LDPT_ADD_INPUT_FILE,
  LDPT_MESSAGE,
  LDPT_GET_INPUT_FILE,
  LDPT_RELEASE_INPUT_FILE,
  LDPT_ADD_INPUT_LIBRARY,
  LDPT_OUTPUT_NAME,
  LDPT_SET_EXTRA_LIBRARY_PATH,
  LDPT_GNU_LD_VERSION,
  LDPT_GET_VIEW,
  LDPT_GET_INPUT_SECTION_COUNT,
  LDPT_GET_INPUT_SECTION_TYPE,
  LDPT_GET_INPUT_SECTION_NAME,
  LDPT_GET_INPUT_SECTION_CONTENTS,
  LDPT_UPDATE_SECTION_ORDER,
  LDPT_ALLOW_SECTION_ORDERING,
  LDPT_GET_SYMBOLS_V2,
  LDPT_ALLOW_UNIQUE_SEGMENT_FOR_SECTIONS,
  LDPT_UNIQUE_SEGMENT_FOR_SECTIONS,
  LDPT_GET_SYMBOLS_V3,
  LDPT_GET_INPUT_SECTION_ALIGNMENT,
  LDPT_GET_INPUT_SECTION_SIZE,
  LDPT_REGISTER_NEW_INPUT_HOOK,
  LDPT_GET_WRAP_SYMBOLS,
  LDPT_ADD_SYMBOLS_V2,
  LDPT_GET_API_VERSION,
  LDPT_REGISTER_CLAIM_FILE_HOOK_V2,
};

typedef PluginStatus pluginOnLoad(struct PluginTV *tv);
typedef PluginStatus pluginClaimFileHandler(const PluginInputFile *file,
                                            int *claimed);
typedef PluginStatus pluginClaimFileHandlerV2(const PluginInputFile *file,
                                              int *claimed, int known_used);
typedef PluginStatus pluginAllSymbolsReadHandler();
typedef PluginStatus pluginMessage(int level, const char *format, ...);
typedef PluginStatus pluginRegisterClaimFile(pluginClaimFileHandler *handler);
typedef PluginStatus
pluginRegisterClaimFileV2(pluginClaimFileHandlerV2 *handler);
typedef PluginStatus pluginAddSymbols(void *handle, int nsyms,
                                      const PluginSymbol *syms);
typedef PluginStatus
pluginRegisterAllSymbolsRead(pluginAllSymbolsReadHandler *handler);
typedef PluginStatus pluginGetSymbols(const void *handle, int nsyms,
                                      PluginSymbol *syms);
typedef PluginStatus pluginAddInputFile(const char *pathname);

struct PluginTV {
  PluginTV(PluginTag tag, int val) : tag(tag), val(val) {}
  PluginTV(PluginTag tag, const char *ptr) : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginClaimFileHandler *ptr)
      : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginMessage *ptr) : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginRegisterClaimFile *ptr)
      : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginRegisterClaimFileV2 *ptr)
      : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginAddSymbols *ptr) : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginRegisterAllSymbolsRead *ptr)
      : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginGetSymbols *ptr) : tag(tag), ptr((void *)ptr) {}
  PluginTV(PluginTag tag, pluginAddInputFile *ptr)
      : tag(tag), ptr((void *)ptr) {}

  PluginTag tag;
  union {
    int val;
    void *ptr;
  };
};
} // namespace lld::elf

#endif // GNULTO_H
