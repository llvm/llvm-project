//===-- sanitizer_procmaps_mac.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Information about the process mappings (Mac-specific parts).
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_APPLE
#include "sanitizer_common.h"
#include "sanitizer_placement_new.h"
#include "sanitizer_procmaps.h"

#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach/mach.h>

// These are not available in older macOS SDKs.
#  ifndef CPU_SUBTYPE_X86_64_H
#    define CPU_SUBTYPE_X86_64_H ((cpu_subtype_t)8) /* Haswell */
#  endif
#  ifndef CPU_SUBTYPE_ARM_V7S
#    define CPU_SUBTYPE_ARM_V7S ((cpu_subtype_t)11) /* Swift */
#  endif
#  ifndef CPU_SUBTYPE_ARM_V7K
#    define CPU_SUBTYPE_ARM_V7K ((cpu_subtype_t)12)
#  endif
#  ifndef CPU_TYPE_ARM64
#    define CPU_TYPE_ARM64 (CPU_TYPE_ARM | CPU_ARCH_ABI64)
#  endif
#  ifndef CPU_SUBTYPE_ARM64E
#    define CPU_SUBTYPE_ARM64E ((cpu_subtype_t)2)
#  endif

namespace __sanitizer {

// Contains information used to iterate through sections.
struct MemoryMappedSegmentData {
  char name[kMaxSegName];
  uptr nsects;
  const char *current_load_cmd_addr;
  u32 lc_type;
  uptr base_virt_addr;
};

template <typename Section>
static void NextSectionLoad(LoadedModule *module, MemoryMappedSegmentData *data,
                            bool isWritable) {
  const Section *sc = (const Section *)data->current_load_cmd_addr;
  data->current_load_cmd_addr += sizeof(Section);

  uptr sec_start = sc->addr + data->base_virt_addr;
  uptr sec_end = sec_start + sc->size;
  module->addAddressRange(sec_start, sec_end, /*executable=*/false, isWritable,
                          sc->sectname);
}

static bool VerifyMemoryMapping(MemoryMappingLayout* mapping) {
  InternalMmapVector<LoadedModule> modules;
  modules.reserve(128);  // matches DumpProcessMap
  mapping->DumpListOfModules(&modules);

  InternalMmapVector<LoadedModule::AddressRange> segments;
  for (uptr i = 0; i < modules.size(); ++i) {
    for (auto& range : modules[i].ranges()) {
      if (range.beg == range.end)
        continue;
      segments.push_back(range);
    }
  }

  // Verify that none of the segments overlap:
  // 1. Sort the segments by the start address
  // 2. Check that every segment starts after the previous one ends.
  Sort(segments.data(), segments.size(),
       [](LoadedModule::AddressRange& a, LoadedModule::AddressRange& b) {
         return a.beg < b.beg;
       });

  // To avoid spam, we only print the report message once-per-process.
  static bool invalid_module_map_reported = false;
  bool well_formed = true;

  for (size_t i = 1; i < segments.size(); i++) {
    uptr cur_start = segments[i].beg;
    uptr prev_end = segments[i - 1].end;
    if (cur_start < prev_end) {
      well_formed = false;
      VReport(2, "Overlapping mappings: %s start = %p, %s end = %p\n",
              segments[i].name, (void*)cur_start, segments[i - 1].name,
              (void*)prev_end);
      if (!invalid_module_map_reported) {
        Report(
            "WARN: Invalid dyld module map detected. This is most likely a bug "
            "in the sanitizer.\n");
        Report("WARN: Backtraces may be unreliable.\n");
        invalid_module_map_reported = true;
      }
    }
  }

  for (auto& m : modules) m.clear();

  mapping->Reset();
  return well_formed;
}

void MemoryMappedSegment::AddAddressRanges(LoadedModule *module) {
  // Don't iterate over sections when the caller hasn't set up the
  // data pointer, when there are no sections, or when the segment
  // is executable. Avoid iterating over executable sections because
  // it will confuse libignore, and because the extra granularity
  // of information is not needed by any sanitizers.
  if (!data_ || !data_->nsects || IsExecutable()) {
    module->addAddressRange(start, end, IsExecutable(), IsWritable(),
                            data_ ? data_->name : nullptr);
    return;
  }

  do {
    if (data_->lc_type == LC_SEGMENT) {
      NextSectionLoad<struct section>(module, data_, IsWritable());
#ifdef MH_MAGIC_64
    } else if (data_->lc_type == LC_SEGMENT_64) {
      NextSectionLoad<struct section_64>(module, data_, IsWritable());
#endif
    }
  } while (--data_->nsects);
}

MemoryMappingLayout::MemoryMappingLayout(bool cache_enabled) {
  Reset();
  VerifyMemoryMapping(this);
}

MemoryMappingLayout::~MemoryMappingLayout() {
}

bool MemoryMappingLayout::Error() const {
  return false;
}

// More information about Mach-O headers can be found in mach-o/loader.h
// Each Mach-O image has a header (mach_header or mach_header_64) starting with
// a magic number, and a list of linker load commands directly following the
// header.
// A load command is at least two 32-bit words: the command type and the
// command size in bytes. We're interested only in segment load commands
// (LC_SEGMENT and LC_SEGMENT_64), which tell that a part of the file is mapped
// into the task's address space.
// The |vmaddr|, |vmsize| and |fileoff| fields of segment_command or
// segment_command_64 correspond to the memory address, memory size and the
// file offset of the current memory segment.
// Because these fields are taken from the images as is, one needs to add
// _dyld_get_image_vmaddr_slide() to get the actual addresses at runtime.

void MemoryMappingLayout::Reset() {
  // Count down from the top.
  // TODO(glider): as per man 3 dyld, iterating over the headers with
  // _dyld_image_count is thread-unsafe. We need to register callbacks for
  // adding and removing images which will invalidate the MemoryMappingLayout
  // state.
  data_.current_image = _dyld_image_count();
  data_.current_load_cmd_count = -1;
  data_.current_load_cmd_addr = 0;
  data_.current_magic = 0;
  data_.current_filetype = 0;
  data_.current_arch = kModuleArchUnknown;
  internal_memset(data_.current_uuid, 0, kModuleUUIDSize);
}

// The dyld load address should be unchanged throughout process execution,
// and it is expensive to compute once many libraries have been loaded,
// so cache it here and do not reset.
static mach_header *dyld_hdr = 0;
static const char kDyldPath[] = "/usr/lib/dyld";
static const int kDyldImageIdx = -1;

// static
void MemoryMappingLayout::CacheMemoryMappings() {
  // No-op on Mac for now.
}

void MemoryMappingLayout::LoadFromCache() {
  // No-op on Mac for now.
}

static bool IsDyldHdr(const mach_header *hdr) {
  return (hdr->magic == MH_MAGIC || hdr->magic == MH_MAGIC_64) &&
         hdr->filetype == MH_DYLINKER;
}

// _dyld_get_image_header() and related APIs don't report dyld itself.
// We work around this by manually recursing through the memory map
// until we hit a Mach header matching dyld instead. These recurse
// calls are expensive, but the first memory map generation occurs
// early in the process, when dyld is one of the only images loaded,
// so it will be hit after only a few iterations.  These assumptions don't hold
// on macOS 13+ anymore (dyld itself has moved into the shared cache).
static mach_header *GetDyldImageHeaderViaVMRegion() {
  vm_address_t address = 0;

  while (true) {
    vm_size_t size = 0;
    unsigned depth = 1;
    struct vm_region_submap_info_64 info;
    mach_msg_type_number_t count = VM_REGION_SUBMAP_INFO_COUNT_64;
    kern_return_t err =
        vm_region_recurse_64(mach_task_self(), &address, &size, &depth,
                             (vm_region_info_t)&info, &count);
    if (err != KERN_SUCCESS) return nullptr;

    if (size >= sizeof(mach_header) && info.protection & kProtectionRead) {
      mach_header *hdr = (mach_header *)address;
      if (IsDyldHdr(hdr)) {
        return hdr;
      }
    }
    address += size;
  }
}

extern "C" {
struct dyld_shared_cache_dylib_text_info {
  uint64_t version;  // current version 2
  // following fields all exist in version 1
  uint64_t loadAddressUnslid;
  uint64_t textSegmentSize;
  uuid_t dylibUuid;
  const char *path;  // pointer invalid at end of iterations
  // following fields all exist in version 2
  uint64_t textSegmentOffset;  // offset from start of cache
};
typedef struct dyld_shared_cache_dylib_text_info
    dyld_shared_cache_dylib_text_info;

extern bool _dyld_get_shared_cache_uuid(uuid_t uuid);
extern const void *_dyld_get_shared_cache_range(size_t *length);
extern intptr_t _dyld_get_image_slide(const struct mach_header* mh);
extern int dyld_shared_cache_iterate_text(
    const uuid_t cacheUuid,
    void (^callback)(const dyld_shared_cache_dylib_text_info *info));
}  // extern "C"

static mach_header *GetDyldImageHeaderViaSharedCache() {
  uuid_t uuid;
  bool hasCache = _dyld_get_shared_cache_uuid(uuid);
  if (!hasCache)
    return nullptr;

  size_t cacheLength;
  __block uptr cacheStart = (uptr)_dyld_get_shared_cache_range(&cacheLength);
  CHECK(cacheStart && cacheLength);

  __block mach_header *dyldHdr = nullptr;
  int res = dyld_shared_cache_iterate_text(
      uuid, ^(const dyld_shared_cache_dylib_text_info *info) {
        CHECK_GE(info->version, 2);
        mach_header *hdr =
            (mach_header *)(cacheStart + info->textSegmentOffset);
        if (IsDyldHdr(hdr))
          dyldHdr = hdr;
      });
  CHECK_EQ(res, 0);

  return dyldHdr;
}

const mach_header *get_dyld_hdr() {
  if (!dyld_hdr) {
    // On macOS 13+, dyld itself has moved into the shared cache.  Looking it up
    // via vm_region_recurse_64() causes spins/hangs/crashes.
    if (GetMacosAlignedVersion() >= MacosVersion(13, 0)) {
      dyld_hdr = GetDyldImageHeaderViaSharedCache();
      if (!dyld_hdr) {
        VReport(1,
                "Failed to lookup the dyld image header in the shared cache on "
                "macOS 13+ (or no shared cache in use).  Falling back to "
                "lookup via vm_region_recurse_64().\n");
        dyld_hdr = GetDyldImageHeaderViaVMRegion();
      }
    } else {
      dyld_hdr = GetDyldImageHeaderViaVMRegion();
    }
    CHECK(dyld_hdr);
  }

  return dyld_hdr;
}

// Next and NextSegmentLoad were inspired by base/sysinfo.cc in
// Google Perftools, https://github.com/gperftools/gperftools.

// NextSegmentLoad scans the current image for the next segment load command
// and returns the start and end addresses and file offset of the corresponding
// segment.
// Note that the segment addresses are not necessarily sorted.
template <u32 kLCSegment, typename SegmentCommand>
static bool NextSegmentLoad(MemoryMappedSegment *segment,
                            MemoryMappedSegmentData *seg_data,
                            MemoryMappingLayoutData *layout_data) {
  const char *lc = layout_data->current_load_cmd_addr;

  layout_data->current_load_cmd_addr += ((const load_command *)lc)->cmdsize;
  layout_data->current_load_cmd_count--;
  if (((const load_command *)lc)->cmd == kLCSegment) {
    const SegmentCommand* sc = (const SegmentCommand *)lc;
    if (internal_strcmp(sc->segname, "__LINKEDIT") == 0) {
      // The LINKEDIT sections are for internal linker use, and may alias
      // with the LINKEDIT section for other modules. (If we included them,
      // our memory map would contain overlappping sections.)
      return false;
    }

    uptr base_virt_addr;
    if (layout_data->current_image == kDyldImageIdx)
      base_virt_addr = (uptr)_dyld_get_image_slide(get_dyld_hdr());
    else
      base_virt_addr =
          (uptr)_dyld_get_image_vmaddr_slide(layout_data->current_image);

    segment->start = sc->vmaddr + base_virt_addr;
    segment->end = segment->start + sc->vmsize;
    // Most callers don't need section information, so only fill this struct
    // when required.
    if (seg_data) {
      seg_data->nsects = sc->nsects;
      seg_data->current_load_cmd_addr =
          (const char *)lc + sizeof(SegmentCommand);
      seg_data->lc_type = kLCSegment;
      seg_data->base_virt_addr = base_virt_addr;
      internal_strncpy(seg_data->name, sc->segname,
                       ARRAY_SIZE(seg_data->name));
      seg_data->name[ARRAY_SIZE(seg_data->name) - 1] = 0;
    }

    // Return the initial protection.
    segment->protection = sc->initprot;
    segment->offset = (layout_data->current_filetype ==
                       /*MH_EXECUTE*/ 0x2)
                          ? sc->vmaddr
                          : sc->fileoff;
    if (segment->filename) {
      const char *src = (layout_data->current_image == kDyldImageIdx)
                            ? kDyldPath
                            : _dyld_get_image_name(layout_data->current_image);
      internal_strncpy(segment->filename, src, segment->filename_size);
      segment->filename[segment->filename_size - 1] = 0;
    }
    segment->arch = layout_data->current_arch;
    internal_memcpy(segment->uuid, layout_data->current_uuid, kModuleUUIDSize);
    return true;
  }
  return false;
}

ModuleArch ModuleArchFromCpuType(cpu_type_t cputype, cpu_subtype_t cpusubtype) {
  cpusubtype = cpusubtype & ~CPU_SUBTYPE_MASK;
  switch (cputype) {
    case CPU_TYPE_I386:
      return kModuleArchI386;
    case CPU_TYPE_X86_64:
      if (cpusubtype == CPU_SUBTYPE_X86_64_ALL)
        return kModuleArchX86_64;
      if (cpusubtype == CPU_SUBTYPE_X86_64_H)
        return kModuleArchX86_64H;
      CHECK(0 && "Invalid subtype of x86_64");
      return kModuleArchUnknown;
    case CPU_TYPE_ARM:
      if (cpusubtype == CPU_SUBTYPE_ARM_V6)
        return kModuleArchARMV6;
      if (cpusubtype == CPU_SUBTYPE_ARM_V7)
        return kModuleArchARMV7;
      if (cpusubtype == CPU_SUBTYPE_ARM_V7S)
        return kModuleArchARMV7S;
      if (cpusubtype == CPU_SUBTYPE_ARM_V7K)
        return kModuleArchARMV7K;
      CHECK(0 && "Invalid subtype of ARM");
      return kModuleArchUnknown;
    case CPU_TYPE_ARM64:
      if (cpusubtype == CPU_SUBTYPE_ARM64E)
        return kModuleArchARM64E;
      return kModuleArchARM64;
    default:
      CHECK(0 && "Invalid CPU type");
      return kModuleArchUnknown;
  }
}

static const load_command *NextCommand(const load_command *lc) {
  return (const load_command *)((const char *)lc + lc->cmdsize);
}

#  ifdef MH_MAGIC_64
static constexpr size_t header_size = sizeof(mach_header_64);
#  else
static constexpr size_t header_size = sizeof(mach_header);
#  endif

static void FindUUID(const load_command *first_lc, const mach_header *hdr,
                     u8 *uuid_output) {
  uint32_t curcmd = 0;
  for (const load_command *lc = first_lc; curcmd < hdr->ncmds;
       curcmd++, lc = NextCommand(lc)) {
    CHECK_LT((const char *)lc,
             (const char *)hdr + header_size + hdr->sizeofcmds);

    if (lc->cmd != LC_UUID)
      continue;

    const uuid_command *uuid_lc = (const uuid_command *)lc;
    const uint8_t *uuid = &uuid_lc->uuid[0];
    internal_memcpy(uuid_output, uuid, kModuleUUIDSize);
    return;
  }
}

static bool IsModuleInstrumented(const load_command *first_lc,
                                 const mach_header *hdr) {
  uint32_t curcmd = 0;
  for (const load_command *lc = first_lc; curcmd < hdr->ncmds;
       curcmd++, lc = NextCommand(lc)) {
    CHECK_LT((const char *)lc,
             (const char *)hdr + header_size + hdr->sizeofcmds);

    if (lc->cmd != LC_LOAD_DYLIB)
      continue;

    const dylib_command *dylib_lc = (const dylib_command *)lc;
    uint32_t dylib_name_offset = dylib_lc->dylib.name.offset;
    const char *dylib_name = ((const char *)dylib_lc) + dylib_name_offset;
    dylib_name = StripModuleName(dylib_name);
    if (dylib_name != 0 && (internal_strstr(dylib_name, "libclang_rt."))) {
      return true;
    }
  }
  return false;
}

const ImageHeader *MemoryMappingLayout::CurrentImageHeader() {
  const mach_header *hdr = (data_.current_image == kDyldImageIdx)
                                ? get_dyld_hdr()
                                : _dyld_get_image_header(data_.current_image);
  return (const ImageHeader *)hdr;
}

bool MemoryMappingLayout::Next(MemoryMappedSegment *segment) {
  for (; data_.current_image >= kDyldImageIdx; data_.current_image--) {
    const mach_header *hdr = (const mach_header *)CurrentImageHeader();
    if (!hdr) continue;
    if (data_.current_load_cmd_count < 0) {
      // Set up for this image;
      data_.current_load_cmd_count = hdr->ncmds;
      data_.current_magic = hdr->magic;
      data_.current_filetype = hdr->filetype;
      data_.current_arch = ModuleArchFromCpuType(hdr->cputype, hdr->cpusubtype);
      switch (data_.current_magic) {
#ifdef MH_MAGIC_64
        case MH_MAGIC_64: {
          data_.current_load_cmd_addr =
              (const char *)hdr + sizeof(mach_header_64);
          break;
        }
#endif
        case MH_MAGIC: {
          data_.current_load_cmd_addr = (const char *)hdr + sizeof(mach_header);
          break;
        }
        default: {
          continue;
        }
      }
      FindUUID((const load_command *)data_.current_load_cmd_addr, hdr,
               data_.current_uuid);
      data_.current_instrumented = IsModuleInstrumented(
          (const load_command *)data_.current_load_cmd_addr, hdr);
    }

    while (data_.current_load_cmd_count > 0) {
      switch (data_.current_magic) {
        // data_.current_magic may be only one of MH_MAGIC, MH_MAGIC_64.
#ifdef MH_MAGIC_64
        case MH_MAGIC_64: {
          if (NextSegmentLoad<LC_SEGMENT_64, struct segment_command_64>(
                  segment, segment->data_, &data_))
            return true;
          break;
        }
#endif
        case MH_MAGIC: {
          if (NextSegmentLoad<LC_SEGMENT, struct segment_command>(
                  segment, segment->data_, &data_))
            return true;
          break;
        }
      }
    }
    // If we get here, no more load_cmd's in this image talk about
    // segments.  Go on to the next image.
    data_.current_load_cmd_count = -1; // This will trigger loading next image
  }
  return false;
}

void MemoryMappingLayout::DumpListOfModules(
    InternalMmapVectorNoCtor<LoadedModule> *modules) {
  Reset();
  InternalMmapVector<char> module_name(kMaxPathLength);
  MemoryMappedSegment segment(module_name.data(), module_name.size());
  MemoryMappedSegmentData data;
  segment.data_ = &data;
  while (Next(&segment)) {
    // skip the __PAGEZERO segment, its vmsize is 0
    if (segment.filename[0] == '\0' || (segment.start == segment.end))
      continue;
    LoadedModule *cur_module = nullptr;
    if (!modules->empty() &&
        0 == internal_strcmp(segment.filename, modules->back().full_name())) {
      cur_module = &modules->back();
    } else {
      modules->push_back(LoadedModule());
      cur_module = &modules->back();
      cur_module->set(segment.filename, segment.start, segment.arch,
                      segment.uuid, data_.current_instrumented);
    }
    segment.AddAddressRanges(cur_module);
  }
}

}  // namespace __sanitizer

#endif  // SANITIZER_APPLE
