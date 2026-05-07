#include "code_object_utils.hpp"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <fstream>

namespace transpiler {

namespace {
inline uint32_t readU32(const uint8_t *p) {
  uint32_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
inline uint16_t readU16(const uint8_t *p) {
  uint16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

// Locate the `<kernelName>.kd` symbol and copy its 64 KD bytes into `out`.
// Returns true on success. The KD symbol is *always* in the .rodata section
// for amdhsa code objects (the AMDGPU asm printer emits it there); we map
// the symbol's virtual address back to a file-level byte offset within the
// section's contents and copy the canonical 64-byte structure. Any
// mismatch (missing symbol, wrong size, address not within .rodata) is
// reported and produces `false`.
//
// We deliberately key off the symbol rather than the MsgPack metadata: the
// MsgPack notes do not include kernarg_preload_length / preload_offset,
// and that information is essential for modelling the gfx1250 user-SGPR
// ABI in Phase 4 of the raiser.
bool readKernelDescriptorBytes(llvm::object::ObjectFile &obj,
                               llvm::StringRef kernelName,
                               std::array<uint8_t, 64> &out) {
  std::string kdSymName = (kernelName + ".kd").str();

  std::optional<llvm::object::SectionRef> rodataSec;
  for (const auto &sec : obj.sections()) {
    auto nameOrErr = sec.getName();
    if (!nameOrErr) {
      (void)llvm::toString(nameOrErr.takeError());
      continue;
    }
    if (*nameOrErr == ".rodata") {
      rodataSec = sec;
      break;
    }
  }
  if (!rodataSec) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: no .rodata "
                    "section in code object\n";
    return false;
  }

  uint64_t rodataAddr = rodataSec->getAddress();
  uint64_t rodataSize = rodataSec->getSize();
  auto rodataContentsOrErr = rodataSec->getContents();
  if (!rodataContentsOrErr) {
    (void)llvm::toString(rodataContentsOrErr.takeError());
    return false;
  }
  auto rodataContents = *rodataContentsOrErr;

  for (const auto &sym : obj.symbols()) {
    auto nameOrErr = sym.getName();
    if (!nameOrErr) {
      (void)llvm::toString(nameOrErr.takeError());
      continue;
    }
    if (*nameOrErr != kdSymName)
      continue;

    auto addrOrErr = sym.getAddress();
    if (!addrOrErr) {
      (void)llvm::toString(addrOrErr.takeError());
      return false;
    }
    uint64_t symAddr = *addrOrErr;

    if (symAddr < rodataAddr || symAddr + 64 > rodataAddr + rodataSize) {
      llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                   << kdSymName << "' at 0x" << llvm::utohexstr(symAddr)
                   << " is not contained within .rodata [0x"
                   << llvm::utohexstr(rodataAddr) << ", 0x"
                   << llvm::utohexstr(rodataAddr + rodataSize) << ")\n";
      return false;
    }

    uint64_t off = symAddr - rodataAddr;
    if (off + 64 > rodataContents.size()) {
      llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                   << kdSymName << "' offset 0x" << llvm::utohexstr(off)
                   << " + 64 exceeds .rodata contents size 0x"
                   << llvm::utohexstr(rodataContents.size()) << "\n";
      return false;
    }

    std::memcpy(out.data(),
                reinterpret_cast<const uint8_t *>(rodataContents.data()) + off,
                64);
    return true;
  }

  llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '" << kdSymName
               << "' not found\n";
  return false;
}

// Parse the four KD register fields we care about into `meta`. Wraps
// readKernelDescriptorBytes so the call site stays compact and the byte-
// offset constants are co-located with their usage.
void populateKernelDescriptorFields(llvm::object::ObjectFile &obj,
                                    KernelMeta &meta) {
  std::array<uint8_t, 64> kdBytes;
  if (!readKernelDescriptorBytes(obj, meta.name, kdBytes)) {
    meta.hasKernelDescriptor = false;
    return;
  }

  using namespace llvm::amdhsa;
  meta.privateSegmentFixedSize =
      readU32(kdBytes.data() + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET);
  meta.computePgmRsrc1 = readU32(kdBytes.data() + COMPUTE_PGM_RSRC1_OFFSET);
  meta.computePgmRsrc2 = readU32(kdBytes.data() + COMPUTE_PGM_RSRC2_OFFSET);
  meta.kernelCodeProperties =
      readU16(kdBytes.data() + KERNEL_CODE_PROPERTIES_OFFSET);
  meta.kernargPreload = readU16(kdBytes.data() + KERNARG_PRELOAD_OFFSET);
  meta.hasKernelDescriptor = true;
}
} // namespace

std::vector<uint8_t> readFile(llvm::StringRef path) {
  std::ifstream f(path.str(), std::ios::binary | std::ios::ate);
  if (!f.is_open()) {
    llvm::errs() << "transpiler: Cannot open file: " << path << "\n";
    return {};
  }
  auto pos = f.tellg();
  if (pos < 0) {
    llvm::errs() << "transpiler: tellg failed for: " << path << "\n";
    return {};
  }
  auto sz = static_cast<size_t>(pos);
  f.seekg(0);
  std::vector<uint8_t> data(sz);
  f.read(reinterpret_cast<char *>(data.data()), sz);
  if (!f) {
    llvm::errs() << "transpiler: short read on: " << path << "\n";
    return {};
  }
  return data;
}

TextSection extractTextSection(llvm::ArrayRef<uint8_t> elfData) {
  TextSection result;
  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "transpiler: Failed to parse ELF: "
                 << llvm::toString(objOrErr.takeError()) << "\n";
    return result;
  }
  auto &obj = *objOrErr;
  for (const auto &sec : obj->sections()) {
    auto nameOrErr = sec.getName();
    if (!nameOrErr) { (void)llvm::toString(nameOrErr.takeError()); continue; }
    if (*nameOrErr == ".text") {
      auto contentsOrErr = sec.getContents();
      if (!contentsOrErr) { (void)llvm::toString(contentsOrErr.takeError()); continue; }
      result.bytes.assign(contentsOrErr->begin(), contentsOrErr->end());
      result.offset = sec.getAddress();
      result.size = sec.getSize();
      result.valid = true;
      return result;
    }
  }
  llvm::errs() << "transpiler: .text section not found in ELF\n";
  return result;
}

std::vector<std::string> listKernelNames(llvm::ArrayRef<uint8_t> elfData) {
  std::vector<std::string> names;

  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "transpiler: listKernelNames: Failed to parse ELF: "
                 << llvm::toString(objOrErr.takeError()) << "\n";
    return names;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(objOrErr->get());
  if (!elf) {
    llvm::errs() << "transpiler: listKernelNames: Not ELF64LE\n";
    return names;
  }

  auto sectionsOrErr = elf->getELFFile().sections();
  if (!sectionsOrErr) {
    (void)llvm::toString(sectionsOrErr.takeError());
    return names;
  }

  for (auto &shdr : *sectionsOrErr) {
    if (shdr.sh_type != 7) // SHT_NOTE
      continue;

    auto dataOrErr = elf->getELFFile().getSectionContents(shdr);
    if (!dataOrErr) { (void)llvm::toString(dataOrErr.takeError()); continue; }
    auto data = *dataOrErr;

    size_t off = 0;
    while (off + 12 <= data.size()) {
      uint32_t namesz = readU32(data.data() + off);
      uint32_t descsz = readU32(data.data() + off + 4);
      uint32_t type   = readU32(data.data() + off + 8);
      off += 12;

      uint32_t nameAligned = (namesz + 3) & ~3u;
      uint64_t needed = static_cast<uint64_t>(nameAligned) + descsz;
      if (needed > data.size() - off) break;

      const char *noteName = reinterpret_cast<const char *>(data.data() + off);
      off += nameAligned;

      if (type == 32 && namesz >= 5 &&
          std::memcmp(noteName, "AMDGPU", 6) == 0) {
        llvm::StringRef blob(reinterpret_cast<const char *>(data.data() + off),
                             descsz);
        llvm::msgpack::Document doc;
        if (!doc.readFromBlob(blob, false)) {
          off += (descsz + 3) & ~3u;
          continue;
        }

        auto &root = doc.getRoot();
        if (!root.isMap()) { off += (descsz + 3) & ~3u; continue; }
        auto &rootMap = root.getMap();

        auto kernelsIt = rootMap.find(doc.getNode("amdhsa.kernels"));
        if (kernelsIt == rootMap.end()) { off += (descsz + 3) & ~3u; continue; }

        auto &kernelsNode = kernelsIt->second;
        if (!kernelsNode.isArray()) { off += (descsz + 3) & ~3u; continue; }

        for (auto &kNode : kernelsNode.getArray()) {
          if (!kNode.isMap()) continue;
          auto &kMap = kNode.getMap();
          auto nameIt = kMap.find(doc.getNode(".name"));
          if (nameIt == kMap.end()) continue;
          names.push_back(nameIt->second.toString());
        }
        return names;
      }
      off += (descsz + 3) & ~3;
    }
  }

  return names;
}

KernelMeta extractKernelMeta(llvm::ArrayRef<uint8_t> elfData,
                             llvm::StringRef kernelName) {
  KernelMeta meta;

  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "transpiler: extractKernelMeta: Failed to parse ELF: "
                 << llvm::toString(objOrErr.takeError()) << "\n";
    return meta;
  }
  auto *elf = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(objOrErr->get());
  if (!elf) {
    llvm::errs() << "transpiler: extractKernelMeta: Not ELF64LE\n";
    return meta;
  }

  // Find .note section
  auto sectionsOrErr = elf->getELFFile().sections();
  if (!sectionsOrErr) { (void)llvm::toString(sectionsOrErr.takeError()); return meta; }

  for (auto &shdr : *sectionsOrErr) {
    if (shdr.sh_type != 7) // SHT_NOTE
      continue;

    auto dataOrErr = elf->getELFFile().getSectionContents(shdr);
    if (!dataOrErr) { (void)llvm::toString(dataOrErr.takeError()); continue; }
    auto data = *dataOrErr;

    size_t off = 0;
    while (off + 12 <= data.size()) {
      uint32_t namesz = readU32(data.data() + off);
      uint32_t descsz = readU32(data.data() + off + 4);
      uint32_t type   = readU32(data.data() + off + 8);
      off += 12;

      uint32_t nameAligned = (namesz + 3) & ~3u;
      uint64_t needed = static_cast<uint64_t>(nameAligned) + descsz;
      if (needed > data.size() - off) break;

      const char *noteName = reinterpret_cast<const char *>(data.data() + off);
      off += nameAligned;

      if (type == 32 && namesz >= 5 &&
          std::memcmp(noteName, "AMDGPU", 6) == 0) {
        llvm::StringRef blob(reinterpret_cast<const char *>(data.data() + off),
                             descsz);
        llvm::msgpack::Document doc;
        if (!doc.readFromBlob(blob, false)) {
          off += (descsz + 3) & ~3u;
          continue;
        }

        auto &root = doc.getRoot();
        if (!root.isMap()) { off += (descsz + 3) & ~3u; continue; }
        auto &rootMap = root.getMap();

        auto kernelsIt = rootMap.find(doc.getNode("amdhsa.kernels"));
        if (kernelsIt == rootMap.end()) { off += (descsz + 3) & ~3u; continue; }

        auto &kernelsNode = kernelsIt->second;
        if (!kernelsNode.isArray()) { off += (descsz + 3) & ~3; continue; }

        for (auto &kNode : kernelsNode.getArray()) {
          if (!kNode.isMap()) continue;
          auto &kMap = kNode.getMap();

          auto nameIt = kMap.find(doc.getNode(".name"));
          if (nameIt == kMap.end()) continue;
          std::string kName = nameIt->second.toString();
          if (kName != kernelName) continue;

          meta.name = kName;

          auto getNodeInt = [](llvm::msgpack::DocNode &n) -> int64_t {
            if (n.getKind() == llvm::msgpack::Type::Int) return n.getInt();
            if (n.getKind() == llvm::msgpack::Type::UInt) return static_cast<int64_t>(n.getUInt());
            return 0;
          };

          auto kasIt = kMap.find(doc.getNode(".kernarg_segment_size"));
          if (kasIt != kMap.end())
            meta.kernargSegmentSize = getNodeInt(kasIt->second);

          auto gsfIt = kMap.find(doc.getNode(".group_segment_fixed_size"));
          if (gsfIt != kMap.end())
            meta.groupSegmentFixedSize = getNodeInt(gsfIt->second);

          auto psfIt = kMap.find(doc.getNode(".private_segment_fixed_size"));
          if (psfIt != kMap.end())
            meta.privateSegmentFixedSize = getNodeInt(psfIt->second);

          auto mfwIt = kMap.find(doc.getNode(".max_flat_workgroup_size"));
          if (mfwIt != kMap.end())
            meta.maxFlatWorkgroupSize = getNodeInt(mfwIt->second);

          auto argsIt = kMap.find(doc.getNode(".args"));
          if (argsIt != kMap.end() && argsIt->second.isArray()) {
            for (auto &argNode : argsIt->second.getArray()) {
              if (!argNode.isMap()) continue;
              auto &aMap = argNode.getMap();
              KernelArgMeta am;
              auto f = [&](const char *key) -> llvm::msgpack::DocNode * {
                auto it = aMap.find(doc.getNode(key));
                return (it != aMap.end()) ? &it->second : nullptr;
              };
              if (auto *n = f(".name")) am.name = n->toString();
              if (auto *n = f(".offset")) am.offset = getNodeInt(*n);
              if (auto *n = f(".size")) am.size = getNodeInt(*n);
              if (auto *n = f(".value_kind")) am.valueKind = n->toString();
              if (auto *n = f(".address_space")) am.addressSpace = getNodeInt(*n);
              meta.args.push_back(am);
            }
          }

          // Parse the KD bytes from .rodata once we know the kernel name
          // matched. populateKernelDescriptorFields sets
          // meta.hasKernelDescriptor on success and emits a diagnostic on
          // failure; the caller (raiser / Phase-4 init) is responsible for
          // refusing the lift if the field is false rather than silently
          // assuming a hardcoded SGPR layout.
          populateKernelDescriptorFields(*objOrErr->get(), meta);
          return meta;
        }
      }
      off += (descsz + 3) & ~3;
    }
  }

  llvm::errs() << "transpiler: extractKernelMeta: kernel '" << kernelName
               << "' not found in metadata\n";
  return meta;
}

uint64_t findKernelSymbolOffset(llvm::ArrayRef<uint8_t> elfData,
                                llvm::StringRef kernelName) {
  auto bufOrErr = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(elfData.data()),
                      elfData.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*bufOrErr);
  if (!objOrErr) {
    llvm::errs() << "transpiler: findKernelSymbolOffset: Failed to parse ELF: "
                 << llvm::toString(objOrErr.takeError()) << "\n";
    return 0;
  }

  uint64_t textBase = UINT64_MAX;
  for (const auto &sec : (*objOrErr)->sections()) {
    auto nameOrErr = sec.getName();
    if (nameOrErr && *nameOrErr == ".text") {
      textBase = sec.getAddress();
      break;
    }
  }
  if (textBase == UINT64_MAX) {
    llvm::errs() << "transpiler: findKernelSymbolOffset: no .text section found\n";
    return 0;
  }

  for (const auto &sym : (*objOrErr)->symbols()) {
    auto nameOrErr = sym.getName();
    if (!nameOrErr) { (void)llvm::toString(nameOrErr.takeError()); continue; }
    if (*nameOrErr == kernelName) {
      auto addrOrErr = sym.getAddress();
      if (!addrOrErr) { (void)llvm::toString(addrOrErr.takeError()); continue; }
      if (*addrOrErr < textBase) {
        llvm::errs() << "transpiler: findKernelSymbolOffset: symbol address 0x"
                     << llvm::utohexstr(*addrOrErr) << " < .text base 0x"
                     << llvm::utohexstr(textBase) << "\n";
        return 0;
      }
      return *addrOrErr - textBase;
    }
  }

  llvm::errs() << "transpiler: findKernelSymbolOffset: symbol '" << kernelName
               << "' not found, defaulting to offset 0\n";
  return 0;
}

std::string detectIsaFromElf(llvm::ArrayRef<uint8_t> elfData) {
  // Read the EI_CLASS / e_machine / e_flags fields straight off the
  // ELF64 header rather than building a full ObjectFile — this is
  // called from raise_cli BEFORE we have a CO opened, and we want
  // zero overhead and zero diagnostics on the malformed-input path.
  // The header layout is fixed by the ELF spec (see ELF.h
  // `Elf64_Ehdr`); the magic check rejects every non-ELF input.
  if (elfData.size() < sizeof(llvm::ELF::Elf64_Ehdr))
    return {};
  const auto *eh =
      reinterpret_cast<const llvm::ELF::Elf64_Ehdr *>(elfData.data());
  if (eh->e_ident[llvm::ELF::EI_MAG0] != llvm::ELF::ElfMagic[0] ||
      eh->e_ident[llvm::ELF::EI_MAG1] != llvm::ELF::ElfMagic[1] ||
      eh->e_ident[llvm::ELF::EI_MAG2] != llvm::ELF::ElfMagic[2] ||
      eh->e_ident[llvm::ELF::EI_MAG3] != llvm::ELF::ElfMagic[3])
    return {};
  if (eh->e_ident[llvm::ELF::EI_CLASS] != llvm::ELF::ELFCLASS64)
    return {};
  if (eh->e_machine != llvm::ELF::EM_AMDGPU)
    return {};
  uint32_t mach = eh->e_flags & llvm::ELF::EF_AMDGPU_MACH;
  // ELF.h's AMDGPU_MACH_LIST X-macro pairs every mach value with
  // its canonical "gfxNNN[a-z]?" / R600 marketing string. The macro
  // covers both R600 and AMDGCN families; we filter R600 mach codes
  // (0x01..0x10) explicitly because the transpiler only targets
  // AMDGCN. EF_AMDGPU_MACH_NONE (0x00) likewise returns "" so a
  // generic / older code object falls through to the filename
  // heuristic in raise_cli.
  if (mach >= llvm::ELF::EF_AMDGPU_MACH_R600_FIRST &&
      mach <= llvm::ELF::EF_AMDGPU_MACH_R600_LAST)
    return {};
  if (mach == llvm::ELF::EF_AMDGPU_MACH_NONE)
    return {};
  switch (mach) {
#define HANDLE_AMDGCN(NUM, ENUM, STR)                                          \
  case NUM:                                                                    \
    return STR;
    AMDGPU_MACH_LIST(HANDLE_AMDGCN)
#undef HANDLE_AMDGCN
  default:
    return {};
  }
}

} // namespace transpiler
