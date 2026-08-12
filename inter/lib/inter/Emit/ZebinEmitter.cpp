#include "inter/Emit/Emit.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <string>

using namespace mlir;
using namespace inter::xemachine;

namespace {

using Bytes = llvm::SmallVector<char>;

constexpr uint16_t kElfTypeZebin = 0xFF12;
constexpr uint32_t kSectionZeInfo = 0xFF000011;
constexpr uint32_t kNoteProductFamily = 1;
constexpr uint32_t kNoteGraphicsCore = 2;
constexpr uint32_t kNoteTargetMetadata = 3;
constexpr uint32_t kNoteZebinVersion = 4;
constexpr uint32_t kNoteProductConfig = 6;
constexpr uint32_t kBmgProductFamily = 0x4FA;
constexpr uint32_t kXe2HpgCore = 0xC09;
constexpr uint32_t kBmgG21A0Config = 0x05004000;

struct Section {
  std::string name;
  uint32_t type;
  uint64_t flags;
  uint64_t alignment;
  Bytes data;
  uint32_t nameOffset = 0;
  uint64_t offset = 0;
};

template <typename T> void appendInteger(Bytes &bytes, T value) {
  for (unsigned byte = 0; byte < sizeof(T); ++byte)
    bytes.push_back(char((uint64_t(value) >> (byte * 8)) & 0xFF));
}

template <typename T> void writeInteger(Bytes &bytes, size_t offset, T value) {
  for (unsigned byte = 0; byte < sizeof(T); ++byte)
    bytes[offset + byte] = char((uint64_t(value) >> (byte * 8)) & 0xFF);
}

void appendBytes(Bytes &bytes, llvm::StringRef value) {
  bytes.append(value.begin(), value.end());
}

void align(Bytes &bytes, uint64_t alignment) {
  bytes.resize(llvm::alignTo(bytes.size(), alignment), 0);
}

Bytes uint32Descriptor(uint32_t value) {
  Bytes bytes;
  appendInteger(bytes, value);
  return bytes;
}

void appendIntelGTNote(Bytes &notes, uint32_t type,
                       llvm::ArrayRef<char> descriptor) {
  appendInteger(notes, uint32_t(8));
  appendInteger(notes, uint32_t(descriptor.size()));
  appendInteger(notes, type);
  appendBytes(notes, llvm::StringRef("IntelGT\0", 8));
  notes.append(descriptor.begin(), descriptor.end());
  align(notes, 4);
}

Bytes buildCompatibilityNotes() {
  Bytes notes;
  Bytes productFamily = uint32Descriptor(kBmgProductFamily);
  Bytes graphicsCore = uint32Descriptor(kXe2HpgCore);
  Bytes targetMetadata = uint32Descriptor(0);
  Bytes productConfig = uint32Descriptor(kBmgG21A0Config);
  appendIntelGTNote(notes, kNoteProductFamily, productFamily);
  appendIntelGTNote(notes, kNoteGraphicsCore, graphicsCore);
  appendIntelGTNote(notes, kNoteTargetMetadata, targetMetadata);
  appendIntelGTNote(notes, kNoteZebinVersion,
                    llvm::ArrayRef<char>("1.64\0", 5));
  appendIntelGTNote(notes, kNoteProductConfig, productConfig);
  return notes;
}

std::string quoteYaml(llvm::StringRef value) {
  std::string result = "'";
  for (char c : value) {
    result += c;
    if (c == '\'')
      result += '\'';
  }
  result += '\'';
  return result;
}

FailureOr<std::string> buildZeInfo(func::FuncOp kernel) {
  auto target = kernel->getAttrOfType<TargetAttr>(kTargetAttrName);
  if (!target || target.getChip().getValue() != "bmg")
    return kernel.emitOpError("zebin emission requires a BMG target"),
           failure();

  ArrayAttr kernelArgs = kernel->getAttrOfType<ArrayAttr>(kKernelArgsAttrName);
  if (failed(verifyKernelArgLayout(kernelArgs, kernel.getOperation())))
    return failure();

  auto grfCount = kernel->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
  auto grfUsed = kernel->getAttrOfType<IntegerAttr>(kGrfUsedAttrName);
  auto simdSize = kernel->getAttrOfType<IntegerAttr>(kSimdSizeAttrName);
  auto barrierCount = kernel->getAttrOfType<IntegerAttr>(kBarrierCountAttrName);
  auto hasGlobalAtomics =
      kernel->getAttrOfType<BoolAttr>(kHasGlobalAtomicsAttrName);
  auto hasNoStatelessWrite =
      kernel->getAttrOfType<BoolAttr>(kHasNoStatelessWriteAttrName);
  auto hasDpas = kernel->getAttrOfType<BoolAttr>(kHasDpasAttrName);
  if (!grfCount || !grfUsed || !simdSize || !barrierCount ||
      !hasGlobalAtomics || !hasNoStatelessWrite || !hasDpas)
    return kernel.emitOpError("missing machine resource attributes"), failure();
  if (grfCount.getInt() != 128 ||
      (simdSize.getInt() != 8 && simdSize.getInt() != 16 &&
       simdSize.getInt() != 32))
    return kernel.emitOpError("unsupported GRF count or SIMD size"), failure();
  if (grfUsed.getInt() < 0 || grfUsed.getInt() > grfCount.getInt())
    return kernel.emitOpError("invalid physical GRF usage attribute"),
           failure();
  if (barrierCount.getInt() < 0 || barrierCount.getInt() > 1)
    return kernel.emitOpError("unsupported barrier count"), failure();
  FailureOr<KernelResourceUsage> usage =
      analyzeKernelResources(kernel, grfCount.getInt());
  if (failed(usage))
    return failure();
  if (grfUsed.getInt() != static_cast<int64_t>(usage->grfUsed) ||
      barrierCount.getInt() != usage->barrierCount ||
      hasGlobalAtomics.getValue() != usage->hasGlobalAtomics ||
      hasNoStatelessWrite.getValue() == usage->hasStatelessWrite ||
      hasDpas.getValue() != usage->hasDpas)
    return kernel.emitOpError("stale machine resource attributes"), failure();

  bool hasBufferArguments = false;
  for (Attribute argument : kernelArgs) {
    if (cast<KernelArgAttr>(argument).getKind() == KernelArgKind::by_pointer)
      hasBufferArguments = true;
  }

  bool usesThreadIds = kernel->hasAttr(kUsesThreadIdsAttrName);
  bool usesPayload = usesThreadIds || !kernelArgs.empty();
  auto inlineSize =
      kernel->getAttrOfType<IntegerAttr>(kInlineDataPayloadSizeAttrName);
  auto perThreadSize =
      kernel->getAttrOfType<IntegerAttr>(kPerThreadPayloadSizeAttrName);
  auto entryOffset =
      kernel->getAttrOfType<IntegerAttr>(kPayloadEntryOffsetAttrName);
  if (usesPayload && (!inlineSize || !entryOffset))
    return kernel.emitOpError("incomplete thread payload attributes"),
           failure();
  if (usesPayload && (inlineSize.getInt() != 32 || entryOffset.getInt() != 192))
    return kernel.emitOpError("unsupported thread payload layout"), failure();
  if (usesThreadIds && (!perThreadSize || perThreadSize.getInt() != 192))
    return kernel.emitOpError("unsupported per-thread payload layout"),
           failure();

  std::string yaml;
  llvm::raw_string_ostream output(yaml);
  output << "---\n"
         << "version: '1.64'\n"
         << "kernels:\n"
         << "  - name: " << quoteYaml(kernel.getName()) << "\n"
         << "    execution_env:\n"
         << "      disable_mid_thread_preemption: true\n"
         << "      grf_count: " << grfCount.getInt() << "\n";
  if (hasBufferArguments)
    output << "      has_4gb_buffers: true\n";
  if (hasGlobalAtomics.getValue())
    output << "      has_global_atomics: true\n";
  if (hasDpas.getValue())
    output << "      has_dpas: true\n";
  output << "      has_no_stateless_write: "
         << (hasNoStatelessWrite.getValue() ? "true" : "false") << "\n";
  if (usesPayload) {
    output << "      inline_data_payload_size: " << inlineSize.getInt() << "\n"
           << "      offset_to_skip_per_thread_data_load: "
           << entryOffset.getInt() << "\n";
  }
  output << "      simd_size: " << simdSize.getInt() << "\n";
  if (barrierCount.getInt() != 0)
    output << "      barrier_count: " << barrierCount.getInt() << "\n";
  if (auto slmSize = kernel->getAttrOfType<IntegerAttr>(kSlmSizeAttrName))
    output << "      slm_size: " << slmSize.getInt() << "\n";
  if (auto scratchSize =
          kernel->getAttrOfType<IntegerAttr>(kScratchSizeAttrName))
    output << "      spill_size: " << scratchSize.getInt() << "\n";

  if (usesThreadIds || !kernelArgs.empty()) {
    output << "    payload_arguments:\n";
    if (usesThreadIds) {
      output << "      - arg_type: global_id_offset\n"
             << "        offset: 0\n"
             << "        size: 12\n"
             << "      - arg_type: enqueued_local_size\n"
             << "        offset: 12\n"
             << "        size: 12\n";
    }
    for (auto [index, descriptorAttr] : llvm::enumerate(kernelArgs)) {
      auto descriptor = cast<KernelArgAttr>(descriptorAttr);
      if (descriptor.getKind() == KernelArgKind::by_pointer) {
        output << "      - arg_type: arg_bypointer\n"
               << "        offset: " << descriptor.getOffset() << "\n"
               << "        size: " << descriptor.getSize() << "\n"
               << "        arg_index: " << index << "\n"
               << "        addrmode: stateless\n"
                << "        addrspace: "
                << descriptor.getAddressSpace().getValue() << "\n"
                << "        access_type: ";
        llvm::StringRef access = descriptor.getAccess().getValue();
        if (access == "read_only")
          output << "readonly\n";
        else if (access == "write_only")
          output << "writeonly\n";
        else
          output << "readwrite\n";
      } else {
        output << "      - arg_type: arg_byvalue\n"
               << "        offset: " << descriptor.getOffset() << "\n"
               << "        size: " << descriptor.getSize() << "\n"
               << "        arg_index: " << index << "\n";
      }
    }
  }
  if (usesThreadIds)
    output << "    per_thread_payload_arguments:\n"
           << "      - arg_type: local_id\n"
           << "        offset: 0\n"
           << "        size: " << perThreadSize.getInt() << "\n";
  if (auto scratchSize =
          kernel->getAttrOfType<IntegerAttr>(kScratchSizeAttrName))
    output << "    per_thread_memory_buffers:\n"
           << "      - type: scratch\n"
           << "        usage: single_space\n"
           << "        size: " << scratchSize.getInt() << "\n";
  output << "...\n";
  return yaml;
}

void appendSectionHeader(Bytes &output, uint32_t name, uint32_t type,
                         uint64_t flags, uint64_t offset, uint64_t size,
                         uint32_t link, uint32_t info, uint64_t alignment,
                         uint64_t entrySize) {
  appendInteger(output, name);
  appendInteger(output, type);
  appendInteger(output, flags);
  appendInteger(output, uint64_t(0));
  appendInteger(output, offset);
  appendInteger(output, size);
  appendInteger(output, link);
  appendInteger(output, info);
  appendInteger(output, alignment);
  appendInteger(output, entrySize);
}

LogicalResult writeElf(func::FuncOp kernel, llvm::ArrayRef<char> text,
                       llvm::StringRef zeInfo, llvm::raw_ostream &output) {
  Bytes strtab;
  strtab.push_back(0);
  appendBytes(strtab, kernel.getName());
  strtab.push_back(0);

  Bytes symtab(24, 0);
  appendInteger(symtab, uint32_t(1));
  symtab.push_back(0x12);
  symtab.push_back(0);
  appendInteger(symtab, uint16_t(1));
  appendInteger(symtab, uint64_t(0));
  appendInteger(symtab, uint64_t(text.size()));

  llvm::SmallVector<Section> sections;
  std::string textSectionName = ".text.";
  textSectionName += kernel.getName();
  sections.push_back({std::move(textSectionName), llvm::ELF::SHT_PROGBITS,
                      llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_EXECINSTR, 64,
                      Bytes(text.begin(), text.end())});
  sections.push_back(
      {".symtab", llvm::ELF::SHT_SYMTAB, 0, 8, std::move(symtab)});
  sections.push_back(
      {".strtab", llvm::ELF::SHT_STRTAB, 0, 1, std::move(strtab)});
  sections.push_back(
      {".ze_info", kSectionZeInfo, 0, 8, Bytes(zeInfo.begin(), zeInfo.end())});
  sections.push_back({".note.intelgt.compat", llvm::ELF::SHT_NOTE, 0, 4,
                      buildCompatibilityNotes()});

  Bytes sectionNames;
  sectionNames.push_back(0);
  for (Section &section : sections) {
    section.nameOffset = sectionNames.size();
    appendBytes(sectionNames, section.name);
    sectionNames.push_back(0);
  }
  uint32_t sectionNamesNameOffset = sectionNames.size();
  appendBytes(sectionNames, ".shstrtab");
  sectionNames.push_back(0);

  Bytes elf(64, 0);
  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2;
  elf[5] = 1;
  elf[6] = 1;
  writeInteger(elf, 16, kElfTypeZebin);
  writeInteger(elf, 18, uint16_t(llvm::ELF::EM_INTELGT));
  writeInteger(elf, 20, uint32_t(1));
  writeInteger(elf, 52, uint16_t(64));
  writeInteger(elf, 58, uint16_t(64));
  writeInteger(elf, 60, uint16_t(sections.size() + 2));
  writeInteger(elf, 62, uint16_t(sections.size() + 1));

  for (Section &section : sections) {
    align(elf, section.alignment);
    section.offset = elf.size();
    elf.append(section.data.begin(), section.data.end());
  }
  uint64_t sectionNamesOffset = elf.size();
  elf.append(sectionNames.begin(), sectionNames.end());
  align(elf, 8);
  uint64_t sectionHeadersOffset = elf.size();
  writeInteger(elf, 40, sectionHeadersOffset);

  elf.resize(elf.size() + 64, 0);
  for (Section &section : sections) {
    bool isSymtab = section.type == llvm::ELF::SHT_SYMTAB;
    appendSectionHeader(elf, section.nameOffset, section.type, section.flags,
                        section.offset, section.data.size(), isSymtab ? 3 : 0,
                        isSymtab ? 1 : 0, section.alignment, isSymtab ? 24 : 0);
  }
  appendSectionHeader(elf, sectionNamesNameOffset, llvm::ELF::SHT_STRTAB, 0,
                      sectionNamesOffset, sectionNames.size(), 0, 0, 1, 0);
  output.write(elf.data(), elf.size());
  return success();
}

} // namespace

LogicalResult inter::emitZebin(ModuleOp moduleOp, llvm::raw_ostream &output) {
  SmallVector<func::FuncOp> kernels;
  for (func::FuncOp function : moduleOp.getOps<func::FuncOp>()) {
    if (function->hasAttr(kTargetAttrName))
      kernels.push_back(function);
  }
  if (kernels.size() != 1)
    return moduleOp.emitError("zebin emission requires exactly one kernel"),
           failure();

  FailureOr<std::string> zeInfo = buildZeInfo(kernels.front());
  if (failed(zeInfo))
    return failure();

  Bytes text;
  llvm::raw_svector_ostream textOutput(text);
  if (failed(emitGedBinary(moduleOp, textOutput)) || text.empty())
    return failure();
  return writeElf(kernels.front(), text, *zeInfo, output);
}
