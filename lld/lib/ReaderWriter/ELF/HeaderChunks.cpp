//===- lib/ReaderWriter/ELF/HeaderChunks.cpp --------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "HeaderChunks.h"
#include "TargetLayout.h"
#include "llvm/ADT/STLExtras.h"

namespace lld {
namespace elf {

template <class ELFT> void ELFHeader<ELFT>::finalize() {
  _eh.e_ident[llvm::ELF::EI_CLASS] =
      (ELFT::Is64Bits) ? llvm::ELF::ELFCLASS64 : llvm::ELF::ELFCLASS32;
  _eh.e_ident[llvm::ELF::EI_DATA] =
      (ELFT::TargetEndianness == llvm::support::little)
          ? llvm::ELF::ELFDATA2LSB
          : llvm::ELF::ELFDATA2MSB;
  _eh.e_type = this->_ctx.getOutputELFType();
  _eh.e_machine = this->_ctx.getOutputMachine();
}

template <class ELFT>
ELFHeader<ELFT>::ELFHeader(const ELFLinkingContext &ctx)
    : Chunk<ELFT>("elfhdr", Chunk<ELFT>::Kind::ELFHeader, ctx) {
  this->_alignment = ELFT::Is64Bits ? 8 : 4;
  this->_fsize = sizeof(Elf_Ehdr);
  this->_msize = sizeof(Elf_Ehdr);
  memset(_eh.e_ident, 0, llvm::ELF::EI_NIDENT);
  e_ident(llvm::ELF::EI_MAG0, 0x7f);
  e_ident(llvm::ELF::EI_MAG1, 'E');
  e_ident(llvm::ELF::EI_MAG2, 'L');
  e_ident(llvm::ELF::EI_MAG3, 'F');
  e_ehsize(sizeof(Elf_Ehdr));
  e_flags(0);
}

template <class ELFT>
void ELFHeader<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                            llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *atomContent = chunkBuffer + this->fileOffset();
  memcpy(atomContent, &_eh, fileSize());
}

template <class ELFT>
bool ProgramHeader<ELFT>::addSegment(Segment<ELFT> *segment) {
  bool allocatedNew = false;
  ELFLinkingContext::OutputMagic outputMagic = this->_ctx.getOutputMagic();
  // For segments that are not a loadable segment, we
  // just pick the values directly from the segment as there
  // wouldnt be any slices within that
  if (segment->segmentType() != llvm::ELF::PT_LOAD) {
    Elf_Phdr *phdr = allocateProgramHeader(allocatedNew);
    phdr->p_type = segment->segmentType();
    phdr->p_offset = segment->fileOffset();
    phdr->p_vaddr = segment->virtualAddr();
    phdr->p_paddr = segment->virtualAddr();
    phdr->p_filesz = segment->fileSize();
    phdr->p_memsz = segment->memSize();
    phdr->p_flags = segment->flags();
    phdr->p_align = segment->alignment();
    this->_fsize = fileSize();
    this->_msize = this->_fsize;
    return allocatedNew;
  }
  // For all other segments, use the slice
  // to derive program headers
  for (auto slice : segment->slices()) {
    Elf_Phdr *phdr = allocateProgramHeader(allocatedNew);
    phdr->p_type = segment->segmentType();
    phdr->p_offset = slice->fileOffset();
    phdr->p_vaddr = slice->virtualAddr();
    phdr->p_paddr = slice->virtualAddr();
    phdr->p_filesz = slice->fileSize();
    phdr->p_memsz = slice->memSize();
    phdr->p_flags = segment->flags();
    phdr->p_align = slice->alignment();
    uint64_t segPageSize = segment->pageSize();
    uint64_t sliceAlign = slice->alignment();
    // Alignment of PT_LOAD segments are set to the page size, but if the
    // alignment of the slice is greater than the page size, set the alignment
    // of the segment appropriately.
    if (outputMagic != ELFLinkingContext::OutputMagic::NMAGIC &&
        outputMagic != ELFLinkingContext::OutputMagic::OMAGIC) {
      phdr->p_align =
          (phdr->p_type == llvm::ELF::PT_LOAD)
              ? (segPageSize < sliceAlign) ? sliceAlign : segPageSize
              : sliceAlign;
    } else
      phdr->p_align = slice->alignment();
  }
  this->_fsize = fileSize();
  this->_msize = this->_fsize;

  return allocatedNew;
}

template <class ELFT>
void ProgramHeader<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                                llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (auto phi : _ph) {
    memcpy(dest, phi, sizeof(Elf_Phdr));
    dest += sizeof(Elf_Phdr);
  }
}

template <class ELFT>
typename ProgramHeader<ELFT>::Elf_Phdr *
ProgramHeader<ELFT>::allocateProgramHeader(bool &allocatedNew) {
  Elf_Phdr *phdr;
  if (_phi == _ph.end()) {
    phdr = new (_allocator) Elf_Phdr;
    _ph.push_back(phdr);
    _phi = _ph.end();
    allocatedNew = true;
  } else {
    phdr = (*_phi);
    ++_phi;
  }
  return phdr;
}

template <class ELFT>
SectionHeader<ELFT>::SectionHeader(const ELFLinkingContext &ctx, int32_t order)
    : Chunk<ELFT>("shdr", Chunk<ELFT>::Kind::SectionHeader, ctx) {
  this->_fsize = 0;
  this->_alignment = 8;
  this->setOrder(order);
  // The first element in the list is always NULL
  auto *nullshdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
  ::memset(nullshdr, 0, sizeof(Elf_Shdr));
  _sectionInfo.push_back(nullshdr);
  this->_fsize += sizeof(Elf_Shdr);
}

template <class ELFT>
void SectionHeader<ELFT>::appendSection(OutputSection<ELFT> *section) {
  auto *shdr = new (_sectionAllocate.Allocate<Elf_Shdr>()) Elf_Shdr;
  shdr->sh_name = _stringSection->addString(section->name());
  shdr->sh_type = section->type();
  shdr->sh_flags = section->flags();
  shdr->sh_offset = section->fileOffset();
  shdr->sh_addr = section->virtualAddr();
  if (section->isLoadableSection())
    shdr->sh_size = section->memSize();
  else
    shdr->sh_size = section->fileSize();
  shdr->sh_link = section->link();
  shdr->sh_info = section->shinfo();
  shdr->sh_addralign = section->alignment();
  shdr->sh_entsize = section->entsize();
  _sectionInfo.push_back(shdr);
}

template <class ELFT>
void SectionHeader<ELFT>::updateSection(Section<ELFT> *section) {
  Elf_Shdr *shdr = _sectionInfo[section->ordinal()];
  shdr->sh_type = section->getType();
  shdr->sh_flags = section->getFlags();
  shdr->sh_offset = section->fileOffset();
  shdr->sh_addr = section->virtualAddr();
  shdr->sh_size = section->fileSize();
  shdr->sh_link = section->getLink();
  shdr->sh_info = section->getInfo();
  shdr->sh_addralign = section->alignment();
  shdr->sh_entsize = section->getEntSize();
}

template <class ELFT>
void SectionHeader<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                                llvm::FileOutputBuffer &buffer) {
  uint8_t *chunkBuffer = buffer.getBufferStart();
  uint8_t *dest = chunkBuffer + this->fileOffset();
  for (auto shi : _sectionInfo) {
    memcpy(dest, shi, sizeof(Elf_Shdr));
    dest += sizeof(Elf_Shdr);
  }
  _stringSection->write(writer, layout, buffer);
}

template class ELFHeader<ELF32LE>;
template class ELFHeader<ELF32BE>;
template class ELFHeader<ELF64LE>;
template class ELFHeader<ELF64BE>;

template class ProgramHeader<ELF32LE>;
template class ProgramHeader<ELF32BE>;
template class ProgramHeader<ELF64LE>;
template class ProgramHeader<ELF64BE>;

template class SectionHeader<ELF32LE>;
template class SectionHeader<ELF32BE>;
template class SectionHeader<ELF64LE>;
template class SectionHeader<ELF64BE>;

} // end namespace elf
} // end namespace lld
