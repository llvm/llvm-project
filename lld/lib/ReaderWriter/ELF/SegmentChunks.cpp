//===- lib/ReaderWriter/ELF/SegmentChunks.h -------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SegmentChunks.h"
#include "TargetLayout.h"

namespace lld {
namespace elf {

template <class ELFT>
bool SegmentSlice<ELFT>::compare_slices(SegmentSlice<ELFT> *a,
                                        SegmentSlice<ELFT> *b) {
  return a->startSection() < b->startSection();
}

template <class ELFT>
Segment<ELFT>::Segment(const ELFLinkingContext &ctx, StringRef name,
                       const typename TargetLayout<ELFT>::SegmentType type)
    : Chunk<ELFT>(name, Chunk<ELFT>::Kind::ELFSegment, ctx), _segmentType(type),
      _flags(0), _atomflags(0), _segmentFlags(false) {
  this->_alignment = 1;
  this->_fsize = 0;
  _outputMagic = ctx.getOutputMagic();
}

// This function actually is used, but not in all instantiations of Segment.
LLVM_ATTRIBUTE_UNUSED
static DefinedAtom::ContentPermissions toAtomPerms(uint64_t flags) {
  switch (flags & (SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR)) {
  case SHF_ALLOC | SHF_WRITE | SHF_EXECINSTR:
    return DefinedAtom::permRWX;
  case SHF_ALLOC | SHF_EXECINSTR:
    return DefinedAtom::permR_X;
  case SHF_ALLOC:
    return DefinedAtom::permR__;
  case SHF_ALLOC | SHF_WRITE:
    return DefinedAtom::permRW_;
  default:
    return DefinedAtom::permUnknown;
  }
}

// This function actually is used, but not in all instantiations of Segment.
LLVM_ATTRIBUTE_UNUSED
static DefinedAtom::ContentPermissions toAtomPermsSegment(uint64_t flags) {
  switch (flags & (llvm::ELF::PF_R | llvm::ELF::PF_W | llvm::ELF::PF_X)) {
  case llvm::ELF::PF_R | llvm::ELF::PF_W | llvm::ELF::PF_X:
    return DefinedAtom::permRWX;
  case llvm::ELF::PF_R | llvm::ELF::PF_X:
    return DefinedAtom::permR_X;
  case llvm::ELF::PF_R:
    return DefinedAtom::permR__;
  case llvm::ELF::PF_R | llvm::ELF::PF_W:
    return DefinedAtom::permRW_;
  default:
    return DefinedAtom::permUnknown;
  }
}

template <class ELFT> void Segment<ELFT>::append(Chunk<ELFT> *chunk) {
  _sections.push_back(chunk);
  Section<ELFT> *section = dyn_cast<Section<ELFT>>(chunk);
  if (!section)
    return;
  if (this->_alignment < section->alignment())
    this->_alignment = section->alignment();

  if (_segmentFlags)
    return;
  if (_flags < section->getFlags())
    _flags |= section->getFlags();
  if (_atomflags < toAtomPerms(_flags))
    _atomflags = toAtomPerms(_flags);
}

template <class ELFT>
bool Segment<ELFT>::compareSegments(Segment<ELFT> *sega, Segment<ELFT> *segb) {
  int64_t type1 = sega->segmentType();
  int64_t type2 = segb->segmentType();

  if (type1 == type2)
    return sega->atomflags() < segb->atomflags();

  // The single PT_PHDR segment is required to precede any loadable
  // segment.  We simply make it always first.
  if (type1 == llvm::ELF::PT_PHDR)
    return true;
  if (type2 == llvm::ELF::PT_PHDR)
    return false;

  // The single PT_INTERP segment is required to precede any loadable
  // segment.  We simply make it always second.
  if (type1 == llvm::ELF::PT_INTERP)
    return true;
  if (type2 == llvm::ELF::PT_INTERP)
    return false;

  // We then put PT_LOAD segments before any other segments.
  if (type1 == llvm::ELF::PT_LOAD)
    return true;
  if (type2 == llvm::ELF::PT_LOAD)
    return false;

  // We put the PT_GNU_RELRO segment last, because that is where the
  // dynamic linker expects to find it
  if (type1 == llvm::ELF::PT_GNU_RELRO)
    return false;
  if (type2 == llvm::ELF::PT_GNU_RELRO)
    return true;

  // We put the PT_TLS segment last except for the PT_GNU_RELRO
  // segment, because that is where the dynamic linker expects to find
  if (type1 == llvm::ELF::PT_TLS)
    return false;
  if (type2 == llvm::ELF::PT_TLS)
    return true;

  // Otherwise compare the types to establish an arbitrary ordering.
  // FIXME: Should figure out if we should just make all other types compare
  // equal, but if so, we should probably do the same for atom flags and change
  // users of this to use stable_sort.
  return type1 < type2;
}

template <class ELFT>
void Segment<ELFT>::assignFileOffsets(uint64_t startOffset) {
  uint64_t fileOffset = startOffset;
  uint64_t curSliceFileOffset = fileOffset;
  bool isDataPageAlignedForNMagic = false;
  bool alignSegments = this->_ctx.alignSegments();
  uint64_t p_align = this->_ctx.getPageSize();
  uint64_t lastVirtualAddress = 0;

  this->setFileOffset(startOffset);
  bool changeOffset = false;
  uint64_t newOffset = 0;
  for (auto &slice : slices()) {
    bool isFirstSection = true;
    for (auto section : slice->sections()) {
      // Handle linker script expressions, which may change the offset
      if (auto expr = dyn_cast<ExpressionChunk<ELFT>>(section)) {
        if (!isFirstSection) {
          changeOffset = true;
          newOffset = fileOffset + expr->virtualAddr() - lastVirtualAddress;
        }
        continue;
      }
      if (changeOffset) {
        changeOffset = false;
        fileOffset = newOffset;
      }
      // Align fileoffset to the alignment of the section.
      fileOffset = llvm::alignTo(fileOffset, section->alignment());
      // If the linker outputmagic is set to OutputMagic::NMAGIC, align the Data
      // to a page boundary
      if (isFirstSection &&
          _outputMagic != ELFLinkingContext::OutputMagic::NMAGIC &&
          _outputMagic != ELFLinkingContext::OutputMagic::OMAGIC) {
        // Align to a page only if the output is not
        // OutputMagic::NMAGIC/OutputMagic::OMAGIC
        if (alignSegments)
          fileOffset = llvm::alignTo(fileOffset, p_align);
        // Align according to ELF spec.
        // in p75, http://www.sco.com/developers/devspecs/gabi41.pdf
        uint64_t virtualAddress = slice->virtualAddr();
        Section<ELFT> *sect = dyn_cast<Section<ELFT>>(section);
        if (sect && sect->isLoadableSection() &&
            ((virtualAddress & (p_align - 1)) != (fileOffset & (p_align - 1))))
          fileOffset =
              llvm::alignTo(fileOffset, p_align) + (virtualAddress % p_align);
      } else if (!isDataPageAlignedForNMagic && needAlign(section)) {
        fileOffset = llvm::alignTo(fileOffset, this->_ctx.getPageSize());
        isDataPageAlignedForNMagic = true;
      }
      if (isFirstSection) {
        slice->setFileOffset(fileOffset);
        isFirstSection = false;
        curSliceFileOffset = fileOffset;
      }
      section->setFileOffset(fileOffset);
      fileOffset += section->fileSize();
      lastVirtualAddress = section->virtualAddr() + section->memSize();
    }
    changeOffset = false;
    slice->setFileSize(fileOffset - curSliceFileOffset);
  }
  this->setFileSize(fileOffset - startOffset);
}

/// \brief Assign virtual addresses to the slices
template <class ELFT> void Segment<ELFT>::assignVirtualAddress(uint64_t addr) {
  int startSection = 0;
  int currSection = 0;
  SectionIter startSectionIter;

  // slice align is set to the max alignment of the chunks that are
  // contained in the slice
  uint64_t sliceAlign = 0;
  // Current slice size
  uint64_t curSliceSize = 0;
  // Current Slice File Offset
  uint64_t curSliceAddress = 0;

  startSectionIter = _sections.begin();
  startSection = 0;
  bool isDataPageAlignedForNMagic = false;
  uint64_t startAddr = addr;
  SegmentSlice<ELFT> *slice = nullptr;
  uint64_t tlsStartAddr = 0;
  bool alignSegments = this->_ctx.alignSegments();
  StringRef prevOutputSectionName = StringRef();
  uint64_t tbssMemsize = 0;

  // If this is first section in the segment, page align the section start
  // address. The linker needs to align the data section to a page boundary
  // only if NMAGIC is set.
  auto si = _sections.begin();
  if (si != _sections.end()) {
    if (alignSegments &&
        _outputMagic != ELFLinkingContext::OutputMagic::NMAGIC &&
        _outputMagic != ELFLinkingContext::OutputMagic::OMAGIC) {
      // Align to a page only if the output is not
      // OutputMagic::NMAGIC/OutputMagic::OMAGIC
      startAddr = llvm::alignTo(startAddr, this->_ctx.getPageSize());
    } else if (needAlign(*si)) {
      // If the linker outputmagic is set to OutputMagic::NMAGIC, align the
      // Data to a page boundary.
      startAddr = llvm::alignTo(startAddr, this->_ctx.getPageSize());
      isDataPageAlignedForNMagic = true;
    }
    // align the startOffset to the section alignment
    uint64_t newAddr = llvm::alignTo(startAddr, (*si)->alignment());
    // Handle linker script expressions, which *may update newAddr* if the
    // expression assigns to "."
    if (auto expr = dyn_cast<ExpressionChunk<ELFT>>(*si))
      expr->evalExpr(newAddr);
    curSliceAddress = newAddr;
    sliceAlign = (*si)->alignment();
    (*si)->setVirtualAddr(curSliceAddress);

    // Handle TLS.
    if (auto section = dyn_cast<Section<ELFT>>(*si)) {
      if (section->getSegmentType() == llvm::ELF::PT_TLS) {
        tlsStartAddr = llvm::alignTo(tlsStartAddr, (*si)->alignment());
        section->assignVirtualAddress(tlsStartAddr);
        tlsStartAddr += (*si)->memSize();
      } else {
        section->assignVirtualAddress(newAddr);
      }
    }
    // TBSS section is special in that it doesn't contribute to memory of any
    // segment. If we see a tbss section, don't add memory size to addr The
    // fileOffset is automatically taken care of since TBSS section does not
    // end up using file size
    if ((*si)->order() != TargetLayout<ELFT>::ORDER_TBSS) {
      curSliceSize = (*si)->memSize();
      tbssMemsize = 0;
    } else {
      tbssMemsize = (*si)->memSize();
    }
    ++currSection;
    ++si;
  }

  uint64_t scriptAddr = 0;
  bool forceScriptAddr = false;
  for (auto e = _sections.end(); si != e; ++si) {
    uint64_t curAddr = curSliceAddress + curSliceSize;
    if (!isDataPageAlignedForNMagic && needAlign(*si)) {
      // If the linker outputmagic is set to OutputMagic::NMAGIC, align the
      // Data
      // to a page boundary
      curAddr = llvm::alignTo(curAddr, this->_ctx.getPageSize());
      isDataPageAlignedForNMagic = true;
    }
    uint64_t newAddr = llvm::alignTo(forceScriptAddr ? scriptAddr : curAddr,
                                     (*si)->alignment());
    forceScriptAddr = false;

    // Handle linker script expressions, which may force an address change if
    // the expression assigns to "."
    if (auto expr = dyn_cast<ExpressionChunk<ELFT>>(*si)) {
      uint64_t oldAddr = newAddr;
      expr->evalExpr(newAddr);
      if (oldAddr != newAddr) {
        forceScriptAddr = true;
        scriptAddr = newAddr;
      }
      (*si)->setVirtualAddr(newAddr);
      continue;
    }
    Section<ELFT> *sec = dyn_cast<Section<ELFT>>(*si);
    StringRef curOutputSectionName =
        sec ? sec->outputSectionName() : (*si)->name();
    bool autoCreateSlice = true;
    if (curOutputSectionName == prevOutputSectionName)
      autoCreateSlice = false;
    // If the newAddress computed is more than a page away, let's create
    // a separate segment, so that memory is not used up while running.
    // Dont create a slice, if the new section falls in the same output
    // section as the previous section.
    if (autoCreateSlice && ((newAddr - curAddr) > this->_ctx.getPageSize()) &&
        (_outputMagic != ELFLinkingContext::OutputMagic::NMAGIC &&
         _outputMagic != ELFLinkingContext::OutputMagic::OMAGIC)) {
      auto sliceIter =
          std::find_if(_segmentSlices.begin(), _segmentSlices.end(),
                       [startSection](SegmentSlice<ELFT> *s) -> bool {
                         return s->startSection() == startSection;
                       });
      if (sliceIter == _segmentSlices.end()) {
        slice = new (_segmentAllocate.Allocate<SegmentSlice<ELFT>>())
            SegmentSlice<ELFT>();
        _segmentSlices.push_back(slice);
      } else {
        slice = *sliceIter;
      }
      slice->setStart(startSection);
      slice->setSections(make_range(startSectionIter, si));
      slice->setMemSize(curSliceSize);
      slice->setAlign(sliceAlign);
      slice->setVirtualAddr(curSliceAddress);
      // Start new slice
      curSliceAddress = newAddr;
      if ((*si)->order() == TargetLayout<ELFT>::ORDER_TBSS)
        curSliceAddress += tbssMemsize;
      (*si)->setVirtualAddr(curSliceAddress);
      startSectionIter = si;
      startSection = currSection;
      if (auto section = dyn_cast<Section<ELFT>>(*si))
        section->assignVirtualAddress(newAddr);
      curSliceSize = newAddr - curSliceAddress + (*si)->memSize();
      sliceAlign = (*si)->alignment();
    } else {
      if (sliceAlign < (*si)->alignment())
        sliceAlign = (*si)->alignment();
      if ((*si)->order() == TargetLayout<ELFT>::ORDER_TBSS)
        newAddr += tbssMemsize;
      (*si)->setVirtualAddr(newAddr);
      // Handle TLS.
      if (auto section = dyn_cast<Section<ELFT>>(*si)) {
        if (section->getSegmentType() == llvm::ELF::PT_TLS) {
          tlsStartAddr = llvm::alignTo(tlsStartAddr, (*si)->alignment());
          section->assignVirtualAddress(tlsStartAddr);
          tlsStartAddr += (*si)->memSize();
        } else {
          section->assignVirtualAddress(newAddr);
        }
      }
      // TBSS section is special in that it doesn't contribute to memory of
      // any segment. If we see a tbss section, don't add memory size to addr
      // The fileOffset is automatically taken care of since TBSS section does
      // not end up using file size.
      if ((*si)->order() != TargetLayout<ELFT>::ORDER_TBSS) {
        curSliceSize = newAddr - curSliceAddress + (*si)->memSize();
        tbssMemsize = 0;
      } else {
        // Although TBSS section does not contribute to memory of any segment,
        // we still need to keep track its total size to correct write it
        // down.  Since it is done based on curSliceAddress, we need to add
        // add it to virtual address.
        tbssMemsize = (*si)->memSize();
      }
    }
    prevOutputSectionName = curOutputSectionName;
    ++currSection;
  }

  auto sliceIter = std::find_if(_segmentSlices.begin(), _segmentSlices.end(),
                                [startSection](SegmentSlice<ELFT> *s) -> bool {
                                  return s->startSection() == startSection;
                                });
  if (sliceIter == _segmentSlices.end()) {
    slice = new (_segmentAllocate.Allocate<SegmentSlice<ELFT>>())
        SegmentSlice<ELFT>();
    _segmentSlices.push_back(slice);
  } else {
    slice = *sliceIter;
  }

  slice->setStart(startSection);
  slice->setVirtualAddr(curSliceAddress);
  slice->setMemSize(curSliceSize);
  slice->setSections(make_range(startSectionIter, _sections.end()));
  slice->setAlign(sliceAlign);

  // Set the segment memory size and the virtual address.
  this->setMemSize(curSliceAddress - startAddr + curSliceSize);
  this->setVirtualAddr(startAddr);
  std::stable_sort(_segmentSlices.begin(), _segmentSlices.end(),
                   SegmentSlice<ELFT>::compare_slices);
}

// Write the Segment
template <class ELFT>
void Segment<ELFT>::write(ELFWriter *writer, TargetLayout<ELFT> &layout,
                          llvm::FileOutputBuffer &buffer) {
  for (auto slice : slices())
    for (auto section : slice->sections())
      section->write(writer, layout, buffer);
}

template <class ELFT> int64_t Segment<ELFT>::flags() const {
  if (_segmentFlags)
    return (int64_t)_flags;

  int64_t fl = 0;
  if (_flags & llvm::ELF::SHF_ALLOC)
    fl |= llvm::ELF::PF_R;
  if (_flags & llvm::ELF::SHF_WRITE)
    fl |= llvm::ELF::PF_W;
  if (_flags & llvm::ELF::SHF_EXECINSTR)
    fl |= llvm::ELF::PF_X;
  return fl;
}

template <class ELFT> void Segment<ELFT>::setSegmentFlags(uint64_t flags) {
  assert(!_segmentFlags && "Segment flags have already been set");
  _segmentFlags = true;
  _flags = flags;
  _atomflags = toAtomPermsSegment(flags);
}

template <class ELFT> void Segment<ELFT>::finalize() {
  // We want to finalize the segment values for now only for non loadable
  // segments, since those values are not set in the Layout
  if (_segmentType == llvm::ELF::PT_LOAD)
    return;
  // The size is the difference of the
  // last section to the first section, especially for TLS because
  // the TLS segment contains both .tdata/.tbss
  this->setFileOffset(_sections.front()->fileOffset());
  this->setVirtualAddr(_sections.front()->virtualAddr());
  size_t startFileOffset = _sections.front()->fileOffset();
  size_t startAddr = _sections.front()->virtualAddr();
  for (auto ai : _sections) {
    this->_fsize = ai->fileOffset() + ai->fileSize() - startFileOffset;
    this->_msize = ai->virtualAddr() + ai->memSize() - startAddr;
  }
}

template <class ELFT> int Segment<ELFT>::getContentType() const {
  int64_t fl = flags();
  switch (_segmentType) {
  case llvm::ELF::PT_LOAD: {
    if (fl && llvm::ELF::PF_X)
      return Chunk<ELFT>::ContentType::Code;
    if (fl && llvm::ELF::PF_W)
      return Chunk<ELFT>::ContentType::Data;
  }
  case llvm::ELF::PT_TLS:
    return Chunk<ELFT>::ContentType::TLS;
  case llvm::ELF::PT_NOTE:
    return Chunk<ELFT>::ContentType::Note;
  default:
    return Chunk<ELFT>::ContentType::Unknown;
  }
}

template <class ELFT> int64_t Segment<ELFT>::atomflags() const {
  switch (_atomflags) {
  case DefinedAtom::permUnknown:
    return permUnknown;
  case DefinedAtom::permRWX:
    return permRWX;
  case DefinedAtom::permR_X:
    return permRX;
  case DefinedAtom::permR__:
    return permR;
  case DefinedAtom::permRW_L:
    return permRWL;
  case DefinedAtom::permRW_:
    return permRW;
  case DefinedAtom::perm___:
  default:
    return permNonAccess;
  }
}

/// \brief Check if the chunk needs to be aligned
template <class ELFT> bool Segment<ELFT>::needAlign(Chunk<ELFT> *chunk) const {
  if (chunk->getContentType() == Chunk<ELFT>::ContentType::Data &&
      _outputMagic == ELFLinkingContext::OutputMagic::NMAGIC)
    return true;
  return false;
}

template <class ELFT> void ProgramHeaderSegment<ELFT>::finalize() {
  // If the segment is of type Program Header, then the values fileOffset
  // and the fileSize need to be picked up from the last section, the first
  // section points to the ELF header and the second chunk points to the
  // actual program headers
  this->setFileOffset(this->_sections.back()->fileOffset());
  this->setVirtualAddr(this->_sections.back()->virtualAddr());
  this->_fsize = this->_sections.back()->fileSize();
  this->_msize = this->_sections.back()->memSize();
}

#define INSTANTIATE(klass)        \
  template class klass<ELF32LE>;  \
  template class klass<ELF32BE>;  \
  template class klass<ELF64LE>;  \
  template class klass<ELF64BE>

INSTANTIATE(ExpressionChunk);
INSTANTIATE(ProgramHeaderSegment);
INSTANTIATE(Segment);
INSTANTIATE(SegmentSlice);

} // end namespace elf
} // end namespace lld
