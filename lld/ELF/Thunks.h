//===- Thunks.h --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_THUNKS_H
#define LLD_ELF_THUNKS_H

#include "Relocations.h"

namespace lld {
namespace elf {
class SymbolBody;
template <class ELFT> class InputSection;

// Class to describe an instance of a Thunk.
// A Thunk is a code-sequence inserted by the linker in between a caller and
// the callee. The relocation to the callee is redirected to the Thunk, which
// after executing transfers control to the callee. Typical uses of Thunks
// include transferring control from non-pi to pi and changing state on
// targets like ARM.
//
// Thunks can be created for DefinedRegular and Shared Symbols. The Thunk
// is stored in a field of the Symbol Destination.
// Thunks to be written to an InputSection are recorded by the InputSection.
template <class ELFT> class Thunk {
  typedef typename ELFT::uint uintX_t;

public:
  Thunk(const SymbolBody &Destination, const InputSection<ELFT> &Owner);
  virtual ~Thunk();

  virtual uint32_t size() const { return 0; }
  virtual void writeTo(uint8_t *Buf) const {}
  uintX_t getVA() const;

protected:
  const SymbolBody &Destination;
  const InputSection<ELFT> &Owner;
  uint64_t Offset;
};

// For a Relocation to symbol S from InputSection Src, create a Thunk and
// update the fields of S and the InputSection that the Thunk body will be
// written to. At present there are implementations for ARM and Mips Thunks.
template <class ELFT>
void addThunk(uint32_t RelocType, SymbolBody &S, InputSection<ELFT> &Src);

} // namespace elf
} // namespace lld

#endif
