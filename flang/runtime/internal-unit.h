//===-- runtime/internal-unit.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Fortran internal I/O "units"

#ifndef FORTRAN_RUNTIME_IO_INTERNAL_UNIT_H_
#define FORTRAN_RUNTIME_IO_INTERNAL_UNIT_H_

#include "connection.h"
#include "flang/Runtime/api-attrs.h"
#include "flang/Runtime/descriptor.h"
#include <cinttypes>
#include <type_traits>

namespace Fortran::runtime::io {

class IoErrorHandler;

// Points to (but does not own) a CHARACTER scalar or array for internal I/O.
// The internal unit does not own the scalar buffer unless it is constructed
// with allocateOwnOutput set to true: in this case, it owns the buffer
// and also prints it to stdout at the end of the statement.
// This is used to support output on offload devices.
// Does not buffer.
template <Direction DIR> class InternalDescriptorUnit : public ConnectionState {
public:
  using Scalar =
      std::conditional_t<DIR == Direction::Input, const char *, char *>;
  RT_API_ATTRS InternalDescriptorUnit(Scalar, std::size_t chars, int kind,
      const Terminator &terminator, bool allocateOwnOutput = false);
  RT_API_ATTRS InternalDescriptorUnit(const Descriptor &, const Terminator &);
  RT_API_ATTRS void EndIoStatement();

  RT_API_ATTRS bool Emit(const char *, std::size_t, IoErrorHandler &);
  RT_API_ATTRS std::size_t GetNextInputBytes(const char *&, IoErrorHandler &);
  RT_API_ATTRS bool AdvanceRecord(IoErrorHandler &);
  RT_API_ATTRS void BackspaceRecord(IoErrorHandler &);
  RT_API_ATTRS std::int64_t InquirePos();

private:
  RT_API_ATTRS Descriptor &descriptor() {
    return staticDescriptor_.descriptor();
  }
  RT_API_ATTRS const Descriptor &descriptor() const {
    return staticDescriptor_.descriptor();
  }
  RT_API_ATTRS Scalar CurrentRecord() const {
    return descriptor().template ZeroBasedIndexedElement<char>(
        currentRecordNumber - 1);
  }
  RT_API_ATTRS void BlankFill(char *, std::size_t);
  RT_API_ATTRS void BlankFillOutputRecord();

  StaticDescriptor<maxRank, true /*addendum*/> staticDescriptor_;
  RT_OFFLOAD_VAR_GROUP_BEGIN
  static constexpr std::size_t ownBufferSizeInBytes{1024};
  RT_OFFLOAD_VAR_GROUP_END
  bool usesOwnBuffer{false};
};

extern template class InternalDescriptorUnit<Direction::Output>;
extern template class InternalDescriptorUnit<Direction::Input>;
} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_INTERNAL_UNIT_H_
