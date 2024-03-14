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
  InternalDescriptorUnit(Scalar, std::size_t chars, int kind,
      const Terminator &terminator, bool allocateOwnOutput = false);
  InternalDescriptorUnit(const Descriptor &, const Terminator &);
  void EndIoStatement();

  bool Emit(const char *, std::size_t, IoErrorHandler &);
  std::size_t GetNextInputBytes(const char *&, IoErrorHandler &);
  bool AdvanceRecord(IoErrorHandler &);
  void BackspaceRecord(IoErrorHandler &);
  std::int64_t InquirePos();

private:
  Descriptor &descriptor() { return staticDescriptor_.descriptor(); }
  const Descriptor &descriptor() const {
    return staticDescriptor_.descriptor();
  }
  Scalar CurrentRecord() const {
    return descriptor().template ZeroBasedIndexedElement<char>(
        currentRecordNumber - 1);
  }
  void BlankFill(char *, std::size_t);
  void BlankFillOutputRecord();

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
