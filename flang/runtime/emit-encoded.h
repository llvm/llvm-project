//===-- runtime/emit-encoded.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Templates for emitting CHARACTER values with conversion

#ifndef FORTRAN_RUNTIME_EMIT_ENCODED_H_
#define FORTRAN_RUNTIME_EMIT_ENCODED_H_

#include "connection.h"
#include "environment.h"
#include "tools.h"
#include "utf.h"

namespace Fortran::runtime::io {

template <typename CONTEXT, typename CHAR>
RT_API_ATTRS bool EmitEncoded(
    CONTEXT &to, const CHAR *data, std::size_t chars) {
  ConnectionState &connection{to.GetConnectionState()};
  if (connection.access == Access::Stream &&
      connection.internalIoCharKind == 0) {
    // Stream output: treat newlines as record advancements so that the left tab
    // limit is correctly managed
    while (const CHAR * nl{FindCharacter(data, CHAR{'\n'}, chars)}) {
      auto pos{static_cast<std::size_t>(nl - data)};
      if (!EmitEncoded(to, data, pos)) {
        return false;
      }
      data += pos + 1;
      chars -= pos + 1;
      to.AdvanceRecord();
    }
  }
  if (connection.useUTF8<CHAR>()) {
    using UnsignedChar = std::make_unsigned_t<CHAR>;
    const UnsignedChar *uData{reinterpret_cast<const UnsignedChar *>(data)};
    char buffer[256];
    std::size_t at{0};
    while (chars-- > 0) {
      auto len{EncodeUTF8(buffer + at, *uData++)};
      at += len;
      if (at + maxUTF8Bytes > sizeof buffer) {
        if (!to.Emit(buffer, at)) {
          return false;
        }
        at = 0;
      }
    }
    return at == 0 || to.Emit(buffer, at);
  } else {
    std::size_t internalKind = connection.internalIoCharKind;
    if (internalKind == 0 || internalKind == sizeof(CHAR)) {
      const char *rawData{reinterpret_cast<const char *>(data)};
      return to.Emit(rawData, chars * sizeof(CHAR), sizeof(CHAR));
    } else {
      // CHARACTER kind conversion for internal output
      while (chars-- > 0) {
        char32_t buffer = *data++;
        char *p{reinterpret_cast<char *>(&buffer)};
        if constexpr (!isHostLittleEndian) {
          p += sizeof(buffer) - internalKind;
        }
        if (!to.Emit(p, internalKind)) {
          return false;
        }
      }
      return true;
    }
  }
}

template <typename CONTEXT>
RT_API_ATTRS bool EmitAscii(CONTEXT &to, const char *data, std::size_t chars) {
  ConnectionState &connection{to.GetConnectionState()};
  if (connection.internalIoCharKind <= 1 &&
      connection.access != Access::Stream) {
    return to.Emit(data, chars);
  } else {
    return EmitEncoded(to, data, chars);
  }
}

template <typename CONTEXT>
RT_API_ATTRS bool EmitRepeated(CONTEXT &to, char ch, std::size_t n) {
  if (n <= 0) {
    return true;
  }
  ConnectionState &connection{to.GetConnectionState()};
  if (connection.internalIoCharKind <= 1 &&
      connection.access != Access::Stream) {
    // faster path, no encoding needed
    while (n-- > 0) {
      if (!to.Emit(&ch, 1)) {
        return false;
      }
    }
  } else {
    while (n-- > 0) {
      if (!EmitEncoded(to, &ch, 1)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_EMIT_ENCODED_H_
