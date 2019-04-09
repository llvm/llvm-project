/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#ifndef LLVM_BINARYFORMAT_MSGPACK_H
#define LLVM_BINARYFORMAT_MSGPACK_H

#include "llvm/Support/DataTypes.h"
#include "llvm/Support/Endian.h"

namespace COMGR {
namespace msgpack {

/// The endianness of all multi-byte encoded values in MessagePack.
constexpr llvm::support::endianness Endianness = llvm::support::big;

/// The first byte identifiers of MessagePack object formats.
namespace FirstByte {
#define HANDLE_MP_FIRST_BYTE(ID, NAME) constexpr uint8_t NAME = ID;
#include "MsgPack.def"
} // namespace FirstByte

/// Most significant bits used to identify "Fix" variants in MessagePack.
///
/// For example, FixStr objects encode their size in the five least significant
/// bits of their first byte, which is identified by the bit pattern "101" in
/// the three most significant bits. So FixBits::String contains 0b10100000.
///
/// A corresponding mask of the bit pattern is found in \c FixBitsMask.
namespace FixBits {
#define HANDLE_MP_FIX_BITS(ID, NAME) constexpr uint8_t NAME = ID;
#include "MsgPack.def"
} // namespace FixBits

/// Mask of bits used to identify "Fix" variants in MessagePack.
///
/// For example, FixStr objects encode their size in the five least significant
/// bits of their first byte, which is identified by the bit pattern "101" in
/// the three most significant bits. So FixBitsMask::String contains
/// 0b11100000.
///
/// The corresponding bit pattern to mask for is found in FixBits.
namespace FixBitsMask {
#define HANDLE_MP_FIX_BITS_MASK(ID, NAME) constexpr uint8_t NAME = ID;
#include "MsgPack.def"
} // namespace FixBitsMask

/// The maximum value or size encodable in "Fix" variants of formats.
///
/// For example, FixStr objects encode their size in the five least significant
/// bits of their first byte, so the largest encodable size is 0b00011111.
namespace FixMax {
#define HANDLE_MP_FIX_MAX(ID, NAME) constexpr uint8_t NAME = ID;
#include "MsgPack.def"
} // namespace FixMax

/// The exact size encodable in "Fix" variants of formats.
///
/// The only objects for which an exact size makes sense are of Extension type.
///
/// For example, FixExt4 stores an extension type containing exactly four bytes.
namespace FixLen {
#define HANDLE_MP_FIX_LEN(ID, NAME) constexpr uint8_t NAME = ID;
#include "MsgPack.def"
} // namespace FixLen

/// The minimum value or size encodable in "Fix" variants of formats.
///
/// The only object for which a minimum makes sense is a negative FixNum.
///
/// Negative FixNum objects encode their signed integer value in one byte, but
/// they must have the pattern "111" as their three most significant bits. This
/// means all values are negative, and the smallest representable value is
/// 0b11100000.
namespace FixMin {
#define HANDLE_MP_FIX_MIN(ID, NAME) constexpr int8_t NAME = ID;
#include "MsgPack.def"
} // namespace FixMin

} // end namespace msgpack
} // end namespace COMGR

#endif // LLVM_BINARYFORMAT_MSGPACK_H
