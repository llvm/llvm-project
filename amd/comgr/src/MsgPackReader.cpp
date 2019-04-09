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

#include "MsgPackReader.h"
#include "MsgPack.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::support;
using namespace COMGR;
using namespace COMGR::msgpack;

Reader::Reader(MemoryBufferRef InputBuffer)
    : InputBuffer(InputBuffer), Current(InputBuffer.getBufferStart()),
      End(InputBuffer.getBufferEnd()) {}

Reader::Reader(StringRef Input) : Reader({Input, "MsgPack"}) {}

bool Reader::read(Object &Obj) {
  if (Current == End) {
    setError("Attempted to read at end of stream");
    return false;
  }

  uint8_t FB = static_cast<uint8_t>(*Current++);

  switch (FB) {
  case FirstByte::Nil:
    Obj.Kind = Type::Nil;
    return true;
  case FirstByte::True:
    Obj.Kind = Type::Boolean;
    Obj.Bool = true;
    return true;
  case FirstByte::False:
    Obj.Kind = Type::Boolean;
    Obj.Bool = false;
    return true;
  case FirstByte::Int8:
    Obj.Kind = Type::Int;
    return readInt<int8_t>(Obj);
  case FirstByte::Int16:
    Obj.Kind = Type::Int;
    return readInt<int16_t>(Obj);
  case FirstByte::Int32:
    Obj.Kind = Type::Int;
    return readInt<int32_t>(Obj);
  case FirstByte::Int64:
    Obj.Kind = Type::Int;
    return readInt<int64_t>(Obj);
  case FirstByte::UInt8:
    Obj.Kind = Type::UInt;
    return readUInt<uint8_t>(Obj);
  case FirstByte::UInt16:
    Obj.Kind = Type::UInt;
    return readUInt<uint16_t>(Obj);
  case FirstByte::UInt32:
    Obj.Kind = Type::UInt;
    return readUInt<uint32_t>(Obj);
  case FirstByte::UInt64:
    Obj.Kind = Type::UInt;
    return readUInt<uint64_t>(Obj);
  case FirstByte::Float32:
    Obj.Kind = Type::Float;
    if (sizeof(float) > remainingSpace()) {
      setError("Invalid Float32 with insufficient payload");
      return false;
    }
    Obj.Float = BitsToFloat(endian::read<uint32_t, Endianness>(Current));
    Current += sizeof(float);
    return true;
  case FirstByte::Float64:
    Obj.Kind = Type::Float;
    if (sizeof(double) > remainingSpace()) {
      setError("Invalid Float64 with insufficient payload");
      return false;
    }
    Obj.Float = BitsToDouble(endian::read<uint64_t, Endianness>(Current));
    Current += sizeof(double);
    return true;
  case FirstByte::Str8:
    Obj.Kind = Type::String;
    return readRaw<uint8_t>(Obj);
  case FirstByte::Str16:
    Obj.Kind = Type::String;
    return readRaw<uint16_t>(Obj);
  case FirstByte::Str32:
    Obj.Kind = Type::String;
    return readRaw<uint32_t>(Obj);
  case FirstByte::Bin8:
    Obj.Kind = Type::Binary;
    return readRaw<uint8_t>(Obj);
  case FirstByte::Bin16:
    Obj.Kind = Type::Binary;
    return readRaw<uint16_t>(Obj);
  case FirstByte::Bin32:
    Obj.Kind = Type::Binary;
    return readRaw<uint32_t>(Obj);
  case FirstByte::Array16:
    Obj.Kind = Type::Array;
    return readLength<uint16_t>(Obj);
  case FirstByte::Array32:
    Obj.Kind = Type::Array;
    return readLength<uint32_t>(Obj);
  case FirstByte::Map16:
    Obj.Kind = Type::Map;
    return readLength<uint16_t>(Obj);
  case FirstByte::Map32:
    Obj.Kind = Type::Map;
    return readLength<uint32_t>(Obj);
  case FirstByte::FixExt1:
    Obj.Kind = Type::Extension;
    return createExt(Obj, FixLen::Ext1);
  case FirstByte::FixExt2:
    Obj.Kind = Type::Extension;
    return createExt(Obj, FixLen::Ext2);
  case FirstByte::FixExt4:
    Obj.Kind = Type::Extension;
    return createExt(Obj, FixLen::Ext4);
  case FirstByte::FixExt8:
    Obj.Kind = Type::Extension;
    return createExt(Obj, FixLen::Ext8);
  case FirstByte::FixExt16:
    Obj.Kind = Type::Extension;
    return createExt(Obj, FixLen::Ext16);
  case FirstByte::Ext8:
    Obj.Kind = Type::Extension;
    return readExt<uint8_t>(Obj);
  case FirstByte::Ext16:
    Obj.Kind = Type::Extension;
    return readExt<uint16_t>(Obj);
  case FirstByte::Ext32:
    Obj.Kind = Type::Extension;
    return readExt<uint32_t>(Obj);
  }

  if ((FB & FixBitsMask::NegativeInt) == FixBits::NegativeInt) {
    Obj.Kind = Type::Int;
    int8_t I;
    static_assert(sizeof(I) == sizeof(FB), "Unexpected type sizes");
    memcpy(&I, &FB, sizeof(FB));
    Obj.Int = I;
    return true;
  }

  if ((FB & FixBitsMask::PositiveInt) == FixBits::PositiveInt) {
    Obj.Kind = Type::UInt;
    Obj.UInt = FB;
    return true;
  }

  if ((FB & FixBitsMask::String) == FixBits::String) {
    Obj.Kind = Type::String;
    uint8_t Size = FB & ~FixBitsMask::String;
    return createRaw(Obj, Size);
  }

  if ((FB & FixBitsMask::Array) == FixBits::Array) {
    Obj.Kind = Type::Array;
    Obj.Length = FB & ~FixBitsMask::Array;
    return true;
  }

  if ((FB & FixBitsMask::Map) == FixBits::Map) {
    Obj.Kind = Type::Map;
    Obj.Length = FB & ~FixBitsMask::Map;
    return true;
  }

  setError("Invalid first byte");
  return false;
}

template <class T> bool Reader::readRaw(Object &Obj) {
  if (sizeof(T) > remainingSpace()) {
    setError("Invalid Raw with insufficient payload");
    return false;
  }
  T Size = endian::read<T, Endianness>(Current);
  Current += sizeof(T);
  return createRaw(Obj, Size);
}

template <class T> bool Reader::readInt(Object &Obj) {
  if (sizeof(T) > remainingSpace()) {
    setError("Invalid Int with insufficient payload");
    return false;
  }
  Obj.Int = static_cast<int64_t>(endian::read<T, Endianness>(Current));
  Current += sizeof(T);
  return true;
}

template <class T> bool Reader::readUInt(Object &Obj) {
  if (sizeof(T) > remainingSpace()) {
    setError("Invalid Int with insufficient payload");
    return false;
  }
  Obj.UInt = static_cast<uint64_t>(endian::read<T, Endianness>(Current));
  Current += sizeof(T);
  return true;
}

template <class T> bool Reader::readLength(Object &Obj) {
  if (sizeof(T) > remainingSpace()) {
    setError("Invalid Map/Array with invalid length");
    return false;
  }
  Obj.Length = static_cast<size_t>(endian::read<T, Endianness>(Current));
  Current += sizeof(T);
  return true;
}

template <class T> bool Reader::readExt(Object &Obj) {
  if (sizeof(T) > remainingSpace()) {
    setError("Invalid Ext with invalid length");
    return false;
  }
  T Size = endian::read<T, Endianness>(Current);
  Current += sizeof(T);
  return createExt(Obj, Size);
}

bool Reader::createRaw(Object &Obj, uint32_t Size) {
  if (Size > remainingSpace()) {
    setError("Invalid Raw with insufficient payload");
    return false;
  }
  Obj.Raw = StringRef(Current, Size);
  Current += Size;
  return true;
}

bool Reader::createExt(Object &Obj, uint32_t Size) {
  if (Current == End) {
    setError("Invalid Ext with no type");
    return false;
  }
  Obj.Extension.Type = *Current++;
  if (Size > remainingSpace()) {
    setError("Invalid Ext with insufficient payload");
    return false;
  }
  Obj.Extension.Bytes = StringRef(Current, Size);
  Current += Size;
  return true;
}
