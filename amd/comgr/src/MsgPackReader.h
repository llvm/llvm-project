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

#ifndef LLVM_SUPPORT_MSGPACKREADER_H
#define LLVM_SUPPORT_MSGPACKREADER_H

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

using namespace llvm;

namespace COMGR {
namespace msgpack {

/// MessagePack types as defined in the standard, with the exception of Integer
/// being divided into a signed Int and unsigned UInt variant in order to map
/// directly to C++ types.
///
/// The types map onto corresponding union members of the \c Object struct.
enum class Type : uint8_t {
  Int,
  UInt,
  Nil,
  Boolean,
  Float,
  String,
  Binary,
  Array,
  Map,
  Extension,
};

/// MessagePack object, represented as a tagged union of C++ types.
///
/// All types except \c Type::Nil (which has only one value, and so is
/// completely represented by the \c Kind itself) map to a exactly one union
/// member.
struct Object {
  Type Kind;
  union {
    /// Value for \c Type::Int.
    int64_t Int;
    /// Value for \c Type::Uint.
    uint64_t UInt;
    /// Value for \c Type::Boolean.
    bool Bool;
    /// Value for \c Type::Float.
    double Float;
    /// Value for \c Type::String and \c Type::Binary.
    StringRef Raw;
    /// Value for \c Type::Array and \c Type::Map.
    size_t Length;
    /// Value for \c Type::Extension.
    struct {
      /// User-defined extension type.
      int8_t Type;
      /// Raw bytes of the extension object.
      StringRef Bytes;
    } Extension;
  };

  Object() : Kind(Type::Int), Int(0) {}
};

/// Reads MessagePack objects from memory, one at a time.
class Reader {
public:
  /// Construct a reader, keeping a reference to the \p InputBuffer.
  Reader(MemoryBufferRef InputBuffer);
  /// Construct a reader, keeping a reference to the \p Input.
  Reader(StringRef Input);

  Reader(const Reader &) = delete;
  Reader &operator=(const Reader &) = delete;

  /// Read one object from the input buffer, advancing past it.
  ///
  /// The \p Obj is updated with the kind of the object read, and the
  /// corresponding union member is updated.
  ///
  /// For the collection objects (Array and Map), only the length is read, and
  /// the caller must make and additional \c N calls (in the case of Array) or
  /// \c N*2 calls (in the case of Map) to \c Read to retrieve the collection
  /// elements.
  ///
  /// \param [out] Obj filled with next object on success.
  ///
  /// \returns true when object successfully read, otherwise false.
  bool read(Object &Obj);

  bool getFailed() { return Failed; }

private:
  MemoryBufferRef InputBuffer;
  StringRef::iterator Current;
  StringRef::iterator End;
  bool Failed = false;

  void setError(const Twine &Message) {
    if (Current >= End)
      Current = End - 1;
    if (!Failed)
      errs() << Message << '\n';
    Failed = true;
  }

  size_t remainingSpace() {
    // The rest of the code maintains the invariant that End >= Current, so
    // that this cast is always defined behavior.
    return static_cast<size_t>(End - Current);
  }

  template <class T> bool readRaw(Object &Obj);
  template <class T> bool readInt(Object &Obj);
  template <class T> bool readUInt(Object &Obj);
  template <class T> bool readLength(Object &Obj);
  template <class T> bool readExt(Object &Obj);
  bool createRaw(Object &Obj, uint32_t Size);
  bool createExt(Object &Obj, uint32_t Size);
};

} // end namespace msgpack
} // end namespace COMGR

#endif // LLVM_SUPPORT_MSGPACKREADER_H
