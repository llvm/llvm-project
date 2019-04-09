/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
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
 *     * Neither the names of Advanced Micro Devices, Inc. nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

#include "comgr-msgpack.h"

using namespace COMGR;
using namespace msgpack;

void Null::anchor() {}
void String::anchor() {}
void Map::anchor() {}
void List::anchor() {}

amd_comgr_status_t parseArray(msgpack::Object &Obj, msgpack::Reader &MPReader,
                              std::shared_ptr<Node> &Out) {
  std::shared_ptr<List> L = std::shared_ptr<List>(new List(Obj.Length));
  for (size_t I = 0; I < Obj.Length; ++I)
    if (auto Status = parse(MPReader, L->Elements[I]))
      return Status;
  Out = L;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t parseMap(msgpack::Object &Obj, msgpack::Reader &MPReader,
                            std::shared_ptr<Node> &Out) {
  std::shared_ptr<Map> M = std::shared_ptr<Map>(new Map());
  for (size_t I = 0; I < Obj.Length; ++I) {
    std::shared_ptr<Node> Key;
    if (auto Status = parse(MPReader, Key))
      return Status;
    std::shared_ptr<Node> Value;
    if (auto Status = parse(MPReader, Value))
      return Status;
    M->Elements.emplace(std::move(Key), std::move(Value));
  }
  Out = M;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t msgpack::parse(msgpack::Reader &MPReader,
                                  std::shared_ptr<Node> &Out) {
  msgpack::Object Obj;

  if (MPReader.read(Obj)) {
    switch (Obj.Kind) {
    case msgpack::Type::Nil:
      Out.reset(new Null());
      break;
    case msgpack::Type::Int:
      Out.reset(new String(std::to_string(Obj.Int)));
      break;
    case msgpack::Type::UInt:
      Out.reset(new String(std::to_string(Obj.UInt)));
      break;
    case msgpack::Type::Boolean:
      Out.reset(new String(std::to_string(Obj.Bool)));
      break;
    case msgpack::Type::Float:
      Out.reset(new String(std::to_string(Obj.Float)));
      break;
    case msgpack::Type::String:
    case msgpack::Type::Binary:
      Out.reset(new String(Obj.Raw.str()));
      break;
    case msgpack::Type::Extension:
      Out.reset(new String(Obj.Extension.Bytes.str()));
      break;
    case msgpack::Type::Array:
      if (auto Status = parseArray(Obj, MPReader, Out))
        return Status;
      break;
    case msgpack::Type::Map:
      if (auto Status = parseMap(Obj, MPReader, Out))
        return Status;
      break;
    }
  }
  if (MPReader.getFailed())
    return AMD_COMGR_STATUS_ERROR;

  return AMD_COMGR_STATUS_SUCCESS;
}
