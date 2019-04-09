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

#ifndef COMGR_MSGPACK_H
#define COMGR_MSGPACK_H

#include "MsgPackReader.h"
#include "amd_comgr.h"
#include "llvm/Support/Casting.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace COMGR {
namespace msgpack {

class Node {
  virtual void anchor() = 0;
  const amd_comgr_metadata_kind_t Kind;

public:
  virtual ~Node() = default;

  amd_comgr_metadata_kind_t getKind() const { return Kind; }
  Node(amd_comgr_metadata_kind_t Kind) : Kind(Kind) {}
};

class Null : public Node {
  void anchor() override;

public:
  static bool classof(const Node *N) {
    return N->getKind() == AMD_COMGR_METADATA_KIND_NULL;
  }
  Null() : Node(AMD_COMGR_METADATA_KIND_NULL) {}
};

class String : public Node {
  void anchor() override;

public:
  std::string Value;

  static bool classof(const Node *N) {
    return N->getKind() == AMD_COMGR_METADATA_KIND_STRING;
  }
  static std::shared_ptr<String> make(std::string Value) {
    return std::make_shared<String>(std::move(Value));
  }
  String(std::string Value)
      : Node(AMD_COMGR_METADATA_KIND_STRING), Value(Value) {}
};

class Map : public Node {
  void anchor() override;

public:
  std::map<std::shared_ptr<Node>, std::shared_ptr<Node>> Elements;

  static bool classof(const Node *N) {
    return N->getKind() == AMD_COMGR_METADATA_KIND_MAP;
  }
  Map() : Node(AMD_COMGR_METADATA_KIND_MAP) {}
};

class List : public Node {
  void anchor() override;

public:
  std::vector<std::shared_ptr<Node>> Elements;

  static bool classof(const Node *N) {
    return N->getKind() == AMD_COMGR_METADATA_KIND_LIST;
  }
  List(size_t Length) : Node(AMD_COMGR_METADATA_KIND_LIST) {
    for (size_t I = 0; I < Length; ++I)
      Elements.emplace_back();
  }
};

amd_comgr_status_t parse(msgpack::Reader &MPReader, std::shared_ptr<Node> &Out);

} // namespace msgpack
} // namespace COMGR

#endif // COMGR_MSGPACK_H
