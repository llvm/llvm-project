//===- WriterUtils.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_WASM_WRITERUTILS_H
#define LLD_WASM_WRITERUTILS_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/raw_ostream.h"

using llvm::raw_ostream;

// Needed for WasmSignatureDenseMapInfo
inline bool operator==(const llvm::wasm::WasmSignature &LHS,
                       const llvm::wasm::WasmSignature &RHS) {
  return LHS.ReturnType == RHS.ReturnType && LHS.ParamTypes == RHS.ParamTypes;
}

inline bool operator!=(const llvm::wasm::WasmSignature &LHS,
                       const llvm::wasm::WasmSignature &RHS) {
  return !(LHS == RHS);
}

// Used for general comparison
inline bool operator==(const llvm::wasm::WasmGlobalType &LHS,
                       const llvm::wasm::WasmGlobalType &RHS) {
  return LHS.Type == RHS.Type && LHS.Mutable == RHS.Mutable;
}

inline bool operator!=(const llvm::wasm::WasmGlobalType &LHS,
                       const llvm::wasm::WasmGlobalType &RHS) {
  return !(LHS == RHS);
}

namespace lld {
namespace wasm {

void debugWrite(uint64_t Offset, const Twine &Msg);

void writeUleb128(raw_ostream &OS, uint32_t Number, const Twine &Msg);

void writeSleb128(raw_ostream &OS, int32_t Number, const Twine &Msg);

void writeBytes(raw_ostream &OS, const char *Bytes, size_t count,
                const Twine &Msg);

void writeStr(raw_ostream &OS, StringRef String, const Twine &Msg);

void writeU8(raw_ostream &OS, uint8_t byte, const Twine &Msg);

void writeU32(raw_ostream &OS, uint32_t Number, const Twine &Msg);

void writeValueType(raw_ostream &OS, uint8_t Type, const Twine &Msg);

void writeSig(raw_ostream &OS, const llvm::wasm::WasmSignature &Sig);

void writeInitExpr(raw_ostream &OS, const llvm::wasm::WasmInitExpr &InitExpr);

void writeLimits(raw_ostream &OS, const llvm::wasm::WasmLimits &Limits);

void writeGlobalType(raw_ostream &OS, const llvm::wasm::WasmGlobalType &Type);

void writeGlobal(raw_ostream &OS, const llvm::wasm::WasmGlobal &Global);

void writeImport(raw_ostream &OS, const llvm::wasm::WasmImport &Import);

void writeExport(raw_ostream &OS, const llvm::wasm::WasmExport &Export);

} // namespace wasm

std::string toString(const llvm::wasm::ValType Type);
std::string toString(const llvm::wasm::WasmSignature &Sig);
std::string toString(const llvm::wasm::WasmGlobalType &Sig);

} // namespace lld

#endif // LLD_WASM_WRITERUTILS_H
