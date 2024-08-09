//===------------------------- SocketMsgSupport.h -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_socket_stream.h"

namespace clang::tooling::cc1modbuildd {

enum class ActionType { HANDSHAKE };
enum class StatusType { REQUEST, SUCCESS, FAILURE };

struct BaseMsg {
  ActionType MsgAction;
  StatusType MsgStatus;

  BaseMsg() = default;
  BaseMsg(ActionType Action, StatusType Status)
      : MsgAction(Action), MsgStatus(Status) {}
};

struct HandshakeMsg : public BaseMsg {
  HandshakeMsg() = default;
  HandshakeMsg(ActionType Action, StatusType Status)
      : BaseMsg(Action, Status) {}
};

llvm::Expected<std::string>
readBufferFromSocket(llvm::raw_socket_stream &Socket);
llvm::Error writeBufferToSocket(llvm::raw_socket_stream &Socket,
                                llvm::StringRef Buffer);

template <typename T> std::string convertMsgStructToBuffer(T MsgStruct) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value);

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::yaml::Output YamlOut(OS);

  // TODO confirm yaml::Output does not have any error messages
  YamlOut << MsgStruct;

  return Buffer;
}

template <typename T>
llvm::Expected<T> convertBufferToMsgStruct(llvm::StringRef Buffer) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value);

  T MsgStruct;
  llvm::yaml::Input YamlIn(Buffer);
  YamlIn >> MsgStruct;

  // YamlIn.error() dumps an error message if there is one
  if (YamlIn.error())
    return llvm::make_error<llvm::StringError>(
        "Syntax or semantic error during YAML parsing",
        llvm::inconvertibleErrorCode());

  return MsgStruct;
}

template <typename T>
llvm::Expected<T> readMsgStructFromSocket(llvm::raw_socket_stream &Socket) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value);

  llvm::Expected<std::string> MaybeBuffer = readBufferFromSocket(Socket);
  if (!MaybeBuffer)
    return std::move(MaybeBuffer.takeError());
  std::string Buffer = std::move(*MaybeBuffer);

  llvm::Expected<T> MaybeMsgStruct = convertBufferToMsgStruct<T>(Buffer);
  if (!MaybeMsgStruct)
    return std::move(MaybeMsgStruct.takeError());
  return std::move(*MaybeMsgStruct);
}

template <typename T>
llvm::Error writeMsgStructToSocket(llvm::raw_socket_stream &Socket,
                                   T MsgStruct) {
  static_assert(std::is_base_of<cc1modbuildd::BaseMsg, T>::value);

  std::string Buffer = convertMsgStructToBuffer(MsgStruct);
  if (llvm::Error Err = writeBufferToSocket(Socket, Buffer))
    return Err;
  return llvm::Error::success();
}

} // namespace clang::tooling::cc1modbuildd

namespace llvm {
namespace yaml {
template <>
struct ScalarEnumerationTraits<clang::tooling::cc1modbuildd::StatusType> {
  static void enumeration(IO &Io, clang::tooling::cc1modbuildd::StatusType &S) {
    Io.enumCase(S, "REQUEST",
                clang::tooling::cc1modbuildd::StatusType::REQUEST);
    Io.enumCase(S, "SUCCESS",
                clang::tooling::cc1modbuildd::StatusType::SUCCESS);
    Io.enumCase(S, "FAILURE",
                clang::tooling::cc1modbuildd::StatusType::FAILURE);
  }
};

template <>
struct ScalarEnumerationTraits<clang::tooling::cc1modbuildd::ActionType> {
  static void enumeration(IO &Io, clang::tooling::cc1modbuildd::ActionType &A) {
    Io.enumCase(A, "HANDSHAKE",
                clang::tooling::cc1modbuildd::ActionType::HANDSHAKE);
  }
};

template <> struct MappingTraits<clang::tooling::cc1modbuildd::HandshakeMsg> {
  static void mapping(IO &Io, clang::tooling::cc1modbuildd::HandshakeMsg &H) {
    Io.mapRequired("Action", H.MsgAction);
    Io.mapRequired("Status", H.MsgStatus);
  }
};
} // namespace yaml
} // namespace llvm
#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H
