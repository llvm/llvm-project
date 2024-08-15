//===-------------------------- SocketSupport.h ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETSUPPORT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETSUPPORT_H

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_socket_stream.h"

namespace clang::tooling::cc1modbuildd {

enum class ActionType { REGISTER, HANDSHAKE };
enum class StatusType { REQUEST, SUCCESS, FAILURE };

struct BaseMsg {
  ActionType MsgAction;
  StatusType MsgStatus;

  BaseMsg() = default;
  BaseMsg(ActionType Action, StatusType Status)
      : MsgAction(Action), MsgStatus(Status) {}
};

struct RegisterMsg : public BaseMsg {
  std::optional<std::string> WorkingDirectory;
  // The scanner requires the path to the clang executable
  std::optional<std::string> ExecutablePath;
  // Does not include executable
  std::optional<std::vector<std::string>> CC1CommandLine;

  RegisterMsg() = default;

  RegisterMsg(ActionType Action, StatusType Status,
              const std::optional<std::string> &CurrentWD,
              const std::optional<std::string> &Argv0,
              const std::optional<std::vector<std::string>> &Argv)
      : BaseMsg(Action, Status), WorkingDirectory(CurrentWD),
        ExecutablePath(Argv0), CC1CommandLine(Argv) {}

  RegisterMsg(ActionType Action, StatusType Status)
      : BaseMsg(Action, Status), WorkingDirectory(std::nullopt),
        ExecutablePath(std::nullopt), CC1CommandLine(std::nullopt) {}
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

template <>
struct llvm::yaml::ScalarEnumerationTraits<
    clang::tooling::cc1modbuildd::StatusType> {
  static void enumeration(IO &Io,
                          clang::tooling::cc1modbuildd::StatusType &Value) {
    Io.enumCase(Value, "REQUEST",
                clang::tooling::cc1modbuildd::StatusType::REQUEST);
    Io.enumCase(Value, "SUCCESS",
                clang::tooling::cc1modbuildd::StatusType::SUCCESS);
    Io.enumCase(Value, "FAILURE",
                clang::tooling::cc1modbuildd::StatusType::FAILURE);
  }
};

template <>
struct llvm::yaml::ScalarEnumerationTraits<
    clang::tooling::cc1modbuildd::ActionType> {
  static void enumeration(IO &Io,
                          clang::tooling::cc1modbuildd::ActionType &Value) {
    Io.enumCase(Value, "REGISTER",
                clang::tooling::cc1modbuildd::ActionType::REGISTER);
    Io.enumCase(Value, "HANDSHAKE",
                clang::tooling::cc1modbuildd::ActionType::HANDSHAKE);
  }
};

template <>
struct llvm::yaml::MappingTraits<clang::tooling::cc1modbuildd::RegisterMsg> {
  static void mapping(IO &Io, clang::tooling::cc1modbuildd::RegisterMsg &Info) {
    Io.mapRequired("Action", Info.MsgAction);
    Io.mapRequired("Status", Info.MsgStatus);
    Io.mapOptional("WorkingDirectory", Info.WorkingDirectory);
    Io.mapOptional("ExecutablePath", Info.ExecutablePath);
    Io.mapOptional("CC1CommandLine", Info.CC1CommandLine);
  }
};

template <>
struct llvm::yaml::MappingTraits<clang::tooling::cc1modbuildd::HandshakeMsg> {
  static void mapping(IO &Io,
                      clang::tooling::cc1modbuildd::HandshakeMsg &Info) {
    Io.mapRequired("Action", Info.MsgAction);
    Io.mapRequired("Status", Info.MsgStatus);
  }
};
#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETSUPPORT_H
