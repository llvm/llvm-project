//===--- JSONTransport.cpp - sending and receiving LSP messages over JSON -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Logger.h"
#include "Protocol.h" // For LSPError
#include "Transport.h"
#include "llvm/Support/Errno.h"

using namespace llvm;
namespace clang {
namespace clangd {
namespace {

json::Object encodeError(Error E) {
  std::string Message;
  ErrorCode Code = ErrorCode::UnknownErrorCode;
  if (Error Unhandled =
          handleErrors(std::move(E), [&](const LSPError &L) -> Error {
            Message = L.Message;
            Code = L.Code;
            return Error::success();
          }))
    Message = toString(std::move(Unhandled));

  return json::Object{
      {"message", std::move(Message)},
      {"code", int64_t(Code)},
  };
}

Error decodeError(const json::Object &O) {
  std::string Msg = O.getString("message").getValueOr("Unspecified error");
  if (auto Code = O.getInteger("code"))
    return make_error<LSPError>(std::move(Msg), ErrorCode(*Code));
  return make_error<StringError>(std::move(Msg), inconvertibleErrorCode());
}

class JSONTransport : public Transport {
public:
  JSONTransport(std::FILE *In, raw_ostream &Out, raw_ostream *InMirror,
                bool Pretty, JSONStreamStyle Style)
      : In(In), Out(Out), InMirror(InMirror ? *InMirror : nulls()),
        Pretty(Pretty), Style(Style) {}

  void notify(StringRef Method, json::Value Params) override {
    sendMessage(json::Object{
        {"jsonrpc", "2.0"},
        {"method", Method},
        {"params", std::move(Params)},
    });
  }
  void call(StringRef Method, json::Value Params, json::Value ID) override {
    sendMessage(json::Object{
        {"jsonrpc", "2.0"},
        {"id", std::move(ID)},
        {"method", Method},
        {"params", std::move(Params)},
    });
  }
  void reply(json::Value ID, Expected<json::Value> Result) override {
    if (Result) {
      sendMessage(json::Object{
          {"jsonrpc", "2.0"},
          {"id", std::move(ID)},
          {"result", std::move(*Result)},
      });
    } else {
      sendMessage(json::Object{
          {"jsonrpc", "2.0"},
          {"id", std::move(ID)},
          {"error", encodeError(Result.takeError())},
      });
    }
  }

  Error loop(MessageHandler &Handler) override {
    while (!feof(In)) {
      if (ferror(In))
        return errorCodeToError(std::error_code(errno, std::system_category()));
      if (auto JSON = readRawMessage()) {
        if (auto Doc = json::parse(*JSON)) {
          vlog(Pretty ? "<<< {0:2}\n" : "<<< {0}\n", *Doc);
          if (!handleMessage(std::move(*Doc), Handler))
            return Error::success(); // we saw the "exit" notification.
        } else {
          // Parse error. Log the raw message.
          vlog("<<< {0}\n", *JSON);
          elog("JSON parse error: {0}", toString(Doc.takeError()));
        }
      }
    }
    return errorCodeToError(std::make_error_code(std::errc::io_error));
  }

private:
  // Dispatches incoming message to Handler onNotify/onCall/onReply.
  bool handleMessage(json::Value Message, MessageHandler &Handler);
  // Writes outgoing message to Out stream.
  void sendMessage(json::Value Message) {
    std::string S;
    raw_string_ostream OS(S);
    OS << formatv(Pretty ? "{0:2}" : "{0}", Message);
    OS.flush();
    Out << "Content-Length: " << S.size() << "\r\n\r\n" << S;
    Out.flush();
    vlog(">>> {0}\n", S);
  }

  // Read raw string messages from input stream.
  Optional<std::string> readRawMessage() {
    return Style == JSONStreamStyle::Delimited ? readDelimitedMessage()
                                               : readStandardMessage();
  }
  Optional<std::string> readDelimitedMessage();
  Optional<std::string> readStandardMessage();

  std::FILE *In;
  raw_ostream &Out;
  raw_ostream &InMirror;
  bool Pretty;
  JSONStreamStyle Style;
};

bool JSONTransport::handleMessage(json::Value Message,
                                  MessageHandler &Handler) {
  // Message must be an object with "jsonrpc":"2.0".
  auto *Object = Message.getAsObject();
  if (!Object || Object->getString("jsonrpc") != Optional<StringRef>("2.0")) {
    elog("Not a JSON-RPC 2.0 message: {0:2}", Message);
    return false;
  }
  // ID may be any JSON value. If absent, this is a notification.
  Optional<json::Value> ID;
  if (auto *I = Object->get("id"))
    ID = std::move(*I);
  auto Method = Object->getString("method");
  if (!Method) { // This is a response.
    if (!ID) {
      elog("No method and no response ID: {0:2}", Message);
      return false;
    }
    if (auto *Err = Object->getObject("error"))
      return Handler.onReply(std::move(*ID), decodeError(*Err));
    // Result should be given, use null if not.
    json::Value Result = nullptr;
    if (auto *R = Object->get("result"))
      Result = std::move(*R);
    return Handler.onReply(std::move(*ID), std::move(Result));
  }
  // Params should be given, use null if not.
  json::Value Params = nullptr;
  if (auto *P = Object->get("params"))
    Params = std::move(*P);

  if (ID)
    return Handler.onCall(*Method, std::move(Params), std::move(*ID));
  else
    return Handler.onNotify(*Method, std::move(Params));
}

// Tries to read a line up to and including \n.
// If failing, feof() or ferror() will be set.
bool readLine(std::FILE *In, std::string &Out) {
  static constexpr int BufSize = 1024;
  size_t Size = 0;
  Out.clear();
  for (;;) {
    Out.resize(Size + BufSize);
    // Handle EINTR which is sent when a debugger attaches on some platforms.
    if (!sys::RetryAfterSignal(nullptr, ::fgets, &Out[Size], BufSize, In))
      return false;
    clearerr(In);
    // If the line contained null bytes, anything after it (including \n) will
    // be ignored. Fortunately this is not a legal header or JSON.
    size_t Read = std::strlen(&Out[Size]);
    if (Read > 0 && Out[Size + Read - 1] == '\n') {
      Out.resize(Size + Read);
      return true;
    }
    Size += Read;
  }
}

// Returns None when:
//  - ferror() or feof() are set.
//  - Content-Length is missing or empty (protocol error)
Optional<std::string> JSONTransport::readStandardMessage() {
  // A Language Server Protocol message starts with a set of HTTP headers,
  // delimited  by \r\n, and terminated by an empty line (\r\n).
  unsigned long long ContentLength = 0;
  std::string Line;
  while (true) {
    if (feof(In) || ferror(In) || !readLine(In, Line))
      return None;
    InMirror << Line;

    StringRef LineRef(Line);

    // We allow comments in headers. Technically this isn't part

    // of the LSP specification, but makes writing tests easier.
    if (LineRef.startswith("#"))
      continue;

    // Content-Length is a mandatory header, and the only one we handle.
    if (LineRef.consume_front("Content-Length: ")) {
      if (ContentLength != 0) {
        elog("Warning: Duplicate Content-Length header received. "
             "The previous value for this message ({0}) was ignored.",
             ContentLength);
      }
      getAsUnsignedInteger(LineRef.trim(), 0, ContentLength);
      continue;
    } else if (!LineRef.trim().empty()) {
      // It's another header, ignore it.
      continue;
    } else {
      // An empty line indicates the end of headers.
      // Go ahead and read the JSON.
      break;
    }
  }

  // The fuzzer likes crashing us by sending "Content-Length: 9999999999999999"
  if (ContentLength > 1 << 30) { // 1024M
    elog("Refusing to read message with long Content-Length: {0}. "
         "Expect protocol errors",
         ContentLength);
    return None;
  }
  if (ContentLength == 0) {
    log("Warning: Missing Content-Length header, or zero-length message.");
    return None;
  }

  std::string JSON(ContentLength, '\0');
  for (size_t Pos = 0, Read; Pos < ContentLength; Pos += Read) {
    // Handle EINTR which is sent when a debugger attaches on some platforms.
    Read = sys::RetryAfterSignal(0u, ::fread, &JSON[Pos], 1,
                                 ContentLength - Pos, In);
    if (Read == 0) {
      elog("Input was aborted. Read only {0} bytes of expected {1}.", Pos,
           ContentLength);
      return None;
    }
    InMirror << StringRef(&JSON[Pos], Read);
    clearerr(In); // If we're done, the error was transient. If we're not done,
                  // either it was transient or we'll see it again on retry.
    Pos += Read;
  }
  return std::move(JSON);
}

// For lit tests we support a simplified syntax:
// - messages are delimited by '---' on a line by itself
// - lines starting with # are ignored.
// This is a testing path, so favor simplicity over performance here.
// When returning None, feof() or ferror() will be set.
Optional<std::string> JSONTransport::readDelimitedMessage() {
  std::string JSON;
  std::string Line;
  while (readLine(In, Line)) {
    InMirror << Line;
    auto LineRef = StringRef(Line).trim();
    if (LineRef.startswith("#")) // comment
      continue;

    // found a delimiter
    if (LineRef.rtrim() == "---")
      break;

    JSON += Line;
  }

  if (ferror(In)) {
    elog("Input error while reading message!");
    return None;
  }
  return std::move(JSON); // Including at EOF
}

} // namespace

std::unique_ptr<Transport> newJSONTransport(std::FILE *In, raw_ostream &Out,
                                            raw_ostream *InMirror, bool Pretty,
                                            JSONStreamStyle Style) {
  return llvm::make_unique<JSONTransport>(In, Out, InMirror, Pretty, Style);
}

} // namespace clangd
} // namespace clang
