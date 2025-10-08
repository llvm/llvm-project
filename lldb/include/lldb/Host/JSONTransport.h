//===-- JSONTransport.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transport layer for encoding and decoding JSON protocol messages.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_JSONTRANSPORT_H
#define LLDB_HOST_JSONTRANSPORT_H

#include "lldb/Host/MainLoop.h"
#include "lldb/Host/MainLoopBase.h"
#include "lldb/Utility/IOObject.h"
#include "lldb/Utility/Status.h"
#include "lldb/lldb-forward.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>
#if __cplusplus >= 202002L
#include <concepts>
#endif

namespace lldb_private::transport {

/// An error to indicate that the transport reached EOF but there were still
/// unhandled contents in the read buffer.
class TransportUnhandledContentsError
    : public llvm::ErrorInfo<TransportUnhandledContentsError> {
public:
  static char ID;

  explicit TransportUnhandledContentsError(std::string unhandled_contents);

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

  const std::string &getUnhandledContents() const {
    return m_unhandled_contents;
  }

private:
  std::string m_unhandled_contents;
};

/// An error to indicate that the parameters of a Req, Resp or Evt could not be
/// deserialized.
class InvalidParams : public llvm::ErrorInfo<InvalidParams> {
public:
  static char ID;

  explicit InvalidParams(std::string method, std::string context)
      : m_method(std::move(method)), m_context(std::move(context)) {}

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

private:
  /// The JSONRPC remote method call.
  std::string m_method;

  /// Additional context from the parsing failure, e.g. "missing value at
  /// (root)[1].str".
  std::string m_context;
};

/// An error to indicate that no handler was registered for a given method.
class MethodNotFound : public llvm::ErrorInfo<MethodNotFound> {
public:
  static char ID;

  static constexpr int kErrorCode = -32601;

  explicit MethodNotFound(std::string method) : m_method(std::move(method)) {}

  void log(llvm::raw_ostream &OS) const override;
  std::error_code convertToErrorCode() const override;

private:
  std::string m_method;
};

#if __cplusplus >= 202002L
/// A ProtocolDescriptor details the types used in a JSONTransport for handling
/// transport communication.
template <typename T>
concept ProtocolDescriptor = requires {
  typename T::Id;
  typename T::Req;
  typename T::Resp;
  typename T::Evt;
};
#endif

/// A transport is responsible for maintaining the connection to a client
/// application, and reading/writing structured messages to it.
///
/// JSONTransport have limited thread safety requirements:
///  - Messages will not be sent concurrently.
///  - Messages MAY be sent while Run() is reading, or its callback is active.
///
#if __cplusplus >= 202002L
template <ProtocolDescriptor Proto>
#else
template <typename Proto>
#endif
class JSONTransport {
public:
  using Req = typename Proto::Req;
  using Resp = typename Proto::Resp;
  using Evt = typename Proto::Evt;
  using Message = std::variant<Req, Resp, Evt>;

  virtual ~JSONTransport() = default;

  /// Sends an event, a message that does not require a response.
  virtual llvm::Error Send(const Evt &) = 0;
  /// Sends a request, a message that expects a response.
  virtual llvm::Error Send(const Req &) = 0;
  /// Sends a response to a specific request.
  virtual llvm::Error Send(const Resp &) = 0;

  /// Implemented to handle incoming messages. (See `RegisterMessageHandler()`
  /// below).
  class MessageHandler {
  public:
    virtual ~MessageHandler() = default;
    /// Called when an event is received.
    virtual void Received(const Evt &) = 0;
    /// Called when a request is received.
    virtual void Received(const Req &) = 0;
    /// Called when a response is received.
    virtual void Received(const Resp &) = 0;

    /// Called when an error occurs while reading from the transport.
    ///
    /// NOTE: This does *NOT* indicate that a specific request failed, but that
    /// there was an error in the underlying transport.
    virtual void OnError(llvm::Error) = 0;

    /// Called on EOF or client disconnect.
    virtual void OnClosed() = 0;
  };

  /// RegisterMessageHandler registers the Transport with the given MainLoop and
  /// handles any incoming messages using the given MessageHandler.
  ///
  /// If an unexpected error occurs, the MainLoop will be terminated and a log
  /// message will include additional information about the termination reason.
  virtual llvm::Expected<MainLoop::ReadHandleUP>
  RegisterMessageHandler(MainLoop &loop, MessageHandler &handler) = 0;

protected:
  template <typename... Ts> inline auto Logv(const char *Fmt, Ts &&...Vals) {
    Log(llvm::formatv(Fmt, std::forward<Ts>(Vals)...).str());
  }
  virtual void Log(llvm::StringRef message) = 0;
};

/// An IOTransport sends and receives messages using an IOObject.
template <typename Proto> class IOTransport : public JSONTransport<Proto> {
public:
  using Message = typename JSONTransport<Proto>::Message;
  using MessageHandler = typename JSONTransport<Proto>::MessageHandler;

  IOTransport(lldb::IOObjectSP in, lldb::IOObjectSP out)
      : m_in(in), m_out(out) {}

  llvm::Error Send(const typename Proto::Evt &evt) override {
    return Write(evt);
  }
  llvm::Error Send(const typename Proto::Req &req) override {
    return Write(req);
  }
  llvm::Error Send(const typename Proto::Resp &resp) override {
    return Write(resp);
  }

  llvm::Expected<MainLoop::ReadHandleUP>
  RegisterMessageHandler(MainLoop &loop, MessageHandler &handler) override {
    Status status;
    MainLoop::ReadHandleUP read_handle = loop.RegisterReadObject(
        m_in,
        std::bind(&IOTransport::OnRead, this, std::placeholders::_1,
                  std::ref(handler)),
        status);
    if (status.Fail()) {
      return status.takeError();
    }
    return read_handle;
  }

  /// Public for testing purposes, otherwise this should be an implementation
  /// detail.
  static constexpr size_t kReadBufferSize = 1024;

protected:
  llvm::Error Write(const llvm::json::Value &message) {
    this->Logv("<-- {0}", message);
    std::string output = Encode(message);
    size_t bytes_written = output.size();
    return m_out->Write(output.data(), bytes_written).takeError();
  }

  virtual llvm::Expected<std::vector<std::string>> Parse() = 0;
  virtual std::string Encode(const llvm::json::Value &message) = 0;

  llvm::SmallString<kReadBufferSize> m_buffer;

private:
  void OnRead(MainLoopBase &loop, MessageHandler &handler) {
    char buf[kReadBufferSize];
    size_t num_bytes = sizeof(buf);
    if (Status status = m_in->Read(buf, num_bytes); status.Fail()) {
      handler.OnError(status.takeError());
      return;
    }

    if (num_bytes)
      m_buffer.append(llvm::StringRef(buf, num_bytes));

    // If the buffer has contents, try parsing any pending messages.
    if (!m_buffer.empty()) {
      llvm::Expected<std::vector<std::string>> raw_messages = Parse();
      if (llvm::Error error = raw_messages.takeError()) {
        handler.OnError(std::move(error));
        return;
      }

      for (const std::string &raw_message : *raw_messages) {
        llvm::Expected<Message> message =
            llvm::json::parse<Message>(raw_message);
        if (!message) {
          handler.OnError(message.takeError());
          return;
        }

        std::visit([&handler](auto &&msg) { handler.Received(msg); }, *message);
      }
    }

    // Check if we reached EOF.
    if (num_bytes == 0) {
      // EOF reached, but there may still be unhandled contents in the buffer.
      if (!m_buffer.empty())
        handler.OnError(llvm::make_error<TransportUnhandledContentsError>(
            std::string(m_buffer.str())));
      handler.OnClosed();
    }
  }

  lldb::IOObjectSP m_in;
  lldb::IOObjectSP m_out;
};

/// A transport class for JSON with a HTTP header.
#if __cplusplus >= 202002L
template <ProtocolDescriptor Proto>
#else
template <typename Proto>
#endif
class HTTPDelimitedJSONTransport : public IOTransport<Proto> {
public:
  using IOTransport<Proto>::IOTransport;

protected:
  /// Encodes messages based on
  /// https://microsoft.github.io/debug-adapter-protocol/overview#base-protocol
  std::string Encode(const llvm::json::Value &message) override {
    std::string output;
    std::string raw_message = llvm::formatv("{0}", message).str();
    llvm::raw_string_ostream OS(output);
    OS << kHeaderContentLength << kHeaderFieldSeparator << ' '
       << std::to_string(raw_message.size()) << kEndOfHeader << raw_message;
    return output;
  }

  /// Parses messages based on
  /// https://microsoft.github.io/debug-adapter-protocol/overview#base-protocol
  llvm::Expected<std::vector<std::string>> Parse() override {
    std::vector<std::string> messages;
    llvm::StringRef buffer = this->m_buffer;
    while (buffer.contains(kEndOfHeader)) {
      auto [headers, rest] = buffer.split(kEndOfHeader);
      size_t content_length = 0;
      // HTTP Headers are formatted like `<field-name> ':' [<field-value>]`.
      for (const llvm::StringRef &header :
           llvm::split(headers, kHeaderSeparator)) {
        auto [key, value] = header.split(kHeaderFieldSeparator);
        // 'Content-Length' is the only meaningful key at the moment. Others
        // are ignored.
        if (!key.equals_insensitive(kHeaderContentLength))
          continue;

        value = value.trim();
        if (!llvm::to_integer(value, content_length, 10)) {
          // Clear the buffer to avoid re-parsing this malformed message.
          this->m_buffer.clear();
          return llvm::createStringError(std::errc::invalid_argument,
                                         "invalid content length: %s",
                                         value.str().c_str());
        }
      }

      // Check if we have enough data.
      if (content_length > rest.size())
        break;

      llvm::StringRef body = rest.take_front(content_length);
      buffer = rest.drop_front(content_length);
      messages.emplace_back(body.str());
      this->Logv("--> {0}", body);
    }

    // Store the remainder of the buffer for the next read callback.
    this->m_buffer = buffer.str();

    return std::move(messages);
  }

  static constexpr llvm::StringLiteral kHeaderContentLength = "Content-Length";
  static constexpr llvm::StringLiteral kHeaderFieldSeparator = ":";
  static constexpr llvm::StringLiteral kHeaderSeparator = "\r\n";
  static constexpr llvm::StringLiteral kEndOfHeader = "\r\n\r\n";
};

/// A transport class for JSON RPC.
#if __cplusplus >= 202002L
template <ProtocolDescriptor Proto>
#else
template <typename Proto>
#endif
class JSONRPCTransport : public IOTransport<Proto> {
public:
  using IOTransport<Proto>::IOTransport;

protected:
  std::string Encode(const llvm::json::Value &message) override {
    return llvm::formatv("{0}{1}", message, kMessageSeparator).str();
  }

  llvm::Expected<std::vector<std::string>> Parse() override {
    std::vector<std::string> messages;
    llvm::StringRef buf = this->m_buffer;
    while (buf.contains(kMessageSeparator)) {
      auto [raw_json, rest] = buf.split(kMessageSeparator);
      buf = rest;
      messages.emplace_back(raw_json.str());
      this->Logv("--> {0}", raw_json);
    }

    // Store the remainder of the buffer for the next read callback.
    this->m_buffer = buf.str();

    return messages;
  }

  static constexpr llvm::StringLiteral kMessageSeparator = "\n";
};

/// A handler for the response to an outgoing request.
template <typename T>
using Reply =
    std::conditional_t<std::is_void_v<T>,
                       llvm::unique_function<void(llvm::Error)>,
                       llvm::unique_function<void(llvm::Expected<T>)>>;

namespace detail {
template <typename R, typename P> struct request_t final {
  using type = llvm::unique_function<void(const P &, Reply<R>)>;
};
template <typename R> struct request_t<R, void> final {
  using type = llvm::unique_function<void(Reply<R>)>;
};
template <typename P> struct event_t final {
  using type = llvm::unique_function<void(const P &)>;
};
template <> struct event_t<void> final {
  using type = llvm::unique_function<void()>;
};
} // namespace detail

template <typename R, typename P>
using OutgoingRequest = typename detail::request_t<R, P>::type;

/// A function to send an outgoing event.
template <typename P> using OutgoingEvent = typename detail::event_t<P>::type;

#if __cplusplus >= 202002L
/// This represents a protocol description that includes additional helpers
/// for constructing requests, responses and events to work with `Binder`.
template <typename T>
concept BindingBuilder =
    ProtocolDescriptor<T> &&
    requires(T::Id id, T::Req req, T::Resp resp, T::Evt evt,
             llvm::StringRef method, std::optional<llvm::json::Value> params,
             std::optional<llvm::json::Value> result, llvm::Error err) {
      /// For initializing the unique sequence identifier;
      { T::InitialId() } -> std::same_as<typename T::Id>;
      /// Incrementing the sequence identifier.
      { id++ } -> std::same_as<typename T::Id>;

      /// Constructing protocol types
      /// @{
      /// Construct a new request.
      { T::Make(id, method, params) } -> std::same_as<typename T::Req>;
      /// Construct a new error response.
      { T::Make(req, std::move(err)) } -> std::same_as<typename T::Resp>;
      /// Construct a new success response.
      { T::Make(req, result) } -> std::same_as<typename T::Resp>;
      /// Construct a new event.
      { T::Make(method, params) } -> std::same_as<typename T::Evt>;
      /// @}

      /// Keys for associated types.
      /// @{
      /// Looking up in flight responses.
      { T::KeyFor(resp) } -> std::same_as<typename T::Id>;
      /// Extract method from request.
      { T::KeyFor(req) } -> std::same_as<std::string>;
      /// Extract method from event.
      { T::KeyFor(evt) } -> std::same_as<std::string>;
      /// @}

      /// Extracting information from associated types.
      /// @{
      /// Extract parameters from a request.
      { T::Extract(req) } -> std::same_as<std::optional<llvm::json::Value>>;
      /// Extract result from a response.
      { T::Extract(resp) } -> std::same_as<llvm::Expected<llvm::json::Value>>;
      /// Extract parameters from an event.
      { T::Extract(evt) } -> std::same_as<std::optional<llvm::json::Value>>;
      /// @}
    };
#endif

/// Binder collects a table of functions that handle calls.
///
/// The wrapper takes care of parsing/serializing responses.
///
/// This allows a JSONTransport to handle incoming and outgoing requests and
/// events.
///
/// A bind of an incoming request to a lambda.
/// \code{cpp}
/// Binder binder{transport};
/// binder.bind<int, vector<int>>("adder", [](const vector<int> &params) {
///   int sum = 0;
///   for (int v : params)
///     sum += v;
///   return sum;
/// });
/// \endcode
///
/// A bind of an outgoing request.
/// \code{cpp}
/// OutgoingRequest<int, vector<int>> call_add =
///     binder.bind<int, vector<int>>("add");
/// call_add({1,2,3}, [](Expected<int> result) {
///   cout << *result << "\n";
/// });
/// \endcode
#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
class Binder : public JSONTransport<Proto>::MessageHandler {
  using Req = typename Proto::Req;
  using Resp = typename Proto::Resp;
  using Evt = typename Proto::Evt;
  using Id = typename Proto::Id;
  using Transport = JSONTransport<Proto>;
  using MessageHandler = typename Transport::MessageHandler;

public:
  explicit Binder(Transport &transport) : m_transport(transport), m_seq(0) {}

  Binder(const Binder &) = delete;
  Binder &operator=(const Binder &) = delete;

  /// Bind a handler on transport disconnect.
  template <typename Fn, typename... Args>
  void OnDisconnect(Fn &&fn, Args &&...args);

  /// Bind a handler on error when communicating with the transport.
  template <typename Fn, typename... Args>
  void OnError(Fn &&fn, Args &&...args);

  /// Bind a handler for an incoming request.
  /// e.g. `bind("peek", &ThisModule::peek, this);`.
  /// Handler should be e.g. `Expected<PeekResult> peek(const PeekParams&);`
  /// PeekParams must be JSON parsable and PeekResult must be serializable.
  template <typename Result, typename Params, typename Fn, typename... Args>
  void Bind(llvm::StringLiteral method, Fn &&fn, Args &&...args);

  /// Bind a handler for an incoming event.
  /// e.g. `bind("peek", &ThisModule::peek, this);`
  /// Handler should be e.g. `void peek(const PeekParams&);`
  /// PeekParams must be JSON parsable.
  template <typename Params, typename Fn, typename... Args>
  void Bind(llvm::StringLiteral method, Fn &&fn, Args &&...args);

  /// Bind a function object to be used for outgoing requests.
  /// e.g. `OutgoingRequest<Params, Result> Edit = bind("edit");`
  /// Params must be JSON-serializable, Result must be parsable.
  template <typename Result, typename Params>
  OutgoingRequest<Result, Params> Bind(llvm::StringLiteral method);

  /// Bind a function object to be used for outgoing events.
  /// e.g. `OutgoingEvent<LogParams> Log = bind("log");`
  /// LogParams must be JSON-serializable.
  template <typename Params>
  OutgoingEvent<Params> Bind(llvm::StringLiteral method);

  void Received(const Evt &evt) override {
    std::scoped_lock<std::recursive_mutex> guard(m_mutex);
    auto it = m_event_handlers.find(Proto::KeyFor(evt));
    if (it == m_event_handlers.end()) {
      OnError(llvm::createStringError(
          llvm::formatv("no handled for event {0}", toJSON(evt))));
      return;
    }
    it->second(evt);
  }

  void Received(const Req &req) override {
    ReplyOnce reply(req, &m_transport, this);

    std::scoped_lock<std::recursive_mutex> guard(m_mutex);
    auto it = m_request_handlers.find(Proto::KeyFor(req));
    if (it == m_request_handlers.end()) {
      reply(Proto::Make(req, llvm::createStringError("method not found")));
      return;
    }

    it->second(req, std::move(reply));
  }

  void Received(const Resp &resp) override {
    std::scoped_lock<std::recursive_mutex> guard(m_mutex);

    Id id = Proto::KeyFor(resp);
    auto it = m_pending_responses.find(id);
    if (it == m_pending_responses.end()) {
      OnError(llvm::createStringError(
          llvm::formatv("no pending request for {0}", toJSON(resp))));
      return;
    }

    it->second(resp);
    m_pending_responses.erase(it);
  }

  void OnError(llvm::Error err) override {
    std::scoped_lock<std::recursive_mutex> guard(m_mutex);
    if (m_error_handler)
      m_error_handler(std::move(err));
  }

  void OnClosed() override {
    std::scoped_lock<std::recursive_mutex> guard(m_mutex);
    if (m_disconnect_handler)
      m_disconnect_handler();
  }

private:
  template <typename T>
  llvm::Expected<T> static Parse(const llvm::json::Value &raw,
                                 llvm::StringRef method);

  template <typename T> using Callback = llvm::unique_function<T>;

  std::recursive_mutex m_mutex;
  Transport &m_transport;
  Id m_seq;
  std::map<Id, Callback<void(const Resp &)>> m_pending_responses;
  llvm::StringMap<Callback<void(const Req &, Callback<void(const Resp &)>)>>
      m_request_handlers;
  llvm::StringMap<Callback<void(const Evt &)>> m_event_handlers;
  Callback<void()> m_disconnect_handler;
  Callback<void(llvm::Error)> m_error_handler;

  /// Function object to reply to a call.
  /// Each instance must be called exactly once, otherwise:
  ///  - the bug is logged, and (in debug mode) an assert will fire
  ///  - if there was no reply, an error reply is sent
  ///  - if there were multiple replies, only the first is sent
  class ReplyOnce {
    std::atomic<bool> replied = {false};
    const Req req;
    Transport *transport;    // Null when moved-from.
    MessageHandler *handler; // Null when moved-from.

  public:
    ReplyOnce(const Req req, Transport *transport, MessageHandler *handler)
        : req(req), transport(transport), handler(handler) {
      assert(handler);
    }
    ReplyOnce(ReplyOnce &&other)
        : replied(other.replied.load()), req(other.req),
          transport(other.transport), handler(other.handler) {
      other.transport = nullptr;
      other.handler = nullptr;
    }
    ReplyOnce &operator=(ReplyOnce &&) = delete;
    ReplyOnce(const ReplyOnce &) = delete;
    ReplyOnce &operator=(const ReplyOnce &) = delete;

    ~ReplyOnce() {
      if (transport && handler && !replied) {
        assert(false && "must reply to all calls!");
        (*this)(Proto::Make(req, llvm::createStringError("failed to reply")));
      }
    }

    void operator()(const Resp &resp) {
      assert(transport && handler && "moved-from!");
      if (replied.exchange(true)) {
        assert(false && "must reply to each call only once!");
        return;
      }

      if (llvm::Error error = transport->Send(resp))
        handler->OnError(std::move(error));
    }
  };
};

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename Fn, typename... Args>
void Binder<Proto>::OnDisconnect(Fn &&fn, Args &&...args) {
  m_disconnect_handler = [fn, args...]() mutable {
    std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
  };
}

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename Fn, typename... Args>
void Binder<Proto>::OnError(Fn &&fn, Args &&...args) {
  m_error_handler = [fn, args...](llvm::Error error) mutable {
    std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...,
                std::move(error));
  };
}

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename Result, typename Params, typename Fn, typename... Args>
void Binder<Proto>::Bind(llvm::StringLiteral method, Fn &&fn, Args &&...args) {
  assert(m_request_handlers.find(method) == m_request_handlers.end() &&
         "request already bound");
  if constexpr (std::is_void_v<Result> && std::is_void_v<Params>) {
    m_request_handlers[method] =
        [fn, args...](const Req &req,
                      llvm::unique_function<void(const Resp &)> reply) mutable {
          llvm::Error result =
              std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
          reply(Proto::Make(req, std::move(result)));
        };
  } else if constexpr (std::is_void_v<Params>) {
    m_request_handlers[method] =
        [fn, args...](const Req &req,
                      llvm::unique_function<void(const Resp &)> reply) mutable {
          llvm::Expected<Result> result =
              std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
          if (!result)
            return reply(Proto::Make(req, result.takeError()));
          reply(Proto::Make(req, toJSON(*result)));
        };
  } else if constexpr (std::is_void_v<Result>) {
    m_request_handlers[method] =
        [method, fn,
         args...](const Req &req,
                  llvm::unique_function<void(const Resp &)> reply) mutable {
          llvm::Expected<Params> params =
              Parse<Params>(Proto::Extract(req), method);
          if (!params)
            return reply(Proto::Make(req, params.takeError()));

          llvm::Error result = std::invoke(
              std::forward<Fn>(fn), std::forward<Args>(args)..., *params);
          reply(Proto::Make(req, std::move(result)));
        };
  } else {
    m_request_handlers[method] =
        [method, fn,
         args...](const Req &req,
                  llvm::unique_function<void(const Resp &)> reply) mutable {
          llvm::Expected<Params> params =
              Parse<Params>(Proto::Extract(req), method);
          if (!params)
            return reply(Proto::Make(req, params.takeError()));

          llvm::Expected<Result> result = std::invoke(
              std::forward<Fn>(fn), std::forward<Args>(args)..., *params);
          if (!result)
            return reply(Proto::Make(req, result.takeError()));

          reply(Proto::Make(req, toJSON(*result)));
        };
  }
}

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename Params, typename Fn, typename... Args>
void Binder<Proto>::Bind(llvm::StringLiteral method, Fn &&fn, Args &&...args) {
  assert(m_event_handlers.find(method) == m_event_handlers.end() &&
         "event already bound");
  if constexpr (std::is_void_v<Params>) {
    m_event_handlers[method] = [fn, args...](const Evt &) mutable {
      std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)...);
    };
  } else {
    m_event_handlers[method] = [this, method, fn,
                                args...](const Evt &evt) mutable {
      llvm::Expected<Params> params =
          Parse<Params>(Proto::Extract(evt), method);
      if (!params)
        return OnError(params.takeError());
      std::invoke(std::forward<Fn>(fn), std::forward<Args>(args)..., *params);
    };
  }
}

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename Result, typename Params>
OutgoingRequest<Result, Params>
Binder<Proto>::Bind(llvm::StringLiteral method) {
  if constexpr (std::is_void_v<Result> && std::is_void_v<Params>) {
    return [this, method](Reply<Result> fn) {
      std::scoped_lock<std::recursive_mutex> guard(m_mutex);
      Id id = ++m_seq;
      Req req = Proto::Make(id, method, std::nullopt);
      m_pending_responses[id] = [fn = std::move(fn)](const Resp &resp) mutable {
        llvm::Expected<llvm::json::Value> result = Proto::Extract(resp);
        if (!result)
          return fn(result.takeError());
        fn(llvm::Error::success());
      };
      if (llvm::Error error = m_transport.Send(req))
        OnError(std::move(error));
    };
  } else if constexpr (std::is_void_v<Params>) {
    return [this, method](Reply<Result> fn) {
      std::scoped_lock<std::recursive_mutex> guard(m_mutex);
      Id id = ++m_seq;
      Req req = Proto::Make(id, method, std::nullopt);
      m_pending_responses[id] = [fn = std::move(fn),
                                 method](const Resp &resp) mutable {
        llvm::Expected<llvm::json::Value> result = Proto::Extract(resp);
        if (!result)
          return fn(result.takeError());
        fn(Parse<Result>(*result, method));
      };
      if (llvm::Error error = m_transport.Send(req))
        OnError(std::move(error));
    };
  } else if constexpr (std::is_void_v<Result>) {
    return [this, method](const Params &params, Reply<Result> fn) {
      std::scoped_lock<std::recursive_mutex> guard(m_mutex);
      Id id = ++m_seq;
      Req req = Proto::Make(id, method, llvm::json::Value(params));
      m_pending_responses[id] = [fn = std::move(fn)](const Resp &resp) mutable {
        llvm::Expected<llvm::json::Value> result = Proto::Extract(resp);
        if (!result)
          return fn(result.takeError());
        fn(llvm::Error::success());
      };
      if (llvm::Error error = m_transport.Send(req))
        OnError(std::move(error));
    };
  } else {
    return [this, method](const Params &params, Reply<Result> fn) {
      std::scoped_lock<std::recursive_mutex> guard(m_mutex);
      Id id = ++m_seq;
      Req req = Proto::Make(id, method, llvm::json::Value(params));
      m_pending_responses[id] = [fn = std::move(fn),
                                 method](const Resp &resp) mutable {
        llvm::Expected<llvm::json::Value> result = Proto::Extract(resp);
        if (llvm::Error err = result.takeError())
          return fn(std::move(err));
        fn(Parse<Result>(*result, method));
      };
      if (llvm::Error error = m_transport.Send(req))
        OnError(std::move(error));
    };
  }
}

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename Params>
OutgoingEvent<Params> Binder<Proto>::Bind(llvm::StringLiteral method) {
  if constexpr (std::is_void_v<Params>) {
    return [this, method]() {
      if (llvm::Error error =
              m_transport.Send(Proto::Make(method, std::nullopt)))
        OnError(std::move(error));
    };
  } else {
    return [this, method](const Params &params) {
      if (llvm::Error error =
              m_transport.Send(Proto::Make(method, toJSON(params))))
        OnError(std::move(error));
    };
  }
}

#if __cplusplus >= 202002L
template <BindingBuilder Proto>
#else
template <typename Proto>
#endif
template <typename T>
llvm::Expected<T> Binder<Proto>::Parse(const llvm::json::Value &raw,
                                       llvm::StringRef method) {
  T result;
  llvm::json::Path::Root root;
  if (!fromJSON(raw, result, root)) {
    // Dump the relevant parts of the broken message.
    std::string context;
    llvm::raw_string_ostream OS(context);
    root.printErrorContext(raw, OS);
    return llvm::make_error<InvalidParams>(method.str(), context);
  }
  return std::move(result);
}

} // namespace lldb_private::transport

#endif
