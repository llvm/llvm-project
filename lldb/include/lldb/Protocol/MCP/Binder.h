//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PROTOCOL_MCP_BINDER_H
#define LLDB_PROTOCOL_MCP_BINDER_H

#include "lldb/Protocol/MCP/MCPError.h"
#include "lldb/Protocol/MCP/Protocol.h"
#include "lldb/Protocol/MCP/Transport.h"
#include "lldb/Utility/Status.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>

namespace lldb_protocol::mcp {

template <typename T> using Callback = llvm::unique_function<T>;

template <typename T>
using Reply = llvm::unique_function<void(llvm::Expected<T>)>;
template <typename Params, typename Result>
using OutgoingRequest =
    llvm::unique_function<void(const Params &, Reply<Result>)>;
template <typename Params>
using OutgoingNotification = llvm::unique_function<void(const Params &)>;

template <typename Params, typename Result>
llvm::Expected<Result> AsyncInvoke(lldb_private::MainLoop &loop,
                                   OutgoingRequest<Params, Result> &fn,
                                   const Params &params) {
  std::promise<llvm::Expected<Result>> result_promise;
  std::future<llvm::Expected<Result>> result_future =
      result_promise.get_future();
  std::thread thr([&loop, &fn, params,
                   result_promise = std::move(result_promise)]() mutable {
    fn(params, [&loop, &result_promise](llvm::Expected<Result> result) mutable {
      result_promise.set_value(std::move(result));
      loop.AddPendingCallback(
          [](lldb_private::MainLoopBase &loop) { loop.RequestTermination(); });
    });
    if (llvm::Error error = loop.Run().takeError())
      result_promise.set_value(std::move(error));
  });
  thr.join();
  return result_future.get();
}

/// Binder collects a table of functions that handle calls.
///
/// The wrapper takes care of parsing/serializing responses.
class Binder {
public:
  explicit Binder(MCPTransport *transport) : m_handlers(transport) {}

  Binder(const Binder &) = delete;
  Binder &operator=(const Binder &) = delete;

  /// Bind a handler on transport disconnect.
  template <typename ThisT, typename... ExtraArgs>
  void disconnected(void (ThisT::*handler)(MCPTransport *), ThisT *_this,
                    ExtraArgs... extra_args) {
    m_handlers.m_disconnect_handler =
        std::bind(handler, _this, std::placeholders::_1,
                  std::forward<ExtraArgs>(extra_args)...);
  }

  /// Bind a handler on error when communicating with the transport.
  template <typename ThisT, typename... ExtraArgs>
  void error(void (ThisT::*handler)(MCPTransport *, llvm::Error), ThisT *_this,
             ExtraArgs... extra_args) {
    m_handlers.m_error_handler =
        std::bind(handler, _this, std::placeholders::_1, std::placeholders::_2,
                  std::forward<ExtraArgs>(extra_args)...);
  }

  /// Bind a handler for a request.
  /// e.g. Bind.request("peek", this, &ThisModule::peek);
  /// Handler should be e.g. Expected<PeekResult> peek(const PeekParams&);
  /// PeekParams must be JSON parsable and PeekResult must be serializable.
  template <typename Result, typename Params, typename ThisT,
            typename... ExtraArgs>
  void request(llvm::StringLiteral method,
               llvm::Expected<Result> (ThisT::*fn)(const Params &,
                                                   ExtraArgs...),
               ThisT *_this, ExtraArgs... extra_args) {
    assert(m_handlers.m_request_handlers.find(method) ==
               m_handlers.m_request_handlers.end() &&
           "request already bound");
    std::function<llvm::Expected<Result>(const Params &)> handler =
        std::bind(fn, _this, std::placeholders::_1,
                  std::forward<ExtraArgs>(extra_args)...);
    m_handlers.m_request_handlers[method] =
        [method, handler](const Request &req,
                          llvm::unique_function<void(const Response &)> reply) {
          Params params;
          llvm::json::Path::Root root(method);
          if (!fromJSON(req.params, params, root)) {
            reply(Response{0, Error{eErrorCodeInvalidParams,
                                    "invalid params for " + method.str() +
                                        ": " + llvm::toString(root.getError()),
                                    std::nullopt}});
            return;
          }
          llvm::Expected<Result> result = handler(params);
          if (llvm::Error error = result.takeError()) {
            Error protocol_error;
            llvm::handleAllErrors(
                std::move(error),
                [&](const MCPError &err) {
                  protocol_error = err.toProtocolError();
                },
                [&](const llvm::ErrorInfoBase &err) {
                  protocol_error.code = MCPError::kInternalError;
                  protocol_error.message = err.message();
                });
            reply(Response{0, protocol_error});
            return;
          }

          reply(Response{0, *result});
        };
  }

  /// Bind a handler for an async request.
  /// e.g. Bind.asyncRequest("peek", this, &ThisModule::peek);
  /// Handler should be e.g. `void peek(const PeekParams&,
  /// Reply<Expected<PeekResult>>);` PeekParams must be JSON parsable and
  /// PeekResult must be serializable.
  template <typename Result, typename Params, typename... ExtraArgs>
  void asyncRequest(
      llvm::StringLiteral method,
      std::function<void(const Params &, ExtraArgs..., Reply<Result>)> fn,
      ExtraArgs... extra_args) {
    assert(m_handlers.m_request_handlers.find(method) ==
               m_handlers.m_request_handlers.end() &&
           "request already bound");
    std::function<void(const Params &, Reply<Result>)> handler = std::bind(
        fn, std::placeholders::_1, std::forward<ExtraArgs>(extra_args)...,
        std::placeholders::_2);
    m_handlers.m_request_handlers[method] =
        [method, handler](const Request &req,
                          Callback<void(const Response &)> reply) {
          Params params;
          llvm::json::Path::Root root(method);
          if (!fromJSON(req.params, params, root)) {
            reply(Response{0, Error{eErrorCodeInvalidParams,
                                    "invalid params for " + method.str() +
                                        ": " + llvm::toString(root.getError()),
                                    std::nullopt}});
            return;
          }

          handler(params, [reply = std::move(reply)](
                              llvm::Expected<Result> result) mutable {
            if (llvm::Error error = result.takeError()) {
              Error protocol_error;
              llvm::handleAllErrors(
                  std::move(error),
                  [&](const MCPError &err) {
                    protocol_error = err.toProtocolError();
                  },
                  [&](const llvm::ErrorInfoBase &err) {
                    protocol_error.code = MCPError::kInternalError;
                    protocol_error.message = err.message();
                  });
              reply(Response{0, protocol_error});
              return;
            }

            reply(Response{0, toJSON(*result)});
          });
        };
  }
  template <typename Result, typename Params, typename ThisT,
            typename... ExtraArgs>
  void asyncRequest(llvm::StringLiteral method,
                    void (ThisT::*fn)(const Params &, ExtraArgs...,
                                      Reply<Result>),
                    ThisT *_this, ExtraArgs... extra_args) {
    assert(m_handlers.m_request_handlers.find(method) ==
               m_handlers.m_request_handlers.end() &&
           "request already bound");
    std::function<void(const Params &, Reply<Result>)> handler = std::bind(
        fn, _this, std::placeholders::_1,
        std::forward<ExtraArgs>(extra_args)..., std::placeholders::_2);
    m_handlers.m_request_handlers[method] =
        [method, handler](const Request &req,
                          Callback<void(const Response &)> reply) {
          Params params;
          llvm::json::Path::Root root;
          if (!fromJSON(req.params, params, root)) {
            reply(Response{0, Error{eErrorCodeInvalidParams,
                                    "invalid params for " + method.str(),
                                    std::nullopt}});
            return;
          }

          handler(params, [reply = std::move(reply)](
                              llvm::Expected<Result> result) mutable {
            if (llvm::Error error = result.takeError()) {
              Error protocol_error;
              llvm::handleAllErrors(
                  std::move(error),
                  [&](const MCPError &err) {
                    protocol_error = err.toProtocolError();
                  },
                  [&](const llvm::ErrorInfoBase &err) {
                    protocol_error.code = MCPError::kInternalError;
                    protocol_error.message = err.message();
                  });
              reply(Response{0, protocol_error});
              return;
            }

            reply(Response{0, toJSON(*result)});
          });
        };
  }

  /// Bind a handler for a notification.
  /// e.g. Bind.notification("peek", this, &ThisModule::peek);
  /// Handler should be e.g. void peek(const PeekParams&);
  /// PeekParams must be JSON parsable.
  template <typename Params, typename ThisT, typename... ExtraArgs>
  void notification(llvm::StringLiteral method,
                    void (ThisT::*fn)(const Params &, ExtraArgs...),
                    ThisT *_this, ExtraArgs... extra_args) {
    std::function<void(const Params &)> handler =
        std::bind(fn, _this, std::placeholders::_1,
                  std::forward<ExtraArgs>(extra_args)...);
    m_handlers.m_notification_handlers[method] =
        [handler](const Notification &note) {
          Params params;
          llvm::json::Path::Root root;
          if (!fromJSON(note.params, params, root))
            return; // FIXME: log error?

          handler(params);
        };
  }
  template <typename Params>
  void notification(llvm::StringLiteral method,
                    std::function<void(const Params &)> handler) {
    assert(m_handlers.m_notification_handlers.find(method) ==
               m_handlers.m_notification_handlers.end() &&
           "notification already bound");
    m_handlers.m_notification_handlers[method] =
        [handler = std::move(handler)](const Notification &note) {
          Params params;
          llvm::json::Path::Root root;
          if (!fromJSON(note.params, params, root))
            return; // FIXME: log error?

          handler(params);
        };
  }

  /// Bind a function object to be used for outgoing requests.
  /// e.g. OutgoingRequest<Params, Result> Edit = Bind.outgoingRequest("edit");
  /// Params must be JSON-serializable, Result must be parsable.
  template <typename Params, typename Result>
  OutgoingRequest<Params, Result> outgoingRequest(llvm::StringLiteral method) {
    return [this, method](const Params &params, Reply<Result> reply) {
      Request request;
      request.method = method;
      request.params = toJSON(params);
      m_handlers.Send(request, [reply = std::move(reply)](
                                   const Response &resp) mutable {
        if (const lldb_protocol::mcp::Error *err =
                std::get_if<lldb_protocol::mcp::Error>(&resp.result)) {
          reply(llvm::make_error<MCPError>(err->message, err->code));
          return;
        }
        Result result;
        llvm::json::Path::Root root;
        if (!fromJSON(std::get<llvm::json::Value>(resp.result), result, root)) {
          reply(llvm::make_error<MCPError>("parsing response failed: " +
                                           llvm::toString(root.getError())));
          return;
        }
        reply(result);
      });
    };
  }

  /// Bind a function object to be used for outgoing notifications.
  /// e.g. OutgoingNotification<LogParams> Log = Bind.outgoingMethod("log");
  /// LogParams must be JSON-serializable.
  template <typename Params>
  OutgoingNotification<Params>
  outgoingNotification(llvm::StringLiteral method) {
    return [this, method](const Params &params) {
      Notification note;
      note.method = method;
      note.params = toJSON(params);
      m_handlers.Send(note);
    };
  }

  operator MCPTransport::MessageHandler &() { return m_handlers; }

private:
  class RawHandler final : public MCPTransport::MessageHandler {
  public:
    explicit RawHandler(MCPTransport *transport);

    void Received(const Notification &note) override;
    void Received(const Request &req) override;
    void Received(const Response &resp) override;
    void OnError(llvm::Error err) override;
    void OnClosed() override;

    void Send(const Request &req,
              Callback<void(const Response &)> response_handler);
    void Send(const Notification &note);
    void Send(const Response &resp);

    friend class Binder;

  private:
    std::recursive_mutex m_mutex;
    MCPTransport *m_transport;
    int m_seq = 0;
    std::map<Id, Callback<void(const Response &)>> m_pending_responses;
    llvm::StringMap<
        Callback<void(const Request &, Callback<void(const Response &)>)>>
        m_request_handlers;
    llvm::StringMap<Callback<void(const Notification &)>>
        m_notification_handlers;
    Callback<void(MCPTransport *)> m_disconnect_handler;
    Callback<void(MCPTransport *, llvm::Error)> m_error_handler;
  };

  RawHandler m_handlers;
};
using BinderUP = std::unique_ptr<Binder>;

} // namespace lldb_protocol::mcp

#endif
