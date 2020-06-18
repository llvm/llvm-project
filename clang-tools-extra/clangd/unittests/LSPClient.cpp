#include "LSPClient.h"
#include "gtest/gtest.h"
#include <condition_variable>

#include "Protocol.h"
#include "TestFS.h"
#include "Transport.h"
#include "support/Threading.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>

namespace clang {
namespace clangd {

llvm::Expected<llvm::json::Value> clang::clangd::LSPClient::CallResult::take() {
  std::unique_lock<std::mutex> Lock(Mu);
  if (!clangd::wait(Lock, CV, timeoutSeconds(10),
                    [this] { return Value.hasValue(); })) {
    ADD_FAILURE() << "No result from call after 10 seconds!";
    return llvm::json::Value(nullptr);
  }
  auto Res = std::move(*Value);
  Value.reset();
  return Res;
}

llvm::json::Value LSPClient::CallResult::takeValue() {
  auto ExpValue = take();
  if (!ExpValue) {
    ADD_FAILURE() << "takeValue(): " << llvm::toString(ExpValue.takeError());
    return llvm::json::Value(nullptr);
  }
  return std::move(*ExpValue);
}

void LSPClient::CallResult::set(llvm::Expected<llvm::json::Value> V) {
  std::lock_guard<std::mutex> Lock(Mu);
  if (Value) {
    ADD_FAILURE() << "Multiple replies";
    llvm::consumeError(V.takeError());
    return;
  }
  Value = std::move(V);
  CV.notify_all();
}

LSPClient::CallResult::~CallResult() {
  if (Value && !*Value) {
    ADD_FAILURE() << llvm::toString(Value->takeError());
  }
}

static void logBody(llvm::StringRef Method, llvm::json::Value V, bool Send) {
  // We invert <<< and >>> as the combined log is from the server's viewpoint.
  vlog("{0} {1}: {2:2}", Send ? "<<<" : ">>>", Method, V);
}

class LSPClient::TransportImpl : public Transport {
public:
  std::pair<llvm::json::Value, CallResult *> addCallSlot() {
    std::lock_guard<std::mutex> Lock(Mu);
    unsigned ID = CallResults.size();
    CallResults.emplace_back();
    return {ID, &CallResults.back()};
  }

  // A null action causes the transport to shut down.
  void enqueue(std::function<void(MessageHandler &)> Action) {
    std::lock_guard<std::mutex> Lock(Mu);
    Actions.push(std::move(Action));
    CV.notify_all();
  }

  std::vector<llvm::json::Value> takeNotifications(llvm::StringRef Method) {
    std::vector<llvm::json::Value> Result;
    {
      std::lock_guard<std::mutex> Lock(Mu);
      std::swap(Result, Notifications[Method]);
    }
    return Result;
  }

private:
  void reply(llvm::json::Value ID,
             llvm::Expected<llvm::json::Value> V) override {
    if (V) // Nothing additional to log for error.
      logBody("reply", *V, /*Send=*/false);
    std::lock_guard<std::mutex> Lock(Mu);
    if (auto I = ID.getAsInteger()) {
      if (*I >= 0 && *I < static_cast<int64_t>(CallResults.size())) {
        CallResults[*I].set(std::move(V));
        return;
      }
    }
    ADD_FAILURE() << "Invalid reply to ID " << ID;
    llvm::consumeError(std::move(V).takeError());
  }

  void notify(llvm::StringRef Method, llvm::json::Value V) override {
    logBody(Method, V, /*Send=*/false);
    std::lock_guard<std::mutex> Lock(Mu);
    Notifications[Method].push_back(std::move(V));
  }

  void call(llvm::StringRef Method, llvm::json::Value Params,
            llvm::json::Value ID) override {
    logBody(Method, Params, /*Send=*/false);
    ADD_FAILURE() << "Unexpected server->client call " << Method;
  }

  llvm::Error loop(MessageHandler &H) override {
    std::unique_lock<std::mutex> Lock(Mu);
    while (true) {
      CV.wait(Lock, [&] { return !Actions.empty(); });
      if (!Actions.front()) // Stop!
        return llvm::Error::success();
      auto Action = std::move(Actions.front());
      Actions.pop();
      Lock.unlock();
      Action(H);
      Lock.lock();
    }
  }

  std::mutex Mu;
  std::deque<CallResult> CallResults;
  std::queue<std::function<void(Transport::MessageHandler &)>> Actions;
  std::condition_variable CV;
  llvm::StringMap<std::vector<llvm::json::Value>> Notifications;
};

LSPClient::LSPClient() : T(std::make_unique<TransportImpl>()) {}
LSPClient::~LSPClient() = default;

LSPClient::CallResult &LSPClient::call(llvm::StringRef Method,
                                       llvm::json::Value Params) {
  auto Slot = T->addCallSlot();
  T->enqueue([ID(Slot.first), Method(Method.str()),
              Params(std::move(Params))](Transport::MessageHandler &H) {
    logBody(Method, Params, /*Send=*/true);
    H.onCall(Method, std::move(Params), ID);
  });
  return *Slot.second;
}

void LSPClient::notify(llvm::StringRef Method, llvm::json::Value Params) {
  T->enqueue([Method(Method.str()),
              Params(std::move(Params))](Transport::MessageHandler &H) {
    logBody(Method, Params, /*Send=*/true);
    H.onNotify(Method, std::move(Params));
  });
}

std::vector<llvm::json::Value>
LSPClient::takeNotifications(llvm::StringRef Method) {
  return T->takeNotifications(Method);
}

void LSPClient::stop() { T->enqueue(nullptr); }

Transport &LSPClient::transport() { return *T; }

using Obj = llvm::json::Object;

llvm::json::Value LSPClient::uri(llvm::StringRef Path) {
  std::string Storage;
  if (!llvm::sys::path::is_absolute(Path))
    Path = Storage = testPath(Path);
  return toJSON(URIForFile::canonicalize(Path, Path));
}
llvm::json::Value LSPClient::documentID(llvm::StringRef Path) {
  return Obj{{"uri", uri(Path)}};
}

void LSPClient::didOpen(llvm::StringRef Path, llvm::StringRef Content) {
  notify(
      "textDocument/didOpen",
      Obj{{"textDocument",
           Obj{{"uri", uri(Path)}, {"text", Content}, {"languageId", "cpp"}}}});
}
void LSPClient::didChange(llvm::StringRef Path, llvm::StringRef Content) {
  notify("textDocument/didChange",
         Obj{{"textDocument", documentID(Path)},
             {"contentChanges", llvm::json::Array{Obj{{"text", Content}}}}});
}
void LSPClient::didClose(llvm::StringRef Path) {
  notify("textDocument/didClose", Obj{{"textDocument", documentID(Path)}});
}

void LSPClient::sync() { call("sync", nullptr).takeValue(); }

llvm::Optional<std::vector<llvm::json::Value>>
LSPClient::diagnostics(llvm::StringRef Path) {
  sync();
  auto Notifications = takeNotifications("textDocument/publishDiagnostics");
  for (const auto &Notification : llvm::reverse(Notifications)) {
    if (const auto *PubDiagsParams = Notification.getAsObject()) {
      auto U = PubDiagsParams->getString("uri");
      auto *D = PubDiagsParams->getArray("diagnostics");
      if (!U || !D) {
        ADD_FAILURE() << "Bad PublishDiagnosticsParams: " << PubDiagsParams;
        continue;
      }
      if (*U == uri(Path))
        return std::vector<llvm::json::Value>(D->begin(), D->end());
    }
  }
  return {};
}

} // namespace clangd
} // namespace clang
