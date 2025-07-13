//===-- ClangdLSPServerTests.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "ClangdLSPServer.h"
#include "ClangdServer.h"
#include "ConfigProvider.h"
#include "Diagnostics.h"
#include "Feature.h"
#include "FeatureModule.h"
#include "LSPBinder.h"
#include "LSPClient.h"
#include "TestFS.h"
#include "support/Function.h"
#include "support/Logger.h"
#include "support/TestTracer.h"
#include "support/Threading.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace clang {
namespace clangd {
namespace {
using testing::ElementsAre;

MATCHER_P(diagMessage, M, "") {
  if (const auto *O = arg.getAsObject()) {
    if (const auto Msg = O->getString("message"))
      return *Msg == M;
  }
  return false;
}

class LSPTest : public ::testing::Test {
protected:
  LSPTest() : LogSession(L) {
    ClangdServer::Options &Base = Opts;
    Base = ClangdServer::optsForTest();
    // This is needed to we can test index-based operations like call hierarchy.
    Base.BuildDynamicSymbolIndex = true;
    Base.FeatureModules = &FeatureModules;
  }

  LSPClient &start() {
    EXPECT_FALSE(Server) << "Already initialized";
    Server.emplace(Client.transport(), FS, Opts);
    ServerThread.emplace([&] { EXPECT_TRUE(Server->run()); });
    Client.call("initialize", llvm::json::Object{});
    return Client;
  }

  void stop() {
    assert(Server);
    Client.call("shutdown", nullptr);
    Client.notify("exit", nullptr);
    Client.stop();
    ServerThread->join();
    Server.reset();
    ServerThread.reset();
  }

  ~LSPTest() {
    if (Server)
      stop();
  }

  MockFS FS;
  ClangdLSPServer::Options Opts;
  FeatureModuleSet FeatureModules;

private:
  class Logger : public clang::clangd::Logger {
    // Color logs so we can distinguish them from test output.
    void log(Level L, const char *Fmt,
             const llvm::formatv_object_base &Message) override {
      raw_ostream::Colors Color;
      switch (L) {
      case Level::Verbose:
        Color = raw_ostream::BLUE;
        break;
      case Level::Error:
        Color = raw_ostream::RED;
        break;
      default:
        Color = raw_ostream::YELLOW;
        break;
      }
      std::lock_guard<std::mutex> Lock(LogMu);
      (llvm::outs().changeColor(Color) << Message << "\n").resetColor();
    }
    std::mutex LogMu;
  };

  Logger L;
  LoggingSession LogSession;
  std::optional<ClangdLSPServer> Server;
  std::optional<std::thread> ServerThread;
  LSPClient Client;
};

TEST_F(LSPTest, GoToDefinition) {
  Annotations Code(R"cpp(
    int [[fib]](int n) {
      return n >= 2 ? ^fib(n - 1) + fib(n - 2) : 1;
    }
  )cpp");
  auto &Client = start();
  Client.didOpen("foo.cpp", Code.code());
  auto &Def = Client.call("textDocument/definition",
                          llvm::json::Object{
                              {"textDocument", Client.documentID("foo.cpp")},
                              {"position", Code.point()},
                          });
  llvm::json::Value Want = llvm::json::Array{llvm::json::Object{
      {"uri", Client.uri("foo.cpp")}, {"range", Code.range()}}};
  EXPECT_EQ(Def.takeValue(), Want);
}

TEST_F(LSPTest, Diagnostics) {
  auto &Client = start();
  Client.didOpen("foo.cpp", "void main(int, char**);");
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("'main' must return 'int' (fix available)"))));

  Client.didChange("foo.cpp", "int x = \"42\";");
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("Cannot initialize a variable of type 'int' with "
                              "an lvalue of type 'const char[3]'"))));

  Client.didClose("foo.cpp");
  EXPECT_THAT(Client.diagnostics("foo.cpp"), llvm::ValueIs(testing::IsEmpty()));
}

TEST_F(LSPTest, DiagnosticsHeaderSaved) {
  auto &Client = start();
  Client.didOpen("foo.cpp", R"cpp(
    #include "foo.h"
    int x = VAR;
  )cpp");
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("'foo.h' file not found"),
                  diagMessage("Use of undeclared identifier 'VAR'"))));
  // Now create the header.
  FS.Files["foo.h"] = "#define VAR original";
  Client.notify(
      "textDocument/didSave",
      llvm::json::Object{{"textDocument", Client.documentID("foo.h")}});
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("Use of undeclared identifier 'original'"))));
  // Now modify the header from within the "editor".
  FS.Files["foo.h"] = "#define VAR changed";
  Client.notify(
      "textDocument/didSave",
      llvm::json::Object{{"textDocument", Client.documentID("foo.h")}});
  // Foo.cpp should be rebuilt with new diagnostics.
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("Use of undeclared identifier 'changed'"))));
}

TEST_F(LSPTest, RecordsLatencies) {
  trace::TestTracer Tracer;
  auto &Client = start();
  llvm::StringLiteral MethodName = "method_name";
  EXPECT_THAT(Tracer.takeMetric("lsp_latency", MethodName), testing::SizeIs(0));
  llvm::consumeError(Client.call(MethodName, {}).take().takeError());
  stop();
  EXPECT_THAT(Tracer.takeMetric("lsp_latency", MethodName), testing::SizeIs(1));
}

// clang-tidy's renames are converted to clangd's internal rename functionality,
// see clangd#1589 and clangd#741
TEST_F(LSPTest, ClangTidyRename) {
  // This test requires clang-tidy checks to be linked in.
  if (!CLANGD_TIDY_CHECKS)
    return;
  Annotations Header(R"cpp(
    void [[foo]]();
  )cpp");
  Annotations Source(R"cpp(
    void [[foo]]() {}
  )cpp");
  constexpr auto ClangTidyProvider = [](tidy::ClangTidyOptions &ClangTidyOpts,
                                        llvm::StringRef) {
    ClangTidyOpts.Checks = {"-*,readability-identifier-naming"};
    ClangTidyOpts.CheckOptions["readability-identifier-naming.FunctionCase"] =
        "CamelCase";
  };
  Opts.ClangTidyProvider = ClangTidyProvider;
  auto &Client = start();
  Client.didOpen("foo.hpp", Header.code());
  Client.didOpen("foo.cpp", Source.code());

  auto Diags = Client.diagnostics("foo.cpp");
  ASSERT_TRUE(Diags && !Diags->empty());
  auto RenameDiag = Diags->front();

  auto RenameCommand =
      (*Client
            .call("textDocument/codeAction",
                  llvm::json::Object{
                      {"textDocument", Client.documentID("foo.cpp")},
                      {"context",
                       llvm::json::Object{
                           {"diagnostics", llvm::json::Array{RenameDiag}}}},
                      {"range", Source.range()}})
            .takeValue()
            .getAsArray())[0];

  ASSERT_EQ((*RenameCommand.getAsObject())["title"], "change 'foo' to 'Foo'");

  Client.expectServerCall("workspace/applyEdit");
  Client.call("workspace/executeCommand", RenameCommand);
  Client.sync();

  auto Params = Client.takeCallParams("workspace/applyEdit");
  auto Uri = [&](llvm::StringRef Path) {
    return Client.uri(Path).getAsString().value().str();
  };
  llvm::json::Object ExpectedEdit = llvm::json::Object{
      {"edit", llvm::json::Object{
                   {"changes",
                    llvm::json::Object{
                        {Uri("foo.hpp"), llvm::json::Array{llvm::json::Object{
                                             {"range", Header.range()},
                                             {"newText", "Foo"},
                                         }}},

                        {Uri("foo.cpp"), llvm::json::Array{llvm::json::Object{
                                             {"range", Source.range()},
                                             {"newText", "Foo"},
                                         }}}

                    }}}}};
  EXPECT_EQ(Params, std::vector{llvm::json::Value(std::move(ExpectedEdit))});
}

TEST_F(LSPTest, ClangTidyCrash_Issue109367) {
  // This test requires clang-tidy checks to be linked in.
  if (!CLANGD_TIDY_CHECKS)
    return;
  constexpr auto ClangTidyProvider = [](tidy::ClangTidyOptions &ClangTidyOpts,
                                        llvm::StringRef) {
    ClangTidyOpts.Checks = {"-*,boost-use-ranges"};
  };
  Opts.ClangTidyProvider = ClangTidyProvider;
  // Check that registering the boost-use-ranges checker's matchers
  // on two different threads does not cause a crash.
  auto &Client = start();
  Client.didOpen("a.cpp", "");
  Client.didOpen("b.cpp", "");
  Client.sync();
}

TEST_F(LSPTest, IncomingCalls) {
  Annotations Code(R"cpp(
    void calle^e(int);
    void caller1() {
      [[callee]](42);
    }
  )cpp");
  auto &Client = start();
  Client.didOpen("foo.cpp", Code.code());
  auto Items = Client
                   .call("textDocument/prepareCallHierarchy",
                         llvm::json::Object{
                             {"textDocument", Client.documentID("foo.cpp")},
                             {"position", Code.point()}})
                   .takeValue();
  auto FirstItem = (*Items.getAsArray())[0];
  auto Calls = Client
                   .call("callHierarchy/incomingCalls",
                         llvm::json::Object{{"item", FirstItem}})
                   .takeValue();
  auto FirstCall = *(*Calls.getAsArray())[0].getAsObject();
  EXPECT_EQ(FirstCall["fromRanges"], llvm::json::Value{Code.range()});
  auto From = *FirstCall["from"].getAsObject();
  EXPECT_EQ(From["name"], "caller1");
}

TEST_F(LSPTest, CDBConfigIntegration) {
  auto CfgProvider =
      config::Provider::fromAncestorRelativeYAMLFiles(".clangd", FS);
  Opts.ConfigProvider = CfgProvider.get();

  // Map bar.cpp to a different compilation database which defines FOO->BAR.
  FS.Files[".clangd"] = R"yaml(
If:
  PathMatch: bar.cpp
CompileFlags:
  CompilationDatabase: bar
)yaml";
  FS.Files["bar/compile_flags.txt"] = "-DFOO=BAR";

  auto &Client = start();
  // foo.cpp gets parsed as normal.
  Client.didOpen("foo.cpp", "int x = FOO;");
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("Use of undeclared identifier 'FOO'"))));
  // bar.cpp shows the configured compile command.
  Client.didOpen("bar.cpp", "int x = FOO;");
  EXPECT_THAT(Client.diagnostics("bar.cpp"),
              llvm::ValueIs(testing::ElementsAre(
                  diagMessage("Use of undeclared identifier 'BAR'"))));
}

TEST_F(LSPTest, ModulesTest) {
  class MathModule final : public FeatureModule {
    OutgoingNotification<int> Changed;
    void initializeLSP(LSPBinder &Bind, const llvm::json::Object &ClientCaps,
                       llvm::json::Object &ServerCaps) override {
      Bind.notification("add", this, &MathModule::add);
      Bind.method("get", this, &MathModule::get);
      Changed = Bind.outgoingNotification("changed");
    }

    int Value = 0;

    void add(const int &X) {
      Value += X;
      Changed(Value);
    }
    void get(const std::nullptr_t &, Callback<int> Reply) {
      scheduler().runQuick(
          "get", "",
          [Reply(std::move(Reply)), Value(Value)]() mutable { Reply(Value); });
    }
  };
  FeatureModules.add(std::make_unique<MathModule>());

  auto &Client = start();
  Client.notify("add", 2);
  Client.notify("add", 8);
  EXPECT_EQ(10, Client.call("get", nullptr).takeValue());
  EXPECT_THAT(Client.takeNotifications("changed"),
              ElementsAre(llvm::json::Value(2), llvm::json::Value(10)));
}

// Creates a Callback that writes its received value into an
// std::optional<Expected>.
template <typename T>
llvm::unique_function<void(llvm::Expected<T>)>
capture(std::optional<llvm::Expected<T>> &Out) {
  Out.reset();
  return [&Out](llvm::Expected<T> V) { Out.emplace(std::move(V)); };
}

TEST_F(LSPTest, FeatureModulesThreadingTest) {
  // A feature module that does its work on a background thread, and so
  // exercises the block/shutdown protocol.
  class AsyncCounter final : public FeatureModule {
    bool ShouldStop = false;
    int State = 0;
    std::deque<Callback<int>> Queue; // null = increment, non-null = read.
    std::condition_variable CV;
    std::mutex Mu;
    std::thread Thread;

    void run() {
      std::unique_lock<std::mutex> Lock(Mu);
      while (true) {
        CV.wait(Lock, [&] { return ShouldStop || !Queue.empty(); });
        if (ShouldStop) {
          Queue.clear();
          CV.notify_all();
          return;
        }
        Callback<int> &Task = Queue.front();
        if (Task)
          Task(State);
        else
          ++State;
        Queue.pop_front();
        CV.notify_all();
      }
    }

    bool blockUntilIdle(Deadline D) override {
      std::unique_lock<std::mutex> Lock(Mu);
      return clangd::wait(Lock, CV, D, [this] { return Queue.empty(); });
    }

    void stop() override {
      {
        std::lock_guard<std::mutex> Lock(Mu);
        ShouldStop = true;
      }
      CV.notify_all();
    }

  public:
    AsyncCounter() : Thread([this] { run(); }) {}
    ~AsyncCounter() {
      // Verify shutdown sequence was performed.
      // Real modules would not do this, to be robust to no ClangdServer.
      {
        // We still need the lock here, as Queue might be empty when
        // ClangdServer calls blockUntilIdle, but run() might not have returned
        // yet.
        std::lock_guard<std::mutex> Lock(Mu);
        EXPECT_TRUE(ShouldStop) << "ClangdServer should request shutdown";
        EXPECT_EQ(Queue.size(), 0u) << "ClangdServer should block until idle";
      }
      Thread.join();
    }

    void initializeLSP(LSPBinder &Bind, const llvm::json::Object &ClientCaps,
                       llvm::json::Object &ServerCaps) override {
      Bind.notification("increment", this, &AsyncCounter::increment);
    }

    // Get the current value, bypassing the queue.
    // Used to verify that sync->blockUntilIdle avoids races in tests.
    int getSync() {
      std::lock_guard<std::mutex> Lock(Mu);
      return State;
    }

    // Increment the current value asynchronously.
    void increment(const std::nullptr_t &) {
      {
        std::lock_guard<std::mutex> Lock(Mu);
        Queue.push_back(nullptr);
      }
      CV.notify_all();
    }
  };

  FeatureModules.add(std::make_unique<AsyncCounter>());
  auto &Client = start();

  Client.notify("increment", nullptr);
  Client.notify("increment", nullptr);
  Client.notify("increment", nullptr);
  Client.sync();
  EXPECT_EQ(3, FeatureModules.get<AsyncCounter>()->getSync());
  // Throw some work on the queue to make sure shutdown blocks on it.
  Client.notify("increment", nullptr);
  Client.notify("increment", nullptr);
  Client.notify("increment", nullptr);
  // And immediately shut down. FeatureModule destructor verifies we blocked.
}

TEST_F(LSPTest, DiagModuleTest) {
  static constexpr llvm::StringLiteral DiagMsg = "DiagMsg";
  class DiagModule final : public FeatureModule {
    struct DiagHooks : public ASTListener {
      void sawDiagnostic(const clang::Diagnostic &, clangd::Diag &D) override {
        D.Message = DiagMsg.str();
      }
    };

  public:
    std::unique_ptr<ASTListener> astListeners() override {
      return std::make_unique<DiagHooks>();
    }
  };
  FeatureModules.add(std::make_unique<DiagModule>());

  auto &Client = start();
  Client.didOpen("foo.cpp", "test;");
  EXPECT_THAT(Client.diagnostics("foo.cpp"),
              llvm::ValueIs(testing::ElementsAre(diagMessage(DiagMsg))));
}
} // namespace
} // namespace clangd
} // namespace clang
