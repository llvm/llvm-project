//===-- Diagnostics.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Diagnostics.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Interpreter/CommandReturnObject.h"
#include "lldb/Interpreter/OptionValueProperties.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/Platform.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Statistics.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/LLDBAssert.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Version/Version.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <optional>
#include <string>
#include <vector>

using namespace lldb_private;
using namespace lldb;
using namespace llvm;

namespace {

#define LLDB_PROPERTIES_diagnostics
#include "CoreProperties.inc"

enum {
#define LLDB_PROPERTIES_diagnostics
#include "CorePropertiesEnum.inc"
};

} // namespace

DiagnosticsProperties::DiagnosticsProperties() {
  m_collection_sp = std::make_shared<OptionValueProperties>("diagnostics");
  m_collection_sp->Initialize(g_diagnostics_properties_def);
}

bool DiagnosticsProperties::GetCollectBinaries() const {
  const uint32_t idx = ePropertyCollectBinaries;
  return GetPropertyAtIndexAs<bool>(
      idx, g_diagnostics_properties[idx].default_uint_value != 0);
}

bool DiagnosticsProperties::SetCollectBinaries(bool collect) {
  return SetPropertyAtIndex(ePropertyCollectBinaries, collect);
}

DiagnosticsProperties &Diagnostics::GetGlobalProperties() {
  static DiagnosticsProperties g_settings;
  return g_settings;
}

static constexpr size_t g_num_log_messages = 100;

void Diagnostics::Initialize() {
  lldbassert(!InstanceImpl() && "Already initialized.");
  InstanceImpl().emplace();
}

void Diagnostics::Terminate() {
  lldbassert(InstanceImpl() && "Already terminated.");
  InstanceImpl().reset();
}

bool Diagnostics::Enabled() { return InstanceImpl().operator bool(); }

std::optional<Diagnostics> &Diagnostics::InstanceImpl() {
  static std::optional<Diagnostics> g_diagnostics;
  return g_diagnostics;
}

Diagnostics &Diagnostics::Instance() { return *InstanceImpl(); }

Diagnostics::Diagnostics() : m_log_handler(g_num_log_messages) {}

Diagnostics::~Diagnostics() {}

bool Diagnostics::Dump(raw_ostream &stream) {
  Expected<FileSpec> diagnostics_dir = CreateUniqueDirectory();
  if (!diagnostics_dir) {
    stream << "unable to create diagnostic dir: "
           << toString(diagnostics_dir.takeError()) << '\n';
    return false;
  }

  return Dump(stream, *diagnostics_dir);
}

bool Diagnostics::Dump(raw_ostream &stream, const FileSpec &dir) {
  stream << "LLDB diagnostics will be written to " << dir.GetPath() << "\n";
  stream << "Please include the directory content when filing a bug report\n";

  if (Error error = Create(dir)) {
    stream << toString(std::move(error)) << '\n';
    return false;
  }

  return true;
}

llvm::Expected<FileSpec> Diagnostics::CreateUniqueDirectory() {
  SmallString<128> diagnostics_dir;
  std::error_code ec =
      sys::fs::createUniqueDirectory("diagnostics", diagnostics_dir);
  if (ec)
    return errorCodeToError(ec);
  return FileSpec(diagnostics_dir.str());
}

Error Diagnostics::Create(const FileSpec &dir) {
  if (Error err = DumpDiangosticsLog(dir))
    return err;

  return Error::success();
}

llvm::Error Diagnostics::DumpDiangosticsLog(const FileSpec &dir) const {
  FileSpec log_file = dir.CopyByAppendingPathComponent("diagnostics.log");
  std::error_code ec;
  llvm::raw_fd_ostream stream(log_file.GetPath(), ec, llvm::sys::fs::OF_None);
  if (ec)
    return errorCodeToError(ec);
  m_log_handler.Dump(stream);
  return Error::success();
}

void Diagnostics::Record(llvm::StringRef message) {
  m_log_handler.Emit(message);
}

Diagnostics::ArtifactProviderID
Diagnostics::AddArtifactProvider(std::string name, ArtifactProvider provider) {
  std::lock_guard<std::mutex> guard(m_artifact_providers_mutex);
  ArtifactProviderID id = m_next_artifact_provider_id++;
  m_artifact_providers.push_back({id, std::move(name), std::move(provider)});
  return id;
}

void Diagnostics::RemoveArtifactProvider(ArtifactProviderID id) {
  std::lock_guard<std::mutex> guard(m_artifact_providers_mutex);
  llvm::erase_if(m_artifact_providers,
                 [id](const ArtifactProviderEntry &e) { return e.id == id; });
}

// Write a single artifact into the bundle and, on success, record its name in
// \p files. Best-effort: a write failure leaves the file out of the list, so a
// missing artifact stays visible. The file is made owner-only because the
// bundle can contain paths, argv, and command history.
static void WriteArtifact(const FileSpec &dir, llvm::StringRef name,
                          llvm::StringRef content,
                          std::vector<std::string> &files) {
  FileSpec file = dir.CopyByAppendingPathComponent(name);
  std::error_code ec;
  llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_Text);
  if (ec)
    return;
  os << content;
  os.flush();
  llvm::sys::fs::setPermissions(file.GetPath(), llvm::sys::fs::owner_read |
                                                    llvm::sys::fs::owner_write);
  files.push_back(name.str());
}

// Copy a file into the bundle, best-effort like WriteArtifact. The basename is
// disambiguated because an executable and its symbol file can share one (a
// Mach-O and the DWARF binary inside its .dSYM), which would otherwise clobber.
static void CopyBinary(const FileSpec &src, const FileSpec &dir,
                       std::vector<std::string> &files) {
  if (!src || !FileSystem::Instance().Exists(src))
    return;

  std::string name = src.GetFilename().str();
  FileSpec dst = dir.CopyByAppendingPathComponent(name);
  for (unsigned i = 1; FileSystem::Instance().Exists(dst); ++i) {
    name = formatv("{0}.{1}", src.GetFilename(), i).str();
    dst = dir.CopyByAppendingPathComponent(name);
  }

  if (llvm::sys::fs::copy_file(src.GetPath(), dst.GetPath()))
    return;
  llvm::sys::fs::setPermissions(dst.GetPath(), llvm::sys::fs::owner_read |
                                                   llvm::sys::fs::owner_write);
  files.push_back(std::move(name));
}

// Run a command through the interpreter and return its combined output and
// error text, for inclusion as a snapshot in the bundle.
static std::string CaptureCommand(Debugger &debugger, llvm::StringRef command) {
  CommandReturnObject result(/*colors=*/false);
  debugger.GetCommandInterpreter().HandleCommand(command.str().c_str(),
                                                 eLazyBoolNo, result);
  return (result.GetOutputString() + result.GetErrorString()).str();
}

namespace {
/// The execution context a triage command needs before it is worth running.
enum class Requires { Always, Target, Process, Frame };

struct TriageCommand {
  llvm::StringRef command;
  Requires requirement;
};
} // namespace

// The commands a triager runs first, captured into the bundle. Add a row to
// extend the snapshot. The requirement keeps a command from emitting a spurious
// "no process" error when its context is absent.
static constexpr TriageCommand g_triage_commands[] = {
    {"target list", Requires::Always},
    {"image list", Requires::Target},
    {"process status", Requires::Process},
    {"thread list", Requires::Process},
    {"thread backtrace all", Requires::Process},
    {"image lookup -va $pc", Requires::Frame},
    {"register read", Requires::Frame},
    {"frame variable", Requires::Frame},
};

static bool Available(Requires requirement, const ExecutionContext &exe_ctx) {
  switch (requirement) {
  case Requires::Always:
    return true;
  case Requires::Target:
    return exe_ctx.GetTargetPtr() != nullptr;
  case Requires::Process:
    return exe_ctx.GetProcessPtr() != nullptr;
  case Requires::Frame:
    return exe_ctx.GetFramePtr() != nullptr;
  }
  llvm_unreachable("unhandled Requires");
}

llvm::Expected<Diagnostics::Report>
Diagnostics::Collect(Debugger &debugger, const ExecutionContext &exe_ctx,
                     const FileSpec &dir) {
  // The bundle holds potentially sensitive data (paths, argv, command history),
  // so restrict the directory to the owner before writing anything into it.
  llvm::sys::fs::setPermissions(dir.GetPath(), llvm::sys::fs::owner_read |
                                                   llvm::sys::fs::owner_write |
                                                   llvm::sys::fs::owner_exe);

  Report report;
  report.attachments.directory = dir.GetPath();
  CollectLogs(debugger, dir, report.attachments.files);
  CollectStatistics(debugger, exe_ctx, dir, report.attachments.files);
  CollectCommands(debugger, exe_ctx, dir, report.attachments.files);
  if (GetGlobalProperties().GetCollectBinaries())
    CollectBinaries(exe_ctx, dir, report.attachments.files);
  CollectArtifactProviders(dir, report.attachments.files);

  report.version = lldb_private::GetVersion();
  report.os = GetHostDescription(exe_ctx);
  report.invocation = GetInvocation();
  return report;
}

void Diagnostics::CollectLogs(Debugger &debugger, const FileSpec &dir,
                              std::vector<std::string> &files) {
  // The always-on diagnostic log.
  if (Error error = Create(dir))
    consumeError(std::move(error));
  else
    files.push_back("diagnostics.log");

  // This debugger's file-backed logs.
  for (std::string &name : debugger.CopyLogFilesToDirectory(dir))
    files.push_back(std::move(name));
}

void Diagnostics::CollectStatistics(Debugger &debugger,
                                    const ExecutionContext &exe_ctx,
                                    const FileSpec &dir,
                                    std::vector<std::string> &files) {
  StatisticsOptions options;
  json::Value stats = DebuggerStats::ReportStatistics(
      debugger, exe_ctx.GetTargetPtr(), options);
  std::string str;
  raw_string_ostream os(str);
  os << formatv("{0:2}", stats);
  WriteArtifact(dir, "statistics.json", str, files);
}

void Diagnostics::CollectCommands(Debugger &debugger,
                                  const ExecutionContext &exe_ctx,
                                  const FileSpec &dir,
                                  std::vector<std::string> &files) {
  std::string snapshot;
  for (const TriageCommand &tc : g_triage_commands) {
    if (!Available(tc.requirement, exe_ctx))
      continue;
    snapshot += formatv("=== {0} ===\n", tc.command).str();
    snapshot += CaptureCommand(debugger, tc.command);
    snapshot += "\n\n";
  }
  WriteArtifact(dir, "commands.txt", snapshot, files);
}

void Diagnostics::CollectBinaries(const ExecutionContext &exe_ctx,
                                  const FileSpec &dir,
                                  std::vector<std::string> &files) {
  if (Target *target = exe_ctx.GetTargetPtr()) {
    if (Module *exe = target->GetExecutableModulePointer()) {
      CopyBinary(exe->GetFileSpec(), dir, files);
      // Skip when symbols are inline: the symbol file is then the executable.
      if (exe->GetSymbolFileFileSpec() != exe->GetFileSpec())
        CopyBinary(exe->GetSymbolFileFileSpec(), dir, files);
    }
  }

  if (Process *process = exe_ctx.GetProcessPtr())
    CopyBinary(process->GetCoreFile(), dir, files);
}

void Diagnostics::CollectArtifactProviders(const FileSpec &dir,
                                           std::vector<std::string> &files) {
  // Snapshot under the lock, then run providers without it: a provider can be
  // slow and must not block registration or removal.
  std::vector<ArtifactProviderEntry> providers;
  {
    std::lock_guard<std::mutex> guard(m_artifact_providers_mutex);
    providers = m_artifact_providers;
  }
  for (const ArtifactProviderEntry &entry : providers)
    WriteArtifact(dir, entry.name, entry.provider(), files);
}

std::string Diagnostics::GetHostDescription(const ExecutionContext &exe_ctx) {
  std::string os = HostInfo::GetTargetTriple().str();
  Target *target = exe_ctx.GetTargetPtr();
  if (!target)
    return os;
  PlatformSP platform_sp = target->GetPlatform();
  if (!platform_sp)
    return os;

  os += formatv(" platform={0}", platform_sp->GetName()).str();
  VersionTuple version = platform_sp->GetOSVersion();
  if (!version.empty())
    os += " os=" + version.getAsString();
  if (std::optional<std::string> build = platform_sp->GetOSBuildString())
    os += " build=" + *build;
  return os;
}

std::string Diagnostics::GetInvocation() {
  // libLLDB does not store its own argv, so read the invocation from the host
  // process.
  ProcessInstanceInfo info;
  if (!Host::GetProcessInfo(Host::GetCurrentProcessID(), info))
    return {};

  const Args &args = info.GetArguments();
  std::string invocation;
  for (size_t i = 0; i < args.GetArgumentCount(); ++i) {
    if (i)
      invocation += ' ';
    invocation += args.GetArgumentAtIndex(i);
  }
  return invocation;
}

llvm::json::Value lldb_private::toJSON(const Diagnostics::Report &report) {
  json::Object obj{
      {"version", report.version},
      {"os", report.os},
  };
  if (!report.invocation.empty())
    obj["invocation"] = report.invocation;
  obj["attachments"] = json::Object{
      {"directory", report.attachments.directory},
      {"files", json::Array(report.attachments.files)},
  };
  return obj;
}
