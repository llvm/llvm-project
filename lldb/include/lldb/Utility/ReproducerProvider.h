//===-- Reproducer.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REPRODUCER_PROVIDER_H
#define LLDB_UTILITY_REPRODUCER_PROVIDER_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/ProcessInfo.h"
#include "lldb/Utility/Reproducer.h"
#include "lldb/Utility/UUID.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileCollector.h"
#include "llvm/Support/YAMLTraits.h"

#include <string>
#include <utility>
#include <vector>

namespace lldb_private {
namespace repro {

/// The recorder is a small object handed out by a provider to record data. It
/// is commonly used in combination with a MultiProvider which is meant to
/// record information for multiple instances of the same source of data.
class AbstractRecorder {
protected:
  AbstractRecorder(const FileSpec &filename, std::error_code &ec)
      : m_filename(filename.GetFilename().GetStringRef()),
        m_os(filename.GetPath(), ec, llvm::sys::fs::OF_TextWithCRLF),
        m_record(true) {}

public:
  const FileSpec &GetFilename() { return m_filename; }

  void Stop() {
    assert(m_record);
    m_record = false;
  }

private:
  FileSpec m_filename;

protected:
  llvm::raw_fd_ostream m_os;
  bool m_record;
};

/// Recorder that records its data as text to a file.
class DataRecorder : public AbstractRecorder {
public:
  DataRecorder(const FileSpec &filename, std::error_code &ec)
      : AbstractRecorder(filename, ec) {}

  static llvm::Expected<std::unique_ptr<DataRecorder>>
  Create(const FileSpec &filename);

  template <typename T> void Record(const T &t, bool newline = false) {
    if (!m_record)
      return;
    m_os << t;
    if (newline)
      m_os << '\n';
    m_os.flush();
  }
};

/// Recorder that records its data as YAML to a file.
class YamlRecorder : public AbstractRecorder {
public:
  YamlRecorder(const FileSpec &filename, std::error_code &ec)
      : AbstractRecorder(filename, ec) {}

  static llvm::Expected<std::unique_ptr<YamlRecorder>>
  Create(const FileSpec &filename);

  template <typename T> void Record(const T &t) {
    if (!m_record)
      return;
    llvm::yaml::Output yout(m_os);
    // The YAML traits are defined as non-const because they are used for
    // serialization and deserialization. The cast is safe because
    // serialization doesn't modify the object.
    yout << const_cast<T &>(t);
    m_os.flush();
  }
};

class FileProvider : public Provider<FileProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  FileProvider(const FileSpec &directory) : Provider(directory) {
    m_collector = std::make_shared<llvm::FileCollector>(
        directory.CopyByAppendingPathComponent("root").GetPath(),
        directory.GetPath());
  }

  std::shared_ptr<llvm::FileCollector> GetFileCollector() {
    return m_collector;
  }

  void Keep() override;

  void RecordInterestingDirectory(const llvm::Twine &dir);
  void RecordInterestingDirectoryRecursive(const llvm::Twine &dir);

  static char ID;

private:
  std::shared_ptr<llvm::FileCollector> m_collector;
};

/// Provider for the LLDB version number.
///
/// When the reproducer is kept, it writes the lldb version to a file named
/// version.txt in the reproducer root.
class VersionProvider : public Provider<VersionProvider> {
public:
  VersionProvider(const FileSpec &directory) : Provider(directory) {}
  struct Info {
    static const char *name;
    static const char *file;
  };
  void SetVersion(std::string version) {
    assert(m_version.empty());
    m_version = std::move(version);
  }
  void Keep() override;
  std::string m_version;
  static char ID;
};

/// Abstract provider to storing directory paths.
template <typename T> class DirectoryProvider : public repro::Provider<T> {
public:
  DirectoryProvider(const FileSpec &root) : Provider<T>(root) {}
  void SetDirectory(std::string directory) {
    m_directory = std::move(directory);
  }
  llvm::StringRef GetDirectory() { return m_directory; }

  void Keep() override {
    FileSpec file = this->GetRoot().CopyByAppendingPathComponent(T::Info::file);
    std::error_code ec;
    llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return;
    os << m_directory << "\n";
  }

protected:
  std::string m_directory;
};

/// Provider for the current working directory.
///
/// When the reproducer is kept, it writes lldb's current working directory to
/// a file named cwd.txt in the reproducer root.
class WorkingDirectoryProvider
    : public DirectoryProvider<WorkingDirectoryProvider> {
public:
  WorkingDirectoryProvider(const FileSpec &directory)
      : DirectoryProvider(directory) {
    llvm::SmallString<128> cwd;
    if (std::error_code EC = llvm::sys::fs::current_path(cwd))
      return;
    SetDirectory(std::string(cwd));
  }
  struct Info {
    static const char *name;
    static const char *file;
  };
  static char ID;
};

/// Provider for the home directory.
///
/// When the reproducer is kept, it writes the user's home directory to a file
/// a file named home.txt in the reproducer root.
class HomeDirectoryProvider : public DirectoryProvider<HomeDirectoryProvider> {
public:
  HomeDirectoryProvider(const FileSpec &directory)
      : DirectoryProvider(directory) {
    llvm::SmallString<128> home_dir;
    llvm::sys::path::home_directory(home_dir);
    SetDirectory(std::string(home_dir));
  }
  struct Info {
    static const char *name;
    static const char *file;
  };
  static char ID;
};

/// Provider for mapping UUIDs to symbol and executable files.
class SymbolFileProvider : public Provider<SymbolFileProvider> {
public:
  SymbolFileProvider(const FileSpec &directory) : Provider(directory) {}

  void AddSymbolFile(const UUID *uuid, const FileSpec &module_path,
                     const FileSpec &symbol_path);
  void Keep() override;

  struct Entry {
    Entry() = default;
    Entry(std::string uuid) : uuid(std::move(uuid)) {}
    Entry(std::string uuid, std::string module_path, std::string symbol_path)
        : uuid(std::move(uuid)), module_path(std::move(module_path)),
          symbol_path(std::move(symbol_path)) {}

    bool operator==(const Entry &rhs) const { return uuid == rhs.uuid; }
    bool operator<(const Entry &rhs) const { return uuid < rhs.uuid; }

    std::string uuid;
    std::string module_path;
    std::string symbol_path;
  };

  struct Info {
    static const char *name;
    static const char *file;
  };
  static char ID;

private:
  std::vector<Entry> m_symbol_files;
};

/// The MultiProvider is a provider that hands out recorder which can be used
/// to capture data for different instances of the same object. The recorders
/// can be passed around or stored as an instance member.
///
/// The Info::file for the MultiProvider contains an index of files for every
/// recorder. Use the MultiLoader to read the index and get the individual
/// files.
template <typename T, typename V>
class MultiProvider : public repro::Provider<V> {
public:
  MultiProvider(const FileSpec &directory) : Provider<V>(directory) {}

  T *GetNewRecorder() {
    std::size_t i = m_recorders.size() + 1;
    std::string filename = (llvm::Twine(V::Info::name) + llvm::Twine("-") +
                            llvm::Twine(i) + llvm::Twine(".yaml"))
                               .str();
    auto recorder_or_error =
        T::Create(this->GetRoot().CopyByAppendingPathComponent(filename));
    if (!recorder_or_error) {
      llvm::consumeError(recorder_or_error.takeError());
      return nullptr;
    }

    m_recorders.push_back(std::move(*recorder_or_error));
    return m_recorders.back().get();
  }

  void Keep() override {
    std::vector<std::string> files;
    for (auto &recorder : m_recorders) {
      recorder->Stop();
      files.push_back(recorder->GetFilename().GetPath());
    }

    FileSpec file = this->GetRoot().CopyByAppendingPathComponent(V::Info::file);
    std::error_code ec;
    llvm::raw_fd_ostream os(file.GetPath(), ec, llvm::sys::fs::OF_TextWithCRLF);
    if (ec)
      return;
    llvm::yaml::Output yout(os);
    yout << files;
  }

  void Discard() override { m_recorders.clear(); }

private:
  std::vector<std::unique_ptr<T>> m_recorders;
};

class CommandProvider : public MultiProvider<DataRecorder, CommandProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  CommandProvider(const FileSpec &directory)
      : MultiProvider<DataRecorder, CommandProvider>(directory) {}

  static char ID;
};

class ProcessInfoRecorder : public AbstractRecorder {
public:
  ProcessInfoRecorder(const FileSpec &filename, std::error_code &ec)
      : AbstractRecorder(filename, ec) {}

  static llvm::Expected<std::unique_ptr<ProcessInfoRecorder>>
  Create(const FileSpec &filename);

  void Record(const ProcessInstanceInfoList &process_infos);
};

class ProcessInfoProvider : public repro::Provider<ProcessInfoProvider> {
public:
  struct Info {
    static const char *name;
    static const char *file;
  };

  ProcessInfoProvider(const FileSpec &directory) : Provider(directory) {}

  ProcessInfoRecorder *GetNewProcessInfoRecorder();

  void Keep() override;
  void Discard() override;

  static char ID;

private:
  std::unique_ptr<llvm::raw_fd_ostream> m_stream_up;
  std::vector<std::unique_ptr<ProcessInfoRecorder>> m_process_info_recorders;
};

/// Loader for data captured with the MultiProvider. It will read the index and
/// return the path to the files in the index.
template <typename T> class MultiLoader {
public:
  MultiLoader(std::vector<std::string> files) : m_files(std::move(files)) {}

  static std::unique_ptr<MultiLoader> Create(Loader *loader) {
    if (!loader)
      return {};

    FileSpec file = loader->GetFile<typename T::Info>();
    if (!file)
      return {};

    auto error_or_file = llvm::MemoryBuffer::getFile(file.GetPath());
    if (auto err = error_or_file.getError())
      return {};

    std::vector<std::string> files;
    llvm::yaml::Input yin((*error_or_file)->getBuffer());
    yin >> files;

    if (auto err = yin.error())
      return {};

    for (auto &file : files) {
      FileSpec absolute_path =
          loader->GetRoot().CopyByAppendingPathComponent(file);
      file = absolute_path.GetPath();
    }

    return std::make_unique<MultiLoader<T>>(std::move(files));
  }

  llvm::Optional<std::string> GetNextFile() {
    if (m_index >= m_files.size())
      return {};
    return m_files[m_index++];
  }

private:
  std::vector<std::string> m_files;
  unsigned m_index = 0;
};

class SymbolFileLoader {
public:
  SymbolFileLoader(Loader *loader);
  std::pair<FileSpec, FileSpec> GetPaths(const UUID *uuid) const;

private:
  // Sorted list of UUID to path mappings.
  std::vector<SymbolFileProvider::Entry> m_symbol_files;
};

/// Helper to read directories written by the DirectoryProvider.
template <typename T>
llvm::Expected<std::string> GetDirectoryFrom(repro::Loader *loader) {
  llvm::Expected<std::string> dir = loader->LoadBuffer<T>();
  if (!dir)
    return dir.takeError();
  return std::string(llvm::StringRef(*dir).rtrim());
}

} // namespace repro
} // namespace lldb_private

LLVM_YAML_IS_SEQUENCE_VECTOR(lldb_private::repro::SymbolFileProvider::Entry)

namespace llvm {
namespace yaml {
template <>
struct MappingTraits<lldb_private::repro::SymbolFileProvider::Entry> {
  static void mapping(IO &io,
                      lldb_private::repro::SymbolFileProvider::Entry &entry) {
    io.mapRequired("uuid", entry.uuid);
    io.mapRequired("module-path", entry.module_path);
    io.mapRequired("symbol-path", entry.symbol_path);
  }
};
} // namespace yaml
} // namespace llvm

#endif // LLDB_UTILITY_REPRODUCER_PROVIDER_H
