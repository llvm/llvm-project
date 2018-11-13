//===-- Reproducer.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REPRODUCER_H
#define LLDB_UTILITY_REPRODUCER_H

#include "lldb/Utility/FileSpec.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/YAMLTraits.h"

#include <mutex>
#include <string>
#include <vector>

namespace lldb_private {
namespace repro {

class Reproducer;

/// Abstraction for information associated with a provider. This information
/// is serialized into an index which is used by the loader.
struct ProviderInfo {
  std::string name;
  std::vector<std::string> files;
};

/// The provider defines an interface for generating files needed for
/// reproducing. The provider must populate its ProviderInfo to communicate
/// its name and files to the index, before registering with the generator,
/// i.e. in the constructor.
///
/// Different components will implement different providers.
class Provider {
public:
  virtual ~Provider() = default;

  const ProviderInfo &GetInfo() { return m_info; }
  const FileSpec &GetDirectory() { return m_directory; }

  /// The Keep method is called when it is decided that we need to keep the
  /// data in order to provide a reproducer.
  virtual void Keep(){};

  /// The Discard method is called when it is decided that we do not need to
  /// keep any information and will not generate a reproducer.
  virtual void Discard(){};

protected:
  Provider(const FileSpec &directory) : m_directory(directory) {}

  /// Every provider keeps track of its own files.
  ProviderInfo m_info;

private:
  /// Every provider knows where to dump its potential files.
  FileSpec m_directory;
};

/// The generator is responsible for the logic needed to generate a
/// reproducer. For doing so it relies on providers, who serialize data that
/// is necessary for reproducing  a failure.
class Generator final {
public:
  Generator();
  ~Generator();

  /// Method to indicate we want to keep the reproducer. If reproducer
  /// generation is disabled, this does nothing.
  void Keep();

  /// Method to indicate we do not want to keep the reproducer. This is
  /// unaffected by whether or not generation reproduction is enabled, as we
  /// might need to clean up files already written to disk.
  void Discard();

  /// Providers are registered at creating time.
  template <typename T> T &CreateProvider() {
    std::unique_ptr<T> provider = llvm::make_unique<T>(m_directory);
    return static_cast<T &>(Register(std::move(provider)));
  }

  void ChangeDirectory(const FileSpec &directory);
  const FileSpec &GetDirectory() const;

private:
  friend Reproducer;

  void SetEnabled(bool enabled) { m_enabled = enabled; }
  Provider &Register(std::unique_ptr<Provider> provider);
  void AddProviderToIndex(const ProviderInfo &provider_info);

  std::vector<std::unique_ptr<Provider>> m_providers;
  std::mutex m_providers_mutex;

  /// The reproducer root directory.
  FileSpec m_directory;

  /// Flag for controlling whether we generate a reproducer when Keep is
  /// called.
  bool m_enabled;

  /// Flag to ensure that we never call both keep and discard.
  bool m_done;
};

class Loader final {
public:
  Loader();

  llvm::Optional<ProviderInfo> GetProviderInfo(llvm::StringRef name);
  llvm::Error LoadIndex(const FileSpec &directory);

  const FileSpec &GetDirectory() { return m_directory; }

private:
  llvm::StringMap<ProviderInfo> m_provider_info;
  FileSpec m_directory;
  bool m_loaded;
};

/// The reproducer enables clients to obtain access to the Generator and
/// Loader.
class Reproducer final {

public:
  static Reproducer &Instance();

  Generator *GetGenerator();
  Loader *GetLoader();

  const Generator *GetGenerator() const;
  const Loader *GetLoader() const;

  llvm::Error SetGenerateReproducer(bool value);
  llvm::Error SetReplayReproducer(bool value);

  llvm::Error SetReproducerPath(const FileSpec &path);
  FileSpec GetReproducerPath() const;

private:
  Generator m_generator;
  Loader m_loader;

  bool m_generate_reproducer = false;
  bool m_replay_reproducer = false;

  mutable std::mutex m_mutex;
};

} // namespace repro
} // namespace lldb_private

LLVM_YAML_IS_DOCUMENT_LIST_VECTOR(lldb_private::repro::ProviderInfo)

namespace llvm {
namespace yaml {

template <> struct MappingTraits<lldb_private::repro::ProviderInfo> {
  static void mapping(IO &io, lldb_private::repro::ProviderInfo &info) {
    io.mapRequired("name", info.name);
    io.mapOptional("files", info.files);
  }
};
} // namespace yaml
} // namespace llvm

#endif // LLDB_UTILITY_REPRODUCER_H
