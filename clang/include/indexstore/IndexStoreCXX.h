//===--- IndexStoreCXX.h - C++ wrapper for the Index Store C API. ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Header-only C++ wrapper for the Index Store C API.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEXSTORE_INDEXSTORECXX_H
#define LLVM_CLANG_INDEXSTORE_INDEXSTORECXX_H

#include "indexstore/indexstore.h"
#include "clang/Basic/PathRemapper.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"

namespace indexstore {
  using llvm::ArrayRef;
  using llvm::Optional;
  using llvm::StringRef;

static inline StringRef stringFromIndexStoreStringRef(indexstore_string_ref_t str) {
  return StringRef(str.data, str.length);
}

template<typename Ret, typename ...Params>
static inline Ret functionPtrFromFunctionRef(void *ctx, Params ...params) {
  auto fn = (llvm::function_ref<Ret(Params...)> *)ctx;
  return (*fn)(std::forward<Params>(params)...);
}

class IndexRecordSymbol {
  indexstore_symbol_t obj;
  friend class IndexRecordReader;

public:
  IndexRecordSymbol(indexstore_symbol_t obj) : obj(obj) {}

  indexstore_symbol_language_t getLanguage() {
    return indexstore_symbol_get_language(obj);
  }
  indexstore_symbol_kind_t getKind() { return indexstore_symbol_get_kind(obj); }
  indexstore_symbol_subkind_t getSubKind() { return indexstore_symbol_get_subkind(obj); }
  uint64_t getProperties() {
    return indexstore_symbol_get_properties(obj);
  }
  uint64_t getRoles() { return indexstore_symbol_get_roles(obj); }
  uint64_t getRelatedRoles() { return indexstore_symbol_get_related_roles(obj); }
  StringRef getName() { return stringFromIndexStoreStringRef(indexstore_symbol_get_name(obj)); }
  StringRef getUSR() { return stringFromIndexStoreStringRef(indexstore_symbol_get_usr(obj)); }
  StringRef getCodegenName() { return stringFromIndexStoreStringRef(indexstore_symbol_get_codegen_name(obj)); }
};

class IndexSymbolRelation {
  indexstore_symbol_relation_t obj;

public:
  IndexSymbolRelation(indexstore_symbol_relation_t obj) : obj(obj) {}

  uint64_t getRoles() { return indexstore_symbol_relation_get_roles(obj); }
  IndexRecordSymbol getSymbol() { return indexstore_symbol_relation_get_symbol(obj); }
};

class IndexRecordOccurrence {
  indexstore_occurrence_t obj;

public:
  IndexRecordOccurrence(indexstore_occurrence_t obj) : obj(obj) {}

  IndexRecordSymbol getSymbol() { return indexstore_occurrence_get_symbol(obj); }
  uint64_t getRoles() { return indexstore_occurrence_get_roles(obj); }

  bool foreachRelation(llvm::function_ref<bool(IndexSymbolRelation)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_occurrence_relations_apply(obj, ^bool(indexstore_symbol_relation_t sym_rel) {
      return receiver(sym_rel);
    });
#else
    return indexstore_occurrence_relations_apply_f(obj, &receiver, functionPtrFromFunctionRef);
#endif
  }

  std::pair<unsigned, unsigned> getLineCol() {
    unsigned line, col;
    indexstore_occurrence_get_line_col(obj, &line, &col);
    return std::make_pair(line, col);
  }
};

class IndexStore;
typedef std::shared_ptr<IndexStore> IndexStoreRef;

class IndexStore {
  indexstore_t obj;
  friend class IndexRecordReader;
  friend class IndexUnitReader;

public:
  IndexStore(StringRef path, const clang::PathRemapper &remapper,
             std::string &error) {
    llvm::SmallString<64> buf = path;
    indexstore_error_t c_err = nullptr;
    indexstore_creation_options_t options = indexstore_creation_options_create();
    for (const auto &Mapping : remapper.getMappings())
      indexstore_creation_options_add_prefix_mapping(options, Mapping.first.c_str(), Mapping.second.c_str());

    obj = indexstore_store_create_with_options(buf.c_str(), options, &c_err);
    indexstore_creation_options_dispose(options);
    if (c_err) {
      error = indexstore_error_get_description(c_err);
      indexstore_error_dispose(c_err);
    }
  }

  IndexStore(IndexStore &&other) : obj(other.obj) {
    other.obj = nullptr;
  }

  ~IndexStore() {
    indexstore_store_dispose(obj);
  }

  static IndexStoreRef create(StringRef path, clang::PathRemapper remapper,
                              std::string &error) {
    auto storeRef = std::make_shared<IndexStore>(path, remapper, error);
    if (storeRef->isInvalid())
      return nullptr;
    return storeRef;
  }

  static unsigned formatVersion() {
    return indexstore_format_version();
  }

  bool isValid() const { return obj; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  bool foreachUnit(bool sorted, llvm::function_ref<bool(StringRef unitName)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_store_units_apply(obj, sorted, ^bool(indexstore_string_ref_t unit_name) {
      return receiver(stringFromIndexStoreStringRef(unit_name));
    });
#else
    return indexstore_store_units_apply_f(obj, sorted, &receiver, functionPtrFromFunctionRef);
#endif
  }

  class UnitEvent {
    indexstore_unit_event_t obj;
  public:
    UnitEvent(indexstore_unit_event_t obj) : obj(obj) {}

    enum class Kind {
      Removed,
      Modified,
      DirectoryDeleted,
      Failure
    };
    Kind getKind() const {
      indexstore_unit_event_kind_t c_k = indexstore_unit_event_get_kind(obj);
      Kind K;
      switch (c_k) {
      case INDEXSTORE_UNIT_EVENT_REMOVED: K = Kind::Removed; break;
      case INDEXSTORE_UNIT_EVENT_MODIFIED: K = Kind::Modified; break;
      case INDEXSTORE_UNIT_EVENT_DIRECTORY_DELETED: K = Kind::DirectoryDeleted; break;
      case INDEXSTORE_UNIT_EVENT_FAILURE: K = Kind::Failure; break;
      }
      return K;
    }

    StringRef getUnitName() const {
      return stringFromIndexStoreStringRef(indexstore_unit_event_get_unit_name(obj));
    }
  };

  class UnitEventNotification {
    indexstore_unit_event_notification_t obj;
  public:
    UnitEventNotification(indexstore_unit_event_notification_t obj) : obj(obj) {}

    bool isInitial() const { return indexstore_unit_event_notification_is_initial(obj); }
    size_t getEventsCount() const { return indexstore_unit_event_notification_get_events_count(obj); }
    UnitEvent getEvent(size_t index) const { return indexstore_unit_event_notification_get_event(obj, index); }
  };

  typedef std::function<void(UnitEventNotification)> UnitEventHandler;

  void setUnitEventHandler(UnitEventHandler handler) {
#if INDEXSTORE_HAS_BLOCKS
    if (!handler) {
      indexstore_store_set_unit_event_handler(obj, nullptr);
      return;
    }

    indexstore_store_set_unit_event_handler(obj, ^(indexstore_unit_event_notification_t evt_note) {
      handler(UnitEventNotification(evt_note));
    });
#else
    if (!handler) {
      indexstore_store_set_unit_event_handler_f(obj, nullptr, nullptr, nullptr);
      return;
    }

    auto fnPtr = new UnitEventHandler(handler);
    indexstore_store_set_unit_event_handler_f(obj, fnPtr, event_handler, event_handler_finalizer);
#endif
  }

private:
  static void event_handler(void *ctx, indexstore_unit_event_notification_t evt) {
    auto fnPtr = (UnitEventHandler*)ctx;
    (*fnPtr)(evt);
  }
  static void event_handler_finalizer(void *ctx) {
    auto fnPtr = (UnitEventHandler*)ctx;
    delete fnPtr;
  }

public:
  bool startEventListening(bool waitInitialSync, std::string &error) {
    indexstore_unit_event_listen_options_t opts;
    opts.wait_initial_sync = waitInitialSync;
    indexstore_error_t c_err = nullptr;
    bool ret = indexstore_store_start_unit_event_listening(obj, &opts, sizeof(opts), &c_err);
    if (c_err) {
      error = indexstore_error_get_description(c_err);
      indexstore_error_dispose(c_err);
    }
    return ret;
  }

  void stopEventListening() {
    return indexstore_store_stop_unit_event_listening(obj);
  }

  void discardUnit(StringRef UnitName) {
    llvm::SmallString<64> buf = UnitName;
    indexstore_store_discard_unit(obj, buf.c_str());
  }

  void discardRecord(StringRef RecordName) {
    llvm::SmallString<64> buf = RecordName;
    indexstore_store_discard_record(obj, buf.c_str());
  }

  void getUnitNameFromOutputPath(StringRef outputPath, llvm::SmallVectorImpl<char> &nameBuf) {
    llvm::SmallString<256> buf = outputPath;
    llvm::SmallString<64> unitName;
    unitName.resize(64);
    size_t nameLen = indexstore_store_get_unit_name_from_output_path(obj, buf.c_str(), unitName.data(), unitName.size());
    if (nameLen+1 > unitName.size()) {
      unitName.resize(nameLen+1);
      indexstore_store_get_unit_name_from_output_path(obj, buf.c_str(), unitName.data(), unitName.size());
    }
    nameBuf.append(unitName.begin(), unitName.begin()+nameLen);
  }

  void purgeStaleData() {
    indexstore_store_purge_stale_data(obj);
  }
};

class IndexRecordReader {
  indexstore_record_reader_t obj;

public:
  IndexRecordReader(IndexStore &store, StringRef recordName, std::string &error) {
    llvm::SmallString<64> buf = recordName;
    indexstore_error_t c_err = nullptr;
    obj = indexstore_record_reader_create(store.obj, buf.c_str(), &c_err);
    if (c_err) {
      error = indexstore_error_get_description(c_err);
      indexstore_error_dispose(c_err);
    }
  }

  IndexRecordReader(IndexRecordReader &&other) : obj(other.obj) {
    other.obj = nullptr;
  }

  ~IndexRecordReader() {
    indexstore_record_reader_dispose(obj);
  }

  bool isValid() const { return obj; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  /// Goes through and passes record decls, after filtering using a \c Checker
  /// function.
  ///
  /// Resulting decls can be used as filter for \c foreachOccurrence. This
  /// allows allocating memory only for the record decls that the caller is
  /// interested in.
  bool searchSymbols(llvm::function_ref<bool(IndexRecordSymbol, bool &stop)> filter,
                     llvm::function_ref<void(IndexRecordSymbol)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_record_reader_search_symbols(obj, ^bool(indexstore_symbol_t symbol, bool *stop) {
      return filter(symbol, *stop);
    }, ^(indexstore_symbol_t symbol) {
      receiver(symbol);
    });
#else
    return indexstore_record_reader_search_symbols_f(obj, &filter, functionPtrFromFunctionRef,
                                                     &receiver, functionPtrFromFunctionRef);
#endif
  }

  bool foreachSymbol(bool noCache, llvm::function_ref<bool(IndexRecordSymbol)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_record_reader_symbols_apply(obj, noCache, ^bool(indexstore_symbol_t sym) {
      return receiver(sym);
    });
#else
    return indexstore_record_reader_symbols_apply_f(obj, noCache, &receiver, functionPtrFromFunctionRef);
#endif
  }

  /// \param DeclsFilter if non-empty indicates the list of decls that we want
  /// to get occurrences for. An empty array indicates that we want occurrences
  /// for all decls.
  /// \param RelatedDeclsFilter Same as \c DeclsFilter but for related decls.
  bool foreachOccurrence(ArrayRef<IndexRecordSymbol> symbolsFilter,
                         ArrayRef<IndexRecordSymbol> relatedSymbolsFilter,
              llvm::function_ref<bool(IndexRecordOccurrence)> receiver) {
    llvm::SmallVector<indexstore_symbol_t, 16> c_symbolsFilter;
    c_symbolsFilter.reserve(symbolsFilter.size());
    for (IndexRecordSymbol sym : symbolsFilter) {
      c_symbolsFilter.push_back(sym.obj);
    }
    llvm::SmallVector<indexstore_symbol_t, 16> c_relatedSymbolsFilter;
    c_relatedSymbolsFilter.reserve(relatedSymbolsFilter.size());
    for (IndexRecordSymbol sym : relatedSymbolsFilter) {
      c_relatedSymbolsFilter.push_back(sym.obj);
    }
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_record_reader_occurrences_of_symbols_apply(obj,
                                c_symbolsFilter.data(), c_symbolsFilter.size(),
                                c_relatedSymbolsFilter.data(),
                                c_relatedSymbolsFilter.size(),
                                ^bool(indexstore_occurrence_t occur) {
                                  return receiver(occur);
                                });
#else
    return indexstore_record_reader_occurrences_of_symbols_apply_f(obj,
                                c_symbolsFilter.data(), c_symbolsFilter.size(),
                                c_relatedSymbolsFilter.data(),
                                c_relatedSymbolsFilter.size(),
                                &receiver, functionPtrFromFunctionRef);
#endif
  }

  bool foreachOccurrence(
              llvm::function_ref<bool(IndexRecordOccurrence)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_record_reader_occurrences_apply(obj, ^bool(indexstore_occurrence_t occur) {
      return receiver(occur);
    });
#else
    return indexstore_record_reader_occurrences_apply_f(obj, &receiver, functionPtrFromFunctionRef);
#endif
  }

  bool foreachOccurrenceInLineRange(unsigned lineStart, unsigned lineEnd,
              llvm::function_ref<bool(IndexRecordOccurrence)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_record_reader_occurrences_in_line_range_apply(obj,
                                                                    lineStart,
                                                                    lineEnd,
                                          ^bool(indexstore_occurrence_t occur) {
      return receiver(occur);
    });
#else
    return indexstore_record_reader_occurrences_in_line_range_apply_f(obj,
                                                                      lineStart,
                                                                      lineEnd,
                                         &receiver, functionPtrFromFunctionRef);
#endif
  }
};

class IndexUnitDependency {
  indexstore_unit_dependency_t obj;
  friend class IndexUnitReader;

public:
  IndexUnitDependency(indexstore_unit_dependency_t obj) : obj(obj) {}

  enum class DependencyKind {
    Unit,
    Record,
    File,
  };
  DependencyKind getKind() {
    switch (indexstore_unit_dependency_get_kind(obj)) {
    case INDEXSTORE_UNIT_DEPENDENCY_UNIT: return DependencyKind::Unit;
    case INDEXSTORE_UNIT_DEPENDENCY_RECORD: return DependencyKind::Record;
    case INDEXSTORE_UNIT_DEPENDENCY_FILE: return DependencyKind::File;
    }
  }
  bool isSystem() { return indexstore_unit_dependency_is_system(obj); }
  StringRef getName() { return stringFromIndexStoreStringRef(indexstore_unit_dependency_get_name(obj)); }
  StringRef getFilePath() { return stringFromIndexStoreStringRef(indexstore_unit_dependency_get_filepath(obj)); }
  StringRef getModuleName() { return stringFromIndexStoreStringRef(indexstore_unit_dependency_get_modulename(obj)); }

};

class IndexUnitInclude {
  indexstore_unit_include_t obj;
  friend class IndexUnitReader;

public:
  IndexUnitInclude(indexstore_unit_include_t obj) : obj(obj) {}

  StringRef getSourcePath() {
    return stringFromIndexStoreStringRef(indexstore_unit_include_get_source_path(obj));
  }
  StringRef getTargetPath() {
    return stringFromIndexStoreStringRef(indexstore_unit_include_get_target_path(obj));
  }
  unsigned getSourceLine() {
    return indexstore_unit_include_get_source_line(obj);
  }
};

class IndexUnitReader {
  indexstore_unit_reader_t obj;

public:
  IndexUnitReader(IndexStore &store, StringRef unitName, std::string &error) {
    llvm::SmallString<64> buf = unitName;
    indexstore_error_t c_err = nullptr;
    obj = indexstore_unit_reader_create(store.obj, buf.c_str(), &c_err);
    if (c_err) {
      error = indexstore_error_get_description(c_err);
      indexstore_error_dispose(c_err);
    }
  }

  IndexUnitReader(IndexUnitReader &&other) : obj(other.obj) {
    other.obj = nullptr;
  }

  ~IndexUnitReader() {
    indexstore_unit_reader_dispose(obj);
  }

  bool isValid() const { return obj; }
  bool isInvalid() const { return !isValid(); }
  explicit operator bool() const { return isValid(); }

  StringRef getProviderIdentifier() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_provider_identifier(obj));
  }
  StringRef getProviderVersion() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_provider_version(obj));
  }

  timespec getModificationTime() {
    int64_t seconds, nanoseconds;
    indexstore_unit_reader_get_modification_time(obj, &seconds, &nanoseconds);
    timespec ts;
    ts.tv_sec = seconds;
    ts.tv_nsec = nanoseconds;
    return ts;
  }

  bool isSystemUnit() { return indexstore_unit_reader_is_system_unit(obj); }
  bool isModuleUnit() { return indexstore_unit_reader_is_module_unit(obj); }
  bool isDebugCompilation() { return indexstore_unit_reader_is_debug_compilation(obj); }
  bool hasMainFile() { return indexstore_unit_reader_has_main_file(obj); }

  StringRef getMainFilePath() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_main_file(obj));
  }
  StringRef getModuleName() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_module_name(obj));
  }
  StringRef getWorkingDirectory() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_working_dir(obj));
  }
  StringRef getOutputFile() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_output_file(obj));
  }
  StringRef getSysrootPath() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_sysroot_path(obj));
  }
  StringRef getTarget() {
    return stringFromIndexStoreStringRef(indexstore_unit_reader_get_target(obj));
  }

  bool foreachDependency(llvm::function_ref<bool(IndexUnitDependency)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_unit_reader_dependencies_apply(obj, ^bool(indexstore_unit_dependency_t dep) {
      return receiver(dep);
    });
#else
    return indexstore_unit_reader_dependencies_apply_f(obj, &receiver, functionPtrFromFunctionRef);
#endif
  }

  bool foreachInclude(llvm::function_ref<bool(IndexUnitInclude)> receiver) {
#if INDEXSTORE_HAS_BLOCKS
    return indexstore_unit_reader_includes_apply(obj, ^bool(indexstore_unit_include_t inc) {
      return receiver(inc);
    });
#else
    return indexstore_unit_reader_includes_apply_f(obj, &receiver, functionPtrFromFunctionRef);
#endif
  }
};

} // namespace indexstore

#endif
