//===- IndexStore.cpp - Index store API -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the API for the index store.
//
//===----------------------------------------------------------------------===//

#include "indexstore/indexstore.h"
#include "clang/Index/IndexDataStore.h"
#include "clang/Index/IndexDataStoreSymbolUtils.h"
#include "clang/Index/IndexRecordReader.h"
#include "clang/Index/IndexUnitReader.h"
#include "clang/Index/IndexUnitWriter.h"
#include "clang/DirectoryWatcher/DirectoryWatcher.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ManagedStatic.h"

#if INDEXSTORE_HAS_BLOCKS
#include <Block.h>
#endif

using namespace clang;
using namespace clang::index;
using namespace llvm;

static indexstore_string_ref_t toIndexStoreString(StringRef str) {
  return indexstore_string_ref_t{ str.data(), str.size() };
}

static timespec toTimeSpec(sys::TimePoint<> tp) {
  std::chrono::seconds sec = std::chrono::time_point_cast<std::chrono::seconds>(
                 tp).time_since_epoch();
  std::chrono::nanoseconds nsec =
    std::chrono::time_point_cast<std::chrono::nanoseconds>(tp - sec)
      .time_since_epoch();
  timespec ts;
  ts.tv_sec = sec.count();
  ts.tv_nsec = nsec.count();
  return ts;
}

//===----------------------------------------------------------------------===//
// Fatal error handling
//===----------------------------------------------------------------------===//

static void fatal_error_handler(void *user_data, const std::string& reason,
                                bool gen_crash_diag) {
  // Write the result out to stderr avoiding errs() because raw_ostreams can
  // call report_fatal_error.
  fprintf(stderr, "INDEXSTORE FATAL ERROR: %s\n", reason.c_str());
  ::abort();
}

namespace {
struct RegisterFatalErrorHandler {
  RegisterFatalErrorHandler() {
    llvm::install_fatal_error_handler(fatal_error_handler, nullptr);
  }
};
}

static llvm::ManagedStatic<RegisterFatalErrorHandler> RegisterFatalErrorHandlerOnce;

//===----------------------------------------------------------------------===//
// C API
//===----------------------------------------------------------------------===//

namespace {

struct IndexStoreError {
  std::string Error;
};

} // anonymous namespace

const char *
indexstore_error_get_description(indexstore_error_t err) {
  return static_cast<IndexStoreError*>(err)->Error.c_str();
}

void
indexstore_error_dispose(indexstore_error_t err) {
  delete static_cast<IndexStoreError*>(err);
}

unsigned
indexstore_format_version(void) {
  return IndexDataStore::getFormatVersion();
}

indexstore_t
indexstore_store_create(const char *store_path, indexstore_error_t *c_error) {
  // Look through the managed static to trigger construction of the managed
  // static which registers our fatal error handler. This ensures it is only
  // registered once.
  (void)*RegisterFatalErrorHandlerOnce;

  std::unique_ptr<IndexDataStore> store;
  std::string error;
  store = IndexDataStore::create(store_path, error);
  if (!store) {
    if (c_error)
      *c_error = new IndexStoreError{ error };
    return nullptr;
  }
  return store.release();
}

void
indexstore_store_dispose(indexstore_t store) {
  delete static_cast<IndexDataStore*>(store);
}

#if INDEXSTORE_HAS_BLOCKS
bool
indexstore_store_units_apply(indexstore_t c_store, unsigned sorted,
                            INDEXSTORE_NOESCAPE bool(^applier)(indexstore_string_ref_t unit_name)) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  return store->foreachUnitName(sorted, [&](StringRef unitName) -> bool {
    return applier(toIndexStoreString(unitName));
  });
}
#endif

bool
indexstore_store_units_apply_f(indexstore_t c_store, unsigned sorted,
                               void *context,
             INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_string_ref_t unit_name)) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  return store->foreachUnitName(sorted, [&](StringRef unitName) -> bool {
    return applier(context, toIndexStoreString(unitName));
  });
}

size_t
indexstore_unit_event_notification_get_events_count(indexstore_unit_event_notification_t c_evtnote) {
  auto *evtnote = static_cast<IndexDataStore::UnitEventNotification*>(c_evtnote);
  return evtnote->Events.size();
}

indexstore_unit_event_t
indexstore_unit_event_notification_get_event(indexstore_unit_event_notification_t c_evtnote, size_t index) {
  auto *evtnote = static_cast<IndexDataStore::UnitEventNotification*>(c_evtnote);
  return (indexstore_unit_event_t)&evtnote->Events[index];
}

bool
indexstore_unit_event_notification_is_initial(indexstore_unit_event_notification_t c_evtnote) {
  auto *evtnote = static_cast<IndexDataStore::UnitEventNotification*>(c_evtnote);
  return evtnote->IsInitial;
}

indexstore_unit_event_kind_t
indexstore_unit_event_get_kind(indexstore_unit_event_t c_evt) {
  auto *evt = static_cast<IndexDataStore::UnitEvent*>(c_evt);
  indexstore_unit_event_kind_t k;
  switch (evt->Kind) {
    case IndexDataStore::UnitEventKind::Added:
      k = INDEXSTORE_UNIT_EVENT_ADDED; break;
    case IndexDataStore::UnitEventKind::Removed:
      k = INDEXSTORE_UNIT_EVENT_REMOVED; break;
    case IndexDataStore::UnitEventKind::Modified:
      k = INDEXSTORE_UNIT_EVENT_MODIFIED; break;
    case IndexDataStore::UnitEventKind::DirectoryDeleted:
      k = INDEXSTORE_UNIT_EVENT_DIRECTORY_DELETED; break;
  }
  return k;
}

indexstore_string_ref_t
indexstore_unit_event_get_unit_name(indexstore_unit_event_t c_evt) {
  auto *evt = static_cast<IndexDataStore::UnitEvent*>(c_evt);
  return toIndexStoreString(evt->UnitName);
}

timespec
indexstore_unit_event_get_modification_time(indexstore_unit_event_t c_evt) {
  auto *evt = static_cast<IndexDataStore::UnitEvent*>(c_evt);
  return toTimeSpec(evt->ModTime);
}

#if INDEXSTORE_HAS_BLOCKS
void
indexstore_store_set_unit_event_handler(indexstore_t c_store,
                                    indexstore_unit_event_handler_t blk_handler) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  if (!blk_handler) {
    store->setUnitEventHandler(nullptr);
    return;
  }

  class BlockWrapper {
    indexstore_unit_event_handler_t blk_handler;
  public:
    BlockWrapper(indexstore_unit_event_handler_t handler) {
      blk_handler = Block_copy(handler);
    }
    BlockWrapper(const BlockWrapper &other) {
      blk_handler = Block_copy(other.blk_handler);
    }
    ~BlockWrapper() {
      Block_release(blk_handler);
    }

    void operator()(indexstore_unit_event_notification_t evt_note) const {
      blk_handler(evt_note);
    }
  };

  BlockWrapper handler(blk_handler);

  store->setUnitEventHandler([handler](IndexDataStore::UnitEventNotification evtNote) {
    handler(&evtNote);
  });
}
#endif

void
indexstore_store_set_unit_event_handler_f(indexstore_t c_store, void *context,
         void(*fn_handler)(void *context, indexstore_unit_event_notification_t),
                                          void(*finalizer)(void *context)) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  if (!fn_handler) {
    store->setUnitEventHandler(nullptr);
    return;
  }

  class BlockWrapper {
    void *context;
    void(*fn_handler)(void *context, indexstore_unit_event_notification_t);
    void(*finalizer)(void *context);

  public:
    BlockWrapper(void *_context,
                 void(*_fn_handler)(void *context, indexstore_unit_event_notification_t),
                 void(*_finalizer)(void *context))
    : context(_context), fn_handler(_fn_handler), finalizer(_finalizer) {}

    ~BlockWrapper() {
      if (finalizer) {
        finalizer(context);
      }
    }

    void operator()(indexstore_unit_event_notification_t evt_note) const {
      fn_handler(context, evt_note);
    }
  };

  auto handler = std::make_shared<BlockWrapper>(context, fn_handler, finalizer);

  store->setUnitEventHandler([handler](IndexDataStore::UnitEventNotification evtNote) {
    (*handler)(&evtNote);
  });
}

bool
indexstore_store_start_unit_event_listening(indexstore_t c_store,
                                            indexstore_unit_event_listen_options_t *client_opts,
                                            size_t listen_options_struct_size,
                                            indexstore_error_t *c_error) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  indexstore_unit_event_listen_options_t listen_opts;
  memset(&listen_opts, 0, sizeof(listen_opts));
  unsigned clientOptSize = listen_options_struct_size < sizeof(listen_opts)
                             ? listen_options_struct_size : sizeof(listen_opts);
  memcpy(&listen_opts, client_opts, clientOptSize);

  std::string error;
  bool err = store->startEventListening(listen_opts.wait_initial_sync, error);
  if (err && c_error)
    *c_error = new IndexStoreError{ error };
  return err;
}

void
indexstore_store_stop_unit_event_listening(indexstore_t c_store) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  store->stopEventListening();
}

void
indexstore_store_discard_unit(indexstore_t c_store, const char *unit_name) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  store->discardUnit(unit_name);
}

void
indexstore_store_discard_record(indexstore_t c_store, const char *record_name) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  store->discardRecord(record_name);
}

void
indexstore_store_purge_stale_data(indexstore_t c_store) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  store->purgeStaleData();
}

indexstore_symbol_kind_t
indexstore_symbol_get_kind(indexstore_symbol_t sym) {
  return getIndexStoreKind(static_cast<IndexRecordDecl *>(sym)->SymInfo.Kind);
}

indexstore_symbol_subkind_t
indexstore_symbol_get_subkind(indexstore_symbol_t sym) {
  return getIndexStoreSubKind(static_cast<IndexRecordDecl *>(sym)->SymInfo.SubKind);
}

indexstore_symbol_language_t
indexstore_symbol_get_language(indexstore_symbol_t sym) {
  return getIndexStoreLang(static_cast<IndexRecordDecl *>(sym)->SymInfo.Lang);
}

uint64_t
indexstore_symbol_get_properties(indexstore_symbol_t sym) {
  return getIndexStoreProperties(static_cast<IndexRecordDecl *>(sym)->SymInfo.Properties);
}

uint64_t
indexstore_symbol_get_roles(indexstore_symbol_t sym) {
  return getIndexStoreRoles(static_cast<IndexRecordDecl *>(sym)->Roles);
}

uint64_t
indexstore_symbol_get_related_roles(indexstore_symbol_t sym) {
  return getIndexStoreRoles(static_cast<IndexRecordDecl *>(sym)->RelatedRoles);
}

indexstore_string_ref_t
indexstore_symbol_get_name(indexstore_symbol_t sym) {
  auto *D = static_cast<IndexRecordDecl*>(sym);
  return toIndexStoreString(D->Name);
}

indexstore_string_ref_t
indexstore_symbol_get_usr(indexstore_symbol_t sym) {
  auto *D = static_cast<IndexRecordDecl*>(sym);
  return toIndexStoreString(D->USR);
}

indexstore_string_ref_t
indexstore_symbol_get_codegen_name(indexstore_symbol_t sym) {
  auto *D = static_cast<IndexRecordDecl*>(sym);
  return toIndexStoreString(D->CodeGenName);
}

uint64_t
indexstore_symbol_relation_get_roles(indexstore_symbol_relation_t sym_rel) {
  return getIndexStoreRoles(static_cast<IndexRecordRelation *>(sym_rel)->Roles);
}

indexstore_symbol_t
indexstore_symbol_relation_get_symbol(indexstore_symbol_relation_t sym_rel) {
  return (indexstore_symbol_t)static_cast<IndexRecordRelation*>(sym_rel)->Dcl;
}

indexstore_symbol_t
indexstore_occurrence_get_symbol(indexstore_occurrence_t occur) {
  return (indexstore_symbol_t)static_cast<IndexRecordOccurrence*>(occur)->Dcl;
}

#if INDEXSTORE_HAS_BLOCKS
bool
indexstore_occurrence_relations_apply(indexstore_occurrence_t occur,
                      INDEXSTORE_NOESCAPE bool(^applier)(indexstore_symbol_relation_t symbol_rel)) {
  auto *recOccur = static_cast<IndexRecordOccurrence*>(occur);
  for (auto &rel : recOccur->Relations) {
    if (!applier(&rel))
      return false;
  }
  return true;
}
#endif

bool
indexstore_occurrence_relations_apply_f(indexstore_occurrence_t occur,
                                        void *context,
       INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_symbol_relation_t symbol_rel)) {
  auto *recOccur = static_cast<IndexRecordOccurrence*>(occur);
  for (auto &rel : recOccur->Relations) {
    if (!applier(context, &rel))
      return false;
  }
  return true;
}

uint64_t
indexstore_occurrence_get_roles(indexstore_occurrence_t occur) {
  return getIndexStoreRoles(static_cast<IndexRecordOccurrence*>(occur)->Roles);
}

void
indexstore_occurrence_get_line_col(indexstore_occurrence_t occur,
                              unsigned *line, unsigned *column) {
  auto *recOccur = static_cast<IndexRecordOccurrence*>(occur);
  if (line)
    *line = recOccur->Line;
  if (column)
    *column = recOccur->Column;
}

typedef void *indexstore_record_reader_t;

indexstore_record_reader_t
indexstore_record_reader_create(indexstore_t c_store, const char *record_name,
                                indexstore_error_t *c_error) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  std::unique_ptr<IndexRecordReader> reader;
  std::string error;
  reader = IndexRecordReader::createWithRecordFilename(record_name,
                                                       store->getFilePath(),
                                                       error);
  if (!reader) {
    if (c_error)
      *c_error = new IndexStoreError{ error };
    return nullptr;
  }
  return reader.release();
}

void
indexstore_record_reader_dispose(indexstore_record_reader_t rdr) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  delete reader;
}

#if INDEXSTORE_HAS_BLOCKS
/// Goes through the symbol data and passes symbols to \c receiver, for the
/// symbol data that \c filter returns true on.
///
/// This allows allocating memory only for the record symbols that the caller is
/// interested in.
bool
indexstore_record_reader_search_symbols(indexstore_record_reader_t rdr,
    INDEXSTORE_NOESCAPE bool(^filter)(indexstore_symbol_t symbol, bool *stop),
    INDEXSTORE_NOESCAPE void(^receiver)(indexstore_symbol_t symbol)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);

  auto filterFn = [&](const IndexRecordDecl &D) -> IndexRecordReader::DeclSearchReturn {
    bool stop = false;
    bool accept = filter((indexstore_symbol_t)&D, &stop);
    return { accept, !stop };
  };
  auto receiverFn = [&](const IndexRecordDecl *D) {
    receiver((indexstore_symbol_t)D);
  };

  return reader->searchDecls(filterFn, receiverFn);
}

bool
indexstore_record_reader_symbols_apply(indexstore_record_reader_t rdr,
                                        bool nocache,
                                   INDEXSTORE_NOESCAPE bool(^applier)(indexstore_symbol_t symbol)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordDecl *D) -> bool {
    return applier((indexstore_symbol_t)D);
  };
  return reader->foreachDecl(nocache, receiverFn);
}

bool
indexstore_record_reader_occurrences_apply(indexstore_record_reader_t rdr,
                                INDEXSTORE_NOESCAPE bool(^applier)(indexstore_occurrence_t occur)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordOccurrence &RO) -> bool {
    return applier((indexstore_occurrence_t)&RO);
  };
  return reader->foreachOccurrence(receiverFn);
}

bool
indexstore_record_reader_occurrences_in_line_range_apply(indexstore_record_reader_t rdr,
                                                         unsigned line_start,
                                                         unsigned line_count,
                                INDEXSTORE_NOESCAPE bool(^applier)(indexstore_occurrence_t occur)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordOccurrence &RO) -> bool {
    return applier((indexstore_occurrence_t)&RO);
  };
  return reader->foreachOccurrenceInLineRange(line_start, line_count, receiverFn);
}

/// \param symbols if non-zero \c symbols_count, indicates the list of symbols
/// that we want to get occurrences for. An empty array indicates that we want
/// occurrences for all symbols.
/// \param related_symbols Same as \c symbols but for related symbols.
bool
indexstore_record_reader_occurrences_of_symbols_apply(indexstore_record_reader_t rdr,
        indexstore_symbol_t *symbols, size_t symbols_count,
        indexstore_symbol_t *related_symbols, size_t related_symbols_count,
        INDEXSTORE_NOESCAPE bool(^applier)(indexstore_occurrence_t occur)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordOccurrence &RO) -> bool {
    return applier((indexstore_occurrence_t)&RO);
  };
  return reader->foreachOccurrence({(IndexRecordDecl**)symbols, symbols_count},
                                   {(IndexRecordDecl**)related_symbols, related_symbols_count},
                                   receiverFn);
}
#endif

bool
indexstore_record_reader_search_symbols_f(indexstore_record_reader_t rdr,
                                          void *filter_ctx,
    INDEXSTORE_NOESCAPE bool(*filter)(void *filter_ctx, indexstore_symbol_t symbol, bool *stop),
                                          void *receiver_ctx,
              INDEXSTORE_NOESCAPE void(*receiver)(void *receiver_ctx, indexstore_symbol_t symbol)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);

  auto filterFn = [&](const IndexRecordDecl &D) -> IndexRecordReader::DeclSearchReturn {
    bool stop = false;
    bool accept = filter(filter_ctx, (indexstore_symbol_t)&D, &stop);
    return { accept, !stop };
  };
  auto receiverFn = [&](const IndexRecordDecl *D) {
    receiver(receiver_ctx, (indexstore_symbol_t)D);
  };

  return reader->searchDecls(filterFn, receiverFn);
}

bool
indexstore_record_reader_symbols_apply_f(indexstore_record_reader_t rdr,
                                         bool nocache,
                                         void *context,
                    INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_symbol_t symbol)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordDecl *D) -> bool {
    return applier(context, (indexstore_symbol_t)D);
  };
  return reader->foreachDecl(nocache, receiverFn);
}

bool
indexstore_record_reader_occurrences_apply_f(indexstore_record_reader_t rdr,
                                             void *context,
                 INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_occurrence_t occur)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordOccurrence &RO) -> bool {
    return applier(context, (indexstore_occurrence_t)&RO);
  };
  return reader->foreachOccurrence(receiverFn);
}

bool
indexstore_record_reader_occurrences_in_line_range_apply_f(indexstore_record_reader_t rdr,
                                                           unsigned line_start,
                                                           unsigned line_count,
                                                           void *context,
                 INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_occurrence_t occur)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordOccurrence &RO) -> bool {
    return applier(context, (indexstore_occurrence_t)&RO);
  };
  return reader->foreachOccurrenceInLineRange(line_start, line_count, receiverFn);
}

bool
indexstore_record_reader_occurrences_of_symbols_apply_f(indexstore_record_reader_t rdr,
        indexstore_symbol_t *symbols, size_t symbols_count,
        indexstore_symbol_t *related_symbols, size_t related_symbols_count,
        void *context,
        INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_occurrence_t occur)) {
  auto *reader = static_cast<IndexRecordReader *>(rdr);
  auto receiverFn = [&](const IndexRecordOccurrence &RO) -> bool {
    return applier(context, (indexstore_occurrence_t)&RO);
  };
  return reader->foreachOccurrence({(IndexRecordDecl**)symbols, symbols_count},
                                   {(IndexRecordDecl**)related_symbols, related_symbols_count},
                                   receiverFn);
}

size_t
indexstore_store_get_unit_name_from_output_path(indexstore_t store,
                                                const char *output_path,
                                                char *name_buf,
                                                size_t buf_size) {
  SmallString<256> unitName;
  IndexUnitWriter::getUnitNameForAbsoluteOutputFile(output_path, unitName);
  size_t nameLen = unitName.size();
  if (buf_size != 0) {
    strncpy(name_buf, unitName.c_str(), buf_size-1);
    name_buf[buf_size-1] = '\0';
  }
  return nameLen;
}

bool
indexstore_store_get_unit_modification_time(indexstore_t c_store,
                                            const char *unit_name,
                                            int64_t *seconds,
                                            int64_t *nanoseconds,
                                            indexstore_error_t *c_error) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  std::string error;
  // FIXME: This provides mod time with second-only accuracy.
  auto optModTime = IndexUnitReader::getModificationTimeForUnit(unit_name,
                                              store->getFilePath(), error);
  if (!optModTime) {
    if (c_error)
      *c_error = new IndexStoreError{ error };
    return true;
  }

  timespec ts = toTimeSpec(*optModTime);
  if (seconds)
    *seconds = ts.tv_sec;
  if (nanoseconds)
    *nanoseconds = ts.tv_nsec;

  return false;
}

indexstore_unit_reader_t
indexstore_unit_reader_create(indexstore_t c_store, const char *unit_name,
                              indexstore_error_t *c_error) {
  IndexDataStore *store = static_cast<IndexDataStore*>(c_store);
  std::unique_ptr<IndexUnitReader> reader;
  std::string error;
  reader = IndexUnitReader::createWithUnitFilename(unit_name,
                                                   store->getFilePath(), error);
  if (!reader) {
    if (c_error)
      *c_error = new IndexStoreError{ error };
    return nullptr;
  }
  return reader.release();
}

void
indexstore_unit_reader_dispose(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  delete reader;
}

indexstore_string_ref_t
indexstore_unit_reader_get_provider_identifier(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getProviderIdentifier());
}

indexstore_string_ref_t
indexstore_unit_reader_get_provider_version(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getProviderVersion());
}

void
indexstore_unit_reader_get_modification_time(indexstore_unit_reader_t rdr,
                                             int64_t *seconds,
                                             int64_t *nanoseconds) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  // FIXME: This provides mod time with second-only accuracy.
  sys::TimePoint<> timeVal = reader->getModificationTime();
  timespec ts = toTimeSpec(timeVal);
  if (seconds)
    *seconds = ts.tv_sec;
  if (nanoseconds)
    *nanoseconds = ts.tv_nsec;
}

bool
indexstore_unit_reader_is_system_unit(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->isSystemUnit();
}

bool
indexstore_unit_reader_is_module_unit(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->isModuleUnit();
}

bool
indexstore_unit_reader_is_debug_compilation(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->isDebugCompilation();
}

bool
indexstore_unit_reader_has_main_file(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->hasMainFile();
}

indexstore_string_ref_t
indexstore_unit_reader_get_main_file(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getMainFilePath());
}

indexstore_string_ref_t
indexstore_unit_reader_get_module_name(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getModuleName());
}

indexstore_string_ref_t
indexstore_unit_reader_get_working_dir(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getWorkingDirectory());
}

indexstore_string_ref_t
indexstore_unit_reader_get_output_file(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getOutputFile());
}

indexstore_string_ref_t
indexstore_unit_reader_get_sysroot_path(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getSysrootPath());
}

indexstore_string_ref_t
indexstore_unit_reader_get_target(indexstore_unit_reader_t rdr) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return toIndexStoreString(reader->getTarget());
}

indexstore_unit_dependency_kind_t
indexstore_unit_dependency_get_kind(indexstore_unit_dependency_t c_dep) {
  auto dep = static_cast<const IndexUnitReader::DependencyInfo*>(c_dep);
  switch (dep->Kind) {
  case IndexUnitReader::DependencyKind::Unit: return INDEXSTORE_UNIT_DEPENDENCY_UNIT;
  case IndexUnitReader::DependencyKind::Record: return INDEXSTORE_UNIT_DEPENDENCY_RECORD;
  case IndexUnitReader::DependencyKind::File: return INDEXSTORE_UNIT_DEPENDENCY_FILE;
  }
}

bool
indexstore_unit_dependency_is_system(indexstore_unit_dependency_t c_dep) {
  auto dep = static_cast<const IndexUnitReader::DependencyInfo*>(c_dep);
  return dep->IsSystem;
}

indexstore_string_ref_t
indexstore_unit_dependency_get_filepath(indexstore_unit_dependency_t c_dep) {
  auto dep = static_cast<const IndexUnitReader::DependencyInfo*>(c_dep);
  return toIndexStoreString(dep->FilePath);
}

indexstore_string_ref_t
indexstore_unit_dependency_get_modulename(indexstore_unit_dependency_t c_dep) {
  auto dep = static_cast<const IndexUnitReader::DependencyInfo*>(c_dep);
  return toIndexStoreString(dep->ModuleName);
}

indexstore_string_ref_t
indexstore_unit_dependency_get_name(indexstore_unit_dependency_t c_dep) {
  auto dep = static_cast<const IndexUnitReader::DependencyInfo*>(c_dep);
  return toIndexStoreString(dep->UnitOrRecordName);
}

indexstore_string_ref_t
indexstore_unit_include_get_source_path(indexstore_unit_include_t c_inc) {
  auto inc = static_cast<const IndexUnitReader::IncludeInfo*>(c_inc);
  return toIndexStoreString(inc->SourcePath);
}

indexstore_string_ref_t
indexstore_unit_include_get_target_path(indexstore_unit_include_t c_inc) {
  auto inc = static_cast<const IndexUnitReader::IncludeInfo*>(c_inc);
  return toIndexStoreString(inc->TargetPath);
}

unsigned
indexstore_unit_include_get_source_line(indexstore_unit_include_t c_inc) {
  auto inc = static_cast<const IndexUnitReader::IncludeInfo*>(c_inc);
  return inc->SourceLine;
}

#if INDEXSTORE_HAS_BLOCKS
bool
indexstore_unit_reader_dependencies_apply(indexstore_unit_reader_t rdr,
                             INDEXSTORE_NOESCAPE bool(^applier)(indexstore_unit_dependency_t)) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->foreachDependency([&](const IndexUnitReader::DependencyInfo &depInfo) -> bool {
    return applier((void*)&depInfo);
  });
}

bool
indexstore_unit_reader_includes_apply(indexstore_unit_reader_t rdr,
                             INDEXSTORE_NOESCAPE bool(^applier)(indexstore_unit_include_t)) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->foreachInclude([&](const IndexUnitReader::IncludeInfo &incInfo) -> bool {
    return applier((void*)&incInfo);
  });
}
#endif

bool
indexstore_unit_reader_dependencies_apply_f(indexstore_unit_reader_t rdr,
                                            void *context,
                  INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_unit_dependency_t)) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->foreachDependency([&](const IndexUnitReader::DependencyInfo &depInfo) -> bool {
    return applier(context, (void*)&depInfo);
  });
}

bool
indexstore_unit_reader_includes_apply_f(indexstore_unit_reader_t rdr,
                                        void *context,
                     INDEXSTORE_NOESCAPE bool(*applier)(void *context, indexstore_unit_include_t)) {
  auto reader = static_cast<IndexUnitReader*>(rdr);
  return reader->foreachInclude([&](const IndexUnitReader::IncludeInfo &incInfo) -> bool {
    return applier(context, (void*)&incInfo);
  });
}
