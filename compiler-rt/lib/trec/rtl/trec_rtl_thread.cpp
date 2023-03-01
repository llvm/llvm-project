//===-- trec_rtl_thread.cpp
//-----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <sys/fcntl.h>

#include "sanitizer_common/sanitizer_placement_new.h"
#include "trec_mman.h"
#include "trec_platform.h"
#include "trec_rtl.h"

namespace __trec {

// ThreadContext implementation.

ThreadContext::ThreadContext(int tid)
    : ThreadContextBase(tid),
      thr(),
      trace_buffer(nullptr),
      metadata_buffer(nullptr),
      debug_buffer(nullptr),
      trace_buffer_size(0),
      metadata_buffer_size(0),
      debug_buffer_size(0),
      event_cnt(0),
      metadata_offset(0),
      debug_offset(0),
      prev_read_pc(0),
      header(tid),
      isFuncEnterMetaVaild(false),
      dbg_temp_buffer_size(0),
      isFuncExitMetaVaild(false) {}

#if !SANITIZER_GO
ThreadContext::~ThreadContext() {}
#endif

void ThreadContext::OnDead() {}

void ThreadContext::OnJoined(void *arg) {}

struct OnCreatedArgs {
  ThreadState *thr;
  uptr pc;
};

void ThreadContext::OnCreated(void *arg) {}

void ThreadContext::OnReset() {}

void ThreadContext::OnDetached(void *arg) {}

bool ThreadContext::state_restore() {
  struct stat _st = {0};
  char path[2 * TREC_DIR_PATH_LEN];
  internal_snprintf(path, 2 * TREC_DIR_PATH_LEN - 1,
                    "%s/trec_%llu/header/%u.bin", ctx->trace_dir,
                    internal_getpid(), tid);
  uptr IS_EXIST = __sanitizer::internal_stat(path, &_st);
  if (IS_EXIST == 0 && _st.st_size > 0) {
    int header_fd = internal_open(path, O_RDONLY);
    if (header_fd < 0) {
      Report("Restore header from %s failed\n", path);
      return false;
    } else {
      internal_read(header_fd, &header, sizeof(header));
      event_cnt = header.state[__trec_header::RecordType::TotalEventCnt];
      metadata_offset =
          header.state[__trec_header::RecordType::MetadataFileLen];
      debug_offset = header.state[__trec_header::RecordType::DebugFileLen];
      header.state[__trec_header::RecordType::ProcessFork] = 0;
      header.state[__trec_header::RecordType::Tid] = tid;
      return true;
    }
  }
  return false;
}

void ThreadContext::flush_trace() {
  char filepath[TREC_DIR_PATH_LEN];
  struct stat _st = {0};
  internal_snprintf(filepath, 2 * TREC_DIR_PATH_LEN - 1, "%s/trec_%llu/trace",
                    ctx->trace_dir, internal_getpid(), tid);
  uptr IS_EXIST = __sanitizer::internal_stat(filepath, &_st);
  if (IS_EXIST != 0) {
    ctx->ppid = ctx->pid;
    ctx->pid = internal_getpid();
    ctx->open_directory(ctx->trace_dir);
  }

  internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/trec_%d/trace/%d.bin",
                    ctx->trace_dir, internal_getpid(), this->tid);
  int fd_trace = internal_open(filepath, O_CREAT | O_WRONLY | O_APPEND, 0700);

  if (UNLIKELY(fd_trace < 0)) {
    Report("Failed to open trace file at %s\n", filepath);
    Die();
  } else if (trace_buffer != nullptr && trace_buffer_size > 0) {
    char *buff_pos = (char *)trace_buffer;
    while (trace_buffer_size > 0) {
      uptr write_bytes = internal_write(fd_trace, buff_pos, trace_buffer_size);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to flush metadata info in %s, errno=%u\n", filepath,
               errno);
        Die();
      } else {
        trace_buffer_size -= write_bytes;
        buff_pos += write_bytes;
      }
    }
  }
  internal_close(fd_trace);
  header.state[__trec_header::RecordType::TotalEventCnt] = event_cnt;
}

void ThreadContext::flush_metadata() {
  char filepath[TREC_DIR_PATH_LEN];

  struct stat _st = {0};
  internal_snprintf(filepath, 2 * TREC_DIR_PATH_LEN - 1,
                    "%s/trec_%llu/metadata", ctx->trace_dir, internal_getpid(),
                    tid);
  uptr IS_EXIST = __sanitizer::internal_stat(filepath, &_st);
  if (IS_EXIST != 0) {
    ctx->ppid = ctx->pid;
    ctx->pid = internal_getpid();
    ctx->open_directory(ctx->trace_dir);
  }

  internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1,
                    "%s/trec_%d/metadata/%d.bin", ctx->trace_dir,
                    internal_getpid(), this->tid);
  int fd_metadata =
      internal_open(filepath, O_CREAT | O_WRONLY | O_APPEND, 0700);

  if (UNLIKELY(fd_metadata < 0)) {
    Report("Failed to open metadata file at %s\n", filepath);
    Die();
  } else if (metadata_buffer != nullptr && metadata_buffer_size > 0) {
    char *buff_pos = (char *)metadata_buffer;
    while (metadata_buffer_size > 0) {
      uptr write_bytes =
          internal_write(fd_metadata, buff_pos, metadata_buffer_size);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to flush metadata info in %s, errno=%u\n", filepath,
               errno);
        Die();
      } else {
        metadata_buffer_size -= write_bytes;
        buff_pos += write_bytes;
        header.state[__trec_header::RecordType::MetadataFileLen] += write_bytes;
      }
    }
  }
  internal_close(fd_metadata);
  if (ctx->flags.output_debug)
    flush_debug_info();
}

void ThreadContext::flush_debug_info() {
  if (debug_buffer_size == 0)
    return;
  char filepath[TREC_DIR_PATH_LEN];

  internal_snprintf(filepath, 2 * TREC_DIR_PATH_LEN - 1, "%s/trec_%llu/debug",
                    ctx->trace_dir, internal_getpid(), tid);
  struct stat _st = {0};
  uptr IS_EXIST = __sanitizer::internal_stat(filepath, &_st);
  if (IS_EXIST != 0) {
    ctx->ppid = ctx->pid;
    ctx->pid = internal_getpid();
    ctx->open_directory(ctx->trace_dir);
  }

  internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/trec_%d/debug/%d.bin",
                    ctx->trace_dir, internal_getpid(), thr->tid);
  int fd_debug = internal_open(filepath, O_CREAT | O_WRONLY | O_APPEND, 0700);

  if (UNLIKELY(fd_debug < 0)) {
    Report("Failed to open debug info file at %s\n", filepath);
    Die();
  } else if (debug_buffer != nullptr && debug_buffer_size > 0) {
    char *buff_pos = (char *)debug_buffer;
    while (debug_buffer_size > 0) {
      uptr write_bytes = internal_write(fd_debug, buff_pos, debug_buffer_size);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to flush debug info in %s, errno=%u\n", filepath, errno);
        Die();
      } else {
        debug_buffer_size -= write_bytes;
        buff_pos += write_bytes;
        header.state[__trec_header::RecordType::DebugFileLen] += write_bytes;
      }
    }
  }
  internal_close(fd_debug);
}

static inline bool CompareBaseAddress(const LoadedModule &a,
                                      const LoadedModule &b) {
  return a.base_address() < b.base_address();
}
void ThreadContext::flush_module() {
  char modulepath[TREC_DIR_PATH_LEN];
  char write_buff[2 * TREC_DIR_PATH_LEN];
  internal_snprintf(modulepath, TREC_DIR_PATH_LEN - 1,
                    "%s/trec_%d/header/modules_%d.txt", ctx->trace_dir,
                    internal_getpid(), thr->tid);
  int fd_module_file =
      internal_open(modulepath, O_CREAT | O_WRONLY | O_TRUNC, 0700);
  MemoryMappingLayout memory_mapping(false);
  InternalMmapVector<LoadedModule> modules(/*initial_capacity*/ 64);
  memory_mapping.DumpListOfModules(&modules);
  Sort(modules.begin(), modules.size(), CompareBaseAddress);
  uptr idx = 0;
  for (auto &item : modules) {
    if (item.full_name() && item.base_address() && item.max_address() &&
        internal_strstr(item.full_name(), "(deleted)") == nullptr) {
      internal_memset(write_buff, 0, sizeof(write_buff));
      int bufflen = internal_snprintf(write_buff, 2 * TREC_DIR_PATH_LEN - 1,
                                      "%s %p-%p\n", item.full_name(),
                                      item.base_address(), item.max_address());
      uptr need_write_bytes = bufflen;
      char *buff_pos = (char *)write_buff;
      while (need_write_bytes > 0) {
        uptr write_bytes =
            internal_write(fd_module_file, buff_pos, need_write_bytes);
        if (write_bytes == -1 && errno != EINTR) {
          Report("Failed to flush module info in %s, errno=%u\n", modulepath,
                 errno);
          Die();
        } else {
          need_write_bytes -= write_bytes;
          buff_pos += write_bytes;
        }
      }
    }
  }
  internal_close(fd_module_file);
}

void ThreadContext::flush_header() {
  char filepath[TREC_DIR_PATH_LEN];

  struct stat _st = {0};
  internal_snprintf(filepath, 2 * TREC_DIR_PATH_LEN - 1, "%s/trec_%llu/header",
                    ctx->trace_dir, internal_getpid(), tid);
  uptr IS_EXIST = __sanitizer::internal_stat(filepath, &_st);
  if (IS_EXIST != 0) {
    ctx->ppid = ctx->pid;
    ctx->pid = internal_getpid();
    ctx->open_directory(ctx->trace_dir);
  }

  internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/trec_%d/header/%d.bin",
                    ctx->trace_dir, internal_getpid(), thr->tid);

  int fd_header = internal_open(filepath, O_CREAT | O_WRONLY | O_TRUNC, 0700);

  if (UNLIKELY(fd_header < 0)) {
    Report("Failed to open header file\n");
    Die();
  } else {
    uptr need_write_bytes = sizeof(header);
    char *buff_pos = (char *)&header;
    while (need_write_bytes > 0) {
      uptr write_bytes = internal_write(fd_header, buff_pos, need_write_bytes);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to flush header in %s, errno=%u\n", filepath, errno);
        Die();
      } else {
        need_write_bytes -= write_bytes;
        buff_pos += write_bytes;
      }
    }
  }

  internal_close(fd_header);
}

void ThreadContext::put_trace(void *msg, uptr len) {
  {
    isFuncEnterMetaVaild = false;
    isFuncExitMetaVaild = false;
    parammetas.Resize(0);
    dbg_temp_buffer_size = 0;
  }
  if (UNLIKELY(trace_buffer == nullptr)) {
    trace_buffer = (char *)internal_alloc(MBlockShadowStack, TREC_BUFFER_SIZE);
    trace_buffer_size = 0;
  }
  if (UNLIKELY(trace_buffer_size + len >= TREC_BUFFER_SIZE)) {
    flush_trace();
    flush_metadata();
    flush_header();
  }
  internal_memcpy(trace_buffer + trace_buffer_size, msg, len);
  trace_buffer_size += len;
  event_cnt += 1;
}

void ThreadContext::put_metadata(void *msg, uptr len) {
  if (UNLIKELY(metadata_buffer == nullptr)) {
    metadata_buffer =
        (char *)internal_alloc(MBlockShadowStack, TREC_BUFFER_SIZE);
    metadata_buffer_size = 0;
  }
  if (UNLIKELY(metadata_buffer_size + len > TREC_BUFFER_SIZE)) {
    flush_trace();
    flush_metadata();
    flush_header();
  }
  internal_memcpy(metadata_buffer + metadata_buffer_size, msg, len);
  metadata_buffer_size += len;
  metadata_offset += len;
}

void ThreadContext::put_debug_info(void *msg, uptr len) {
  if (UNLIKELY(debug_buffer == nullptr)) {
    debug_buffer = (char *)internal_alloc(MBlockShadowStack, TREC_BUFFER_SIZE);
    debug_buffer_size = 0;
  }
  if (UNLIKELY(debug_buffer_size + len > TREC_BUFFER_SIZE)) {
    flush_trace();
    flush_metadata();
    flush_header();
  }
  internal_memcpy(debug_buffer + debug_buffer_size, msg, len);
  debug_buffer_size += len;
  debug_offset += len;
}

struct OnStartedArgs {
  ThreadState *thr;
};

void ThreadContext::OnStarted(void *arg) {
  OnStartedArgs *args = static_cast<OnStartedArgs *>(arg);
  thr = args->thr;
  new (thr) ThreadState(ctx, tid, unique_id);
  thr->is_inited = true;
  DPrintf("#%d: ThreadStart\n", tid);
}

void ThreadContext::OnFinished() {
#if !SANITIZER_GO
  PlatformCleanUpThreadState(thr);
#endif
  thr->~ThreadState();
  thr = 0;
}

void ThreadFinalize(ThreadState *thr) {
  if (LIKELY(ctx->flags.output_trace)) {
    if (ctx->flags.trace_mode == 1) {
      __seqc_trace::Event e;
      e.type = __seqc_trace::EventType::END;
      e.eid = thr->tctx->event_cnt++;
      e.iid = 0;
      e.oid = 0;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->flush_seqc_summary();
      ctx->flush_seqc_trace();
      if (ctx->seqc_trace_buffer) {
        internal_free(ctx->seqc_trace_buffer);
        ctx->seqc_trace_buffer = nullptr;
      }
      ctx->seqc_trace_buffer_size = 0;
      ctx->seqc_mtx.Unlock();
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::ThreadEnd, thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), thr->tid,
          0, 0);

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->flush_trace();
      thr->tctx->flush_metadata();
      thr->tctx->flush_header();
      if (thr->tctx->trace_buffer) {
        internal_free(thr->tctx->trace_buffer);
        thr->tctx->trace_buffer = nullptr;
      }
      if (thr->tctx->metadata_buffer) {
        internal_free(thr->tctx->metadata_buffer);
        thr->tctx->metadata_buffer = nullptr;
      }
      if (thr->tctx->debug_buffer) {
        internal_free(thr->tctx->debug_buffer);
        thr->tctx->debug_buffer = nullptr;
      }
      thr->tctx->trace_buffer_size = 0;
      thr->tctx->metadata_buffer_size = 0;
      thr->tctx->debug_buffer_size = 0;
    }
  }
}

int ThreadCount(ThreadState *thr) {
  uptr result;
  ctx->thread_registry->GetNumberOfThreads(0, 0, &result);
  return (int)result;
}

int ThreadCreate(ThreadState *thr, uptr pc, uptr uid, bool detached) {
  OnCreatedArgs args = {thr, pc};
  u32 parent_tid = thr ? thr->tid : kInvalidTid;  // No parent for GCD workers.
  int tid =
      ctx->thread_registry->CreateThread(uid, detached, parent_tid, &args);
  DPrintf("#%d: ThreadCreate tid=%d uid=%zu\n", parent_tid, tid, uid);
  if (tid == 0) {
    const char *mode = GetEnv("TREC_MODE");
    if (mode == nullptr) {
      ctx->flags.output_trace = false;
      ctx->flags.output_debug = false;
    } else {
      const char *trace_dir_env = GetEnv("TREC_TRACE_DIR");
      if (trace_dir_env == nullptr) {
        Report("TREC_TRACE_DIR has not been set!\n");
        Die();
      } else
        internal_strncpy(ctx->trace_dir, trace_dir_env,
                         internal_strlen(trace_dir_env));
      internal_strncpy(ctx->record_mode, mode, internal_strlen(mode));
      if (internal_strcmp(ctx->record_mode, "seqcheck") == 0) {
        // Report("Trace mode: SeqCheck\n");
        ctx->flags.record_alloc_free = false;
        ctx->flags.trace_mode = 1;
        ctx->flags.output_debug = false;
        ctx->trace_summary.tNum = ctx->trace_summary.totNum =
            ctx->trace_summary.arNum = ctx->trace_summary.brNum =
                ctx->trace_summary.lNum = ctx->trace_summary.mNum =
                    ctx->trace_summary.rwNum = 0;
      } else if (internal_strcmp(ctx->record_mode, "eagle") == 0) {
        // Report("Trace mode: Eagle\n");
        ctx->flags.trace_mode = 2;
      } else if (internal_strcmp(ctx->record_mode, "verification") == 0) {
        // Report("Trace mode: Program Verification\n");
        // ctx->flags.record_alloc_free = false;
        // ctx->flags.record_mutex = false;
        // ctx->flags.record_read = false;
        ctx->flags.trace_mode = 3;
      } else {
        Report("Unknown mode: %s\n", mode);
        Die();
      }
      atomic_store(&ctx->global_id, 0, memory_order_relaxed);
      atomic_store(&ctx->forked_cnt, 0, memory_order_relaxed);
      if (ctx->flags.trace_mode == 1) {
        ctx->seqc_trace_buffer_size = 0;
        ctx->trace_summary.tNum = 1;
      }
    }
  } else if (LIKELY(thr != nullptr && thr->tctx != nullptr)) {
    if (LIKELY(ctx->flags.output_trace)) {
      if (ctx->flags.trace_mode == 1) {
        __seqc_trace::Event e;
        e.type = __seqc_trace::EventType::FORK;
        e.eid = thr->tctx->event_cnt++;
        e.iid = pc;
        e.oid = tid;
        e.tid = thr->tid;
        ctx->seqc_mtx.Lock();
        e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
        ctx->put_seqc_trace(&e, sizeof(e));
        ctx->seqc_mtx.Unlock();
      } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
        __trec_trace::Event e(
            __trec_trace::EventType::ThreadCreate, thr->tid,
            atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), tid, 0,
            pc);

        thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
        thr->tctx->header.StateInc(__trec_header::RecordType::ThreadCreate);
      }
    }
  }
  return tid;
}

void ThreadStart(ThreadState *thr, int tid, tid_t os_id,
                 ThreadType thread_type) {
  ThreadRegistry *tr = ctx->thread_registry;
  OnStartedArgs args = {thr};
  tr->StartThread(tid, os_id, thread_type, &args);

  tr->Lock();
  thr->tctx = (ThreadContext *)tr->GetThreadLocked(tid);
  tr->Unlock();

  // gyq: never touch this
  // we should put the trace after it thr->tctx has been initialized
  if (LIKELY(ctx->flags.output_trace)) {
    if (ctx->flags.trace_mode == 1) {
      if (!ctx->thread_after_fork) {
        __seqc_trace::Event e;
        e.type = __seqc_trace::EventType::BEGIN;
        e.eid = thr->tctx->event_cnt++;
        e.iid = e.oid = 0;
        e.tid = thr->tid;
        ctx->seqc_mtx.Lock();
        e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
        ctx->put_seqc_trace(&e, sizeof(e));
        ctx->seqc_mtx.Unlock();
        ctx->trace_summary.tNum = max(ctx->trace_summary.tNum, (u32)(tid + 1));

      } else {
        ctx->seqc_mtx.Lock();
        if (ctx->seqc_trace_buffer) {
          internal_free(ctx->seqc_trace_buffer);
          ctx->seqc_trace_buffer = nullptr;
        }
        ctx->seqc_trace_buffer_size = 0;
        ctx->seqc_mtx.Unlock();
      }
      ctx->thread_after_fork = false;
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      if (thr->tctx->trace_buffer) {
        internal_free(thr->tctx->trace_buffer);
        thr->tctx->trace_buffer = nullptr;
      }
      if (thr->tctx->metadata_buffer) {
        internal_free(thr->tctx->metadata_buffer);
        thr->tctx->metadata_buffer = nullptr;
      }
      if (thr->tctx->debug_buffer) {
        internal_free(thr->tctx->debug_buffer);
        thr->tctx->debug_buffer = nullptr;
      }
      if (!thr->tctx->state_restore()) {
        thr->tctx->trace_buffer_size = 0;
        thr->tctx->metadata_buffer_size = 0;
        thr->tctx->debug_buffer_size = 0;
        thr->tctx->event_cnt = 0;
        thr->tctx->metadata_offset = 0;
        thr->tctx->debug_offset = 0;
        thr->tctx->prev_read_pc = 0;
        thr->tctx->dbg_temp_buffer_size = 0;
        __trec_trace::Event pad(__trec_trace::EventType::None,
                                cur_thread()->tid, 0,
                                __trec_trace::TREC_TRACE_VER, 0, 0);
        thr->tctx->put_trace(&pad, sizeof(pad));
        thr->tctx->put_metadata(
            (void *)__trec_metadata::TREC_METADATA_VER,
            internal_strlen(__trec_metadata::TREC_METADATA_VER));
        thr->tctx->put_debug_info(
            (void *)__trec_debug_info::TREC_DEBUG_VER,
            internal_strlen(__trec_debug_info::TREC_DEBUG_VER));
        __trec_trace::Event e(
            __trec_trace::EventType::ThreadBegin, thr->tid,
            atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), 0, 0,
            0);
        thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      }
      if (internal_strnlen(thr->tctx->header.cmd,
                           sizeof(thr->tctx->header.cmd) - 1) == 0) {
        char **cmds = GetArgv();
        int cmd_len = 0;
        internal_strlcpy(thr->tctx->header.binary_path, cmds[0],
                         2 * TREC_DIR_PATH_LEN - 1);
        for (int i = 0; cmds[i]; i++) {
          if (i != 0) {
            thr->tctx->header.cmd[cmd_len++] = ' ';
          }
          cmd_len +=
              internal_strlcpy(thr->tctx->header.cmd + cmd_len, cmds[i],
                               sizeof(thr->tctx->header.cmd) - 1 - cmd_len);
        }
      }
    }
  }
#if !SANITIZER_GO
  if (ctx->after_multithreaded_fork) {
    thr->ignore_interceptors++;
  }
#endif
}

void ThreadFinish(ThreadState *thr) {
  if (LIKELY(ctx->flags.output_trace)) {
    if (ctx->flags.trace_mode == 1) {
      __seqc_trace::Event e;
      e.type = __seqc_trace::EventType::END;
      e.eid = thr->tctx->event_cnt++;
      e.iid = 0;
      e.oid = 0;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->flush_seqc_summary();
      ctx->flush_seqc_trace();
      ctx->seqc_mtx.Unlock();
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::ThreadEnd, thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), thr->tid,
          0, 0);

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->flush_trace();
      thr->tctx->flush_metadata();
      thr->tctx->flush_header();
    }
  }
  if (thr->tctx->trace_buffer) {
    internal_free(thr->tctx->trace_buffer);
    thr->tctx->trace_buffer = nullptr;
  }
  if (thr->tctx->metadata_buffer) {
    internal_free(thr->tctx->metadata_buffer);
    thr->tctx->metadata_buffer = nullptr;
  }
  if (thr->tctx->debug_buffer) {
    internal_free(thr->tctx->debug_buffer);
    thr->tctx->debug_buffer = nullptr;
  }
  thr->tctx->trace_buffer_size = 0;
  thr->tctx->metadata_buffer_size = 0;
  thr->tctx->debug_buffer_size = 0;
  thr->is_dead = true;
  ctx->thread_registry->FinishThread(thr->tid);
}

struct ConsumeThreadContext {
  uptr uid;
  ThreadContextBase *tctx;
};

static bool ConsumeThreadByUid(ThreadContextBase *tctx, void *arg) {
  ConsumeThreadContext *findCtx = (ConsumeThreadContext *)arg;
  if (tctx->user_id == findCtx->uid && tctx->status != ThreadStatusInvalid) {
    if (findCtx->tctx) {
      // Ensure that user_id is unique. If it's not the case we are screwed.
      // Something went wrong before, but now there is no way to recover.
      // Returning a wrong thread is not an option, it may lead to very hard
      // to debug false positives (e.g. if we join a wrong thread).
      Report("TraceRecorder: dup thread with used id 0x%zx\n", findCtx->uid);
      Die();
    }
    findCtx->tctx = tctx;
    tctx->user_id = 0;
  }
  return false;
}

int ThreadConsumeTid(ThreadState *thr, uptr pc, uptr uid) {
  int tid = ctx->thread_registry->ConsumeThreadUserId(uid);
  DPrintf("#%d: ThreadTid uid=%zu tid=%d\n", thr->tid, uid, tid);
  return tid;
}

void ThreadJoin(ThreadState *thr, uptr pc, int tid) {
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  DPrintf("#%d: ThreadJoin tid=%d\n", thr->tid, tid);
  if (LIKELY(ctx->flags.output_trace)) {
    if (ctx->flags.trace_mode == 1) {
      __seqc_trace::Event e;
      e.type = __seqc_trace::EventType::JOIN;
      e.eid = thr->tctx->event_cnt++;
      e.iid = pc;
      e.oid = tid;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->seqc_mtx.Unlock();
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::ThreadJoin, thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), tid, 0,
          pc);

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::ThreadJoin);
    }
  }
  ctx->thread_registry->JoinThread(tid, thr);
}

void ThreadDetach(ThreadState *thr, uptr pc, int tid) {
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  ctx->thread_registry->DetachThread(tid, thr);
}

void ThreadNotJoined(ThreadState *thr, uptr pc, int tid, uptr uid) {
  CHECK_GT(tid, 0);
  CHECK_LT(tid, kMaxTid);
  ctx->thread_registry->SetThreadUserId(tid, uid);
}

void ThreadSetName(ThreadState *thr, const char *name) {
  ctx->thread_registry->SetThreadName(thr->tid, name);
}

void MemoryAccessRange(ThreadState *thr, uptr pc, uptr addr, uptr size,
                       bool is_write, __trec_metadata::SourceAddressInfo SAI) {
  // Currently we do not use memory range access: use memory access instead.

  // if (LIKELY(ctx->flags.output_trace)&&
  //    LIKELY(cur_thread()->ignore_interceptors == 0)) {
  //   if ((ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3)) {
  //     __trec_trace::Event e(
  //         is_write ? __trec_trace::EventType::MemRangeWrite
  //                  : __trec_trace::EventType::MemRangeRead,thr->tid,
  //         atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
  //         (((size & 0xffff) << 48) | (addr & ((((u64)1) << 48) - 1))),
  //         thr->tctx->metadata_offset);

  //     __trec_metadata::MemRangeMeta meta(SAI.idx, SAI.addr);
  //     thr->tctx->put_metadata(&meta, sizeof(meta));

  //     thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
  //     thr->tctx->header.StateInc(is_write
  //                                    ?
  //                                    __trec_header::RecordType::MemRangeWrite
  //                                    :
  //                                    __trec_header::RecordType::MemRangeRead);
  //   }
  // }
  return;
}

}  // namespace __trec
