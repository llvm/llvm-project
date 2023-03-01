//===-- trec_rtl.cpp
//------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
// Main file (entry points) for the TRec run-time.
//===----------------------------------------------------------------------===//

#include "trec_rtl.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <time.h>

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_libc.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_symbolizer.h"
#include "trec_defs.h"
#include "trec_mman.h"
#include "trec_platform.h"
#include "ubsan/ubsan_init.h"

#ifdef __SSE3__
// <emmintrin.h> transitively includes <stdlib.h>,
// and it's prohibited to include std headers into trec runtime.
// So we do this dirty trick.
#define _MM_MALLOC_H_INCLUDED
#define __MM_MALLOC_H
#include <emmintrin.h>
typedef __m128i m128;
#endif

namespace __trec {

#if !SANITIZER_GO && !SANITIZER_MAC
__attribute__((tls_model("initial-exec")))
THREADLOCAL char cur_thread_placeholder[sizeof(ThreadState)] ALIGNED(64);
#endif
static char ctx_placeholder[sizeof(Context)] ALIGNED(64);
Context *ctx;

static char thread_registry_placeholder[sizeof(ThreadRegistry)];

static ThreadContextBase *CreateThreadContext(u32 tid) {
  void *mem = internal_alloc(MBlockThreadContex, sizeof(ThreadContext));
  return new (mem) ThreadContext(tid);
}

#if !SANITIZER_GO
static const u32 kThreadQuarantineSize = 16;
#else
static const u32 kThreadQuarantineSize = 64;
#endif

Context::Context()
    : initialized(),
      pid(internal_getpid()),
      thread_registry(new(thread_registry_placeholder) ThreadRegistry(
          CreateThreadContext, kMaxTid, kThreadQuarantineSize, kMaxTidReuse)),
      seqc_trace_buffer(nullptr),
      seqc_trace_buffer_size(0),
      temp_dir_path(nullptr) {}
int Context::CopyFile(const char *src_path, const char *dest_path) {
  char *read_buff = (char *)internal_alloc(
      MBlockShadowStack, TREC_BUFFER_SIZE);  // buffer size:32M
  int src_fd = internal_open(src_path, O_RDONLY);
  int dest_fd = internal_open(dest_path, O_CREAT | O_WRONLY | O_APPEND, 0700);
  if (src_fd < 0 || dest_fd < 0)
    return 1;
  uptr read_bytes = 0;
  while ((read_bytes = internal_read(src_fd, read_buff, TREC_BUFFER_SIZE)) >
         0) {
    while (read_bytes > 0) {
      char *buff_pos = read_buff;
      uptr write_bytes = internal_write(dest_fd, buff_pos, read_bytes);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to copy file from %s to %s, errno=%u\n", src_path,
               dest_path, errno);
        Die();
      } else {
        read_bytes -= write_bytes;
        buff_pos += write_bytes;
      }
    }
  }
  internal_close(src_fd);
  internal_close(dest_fd);
  internal_free(read_buff);
  return 0;
}

void Context::CopyDir(const char *path, int Maintid) {
  char parent_path[TREC_DIR_PATH_LEN];
  internal_snprintf(parent_path, TREC_DIR_PATH_LEN - 1, "%s/trec_%d", trace_dir,
                    internal_getpid());
  char src_path[2 * TREC_DIR_PATH_LEN], dest_path[2 * TREC_DIR_PATH_LEN];

  if (mkdir(path, 0700)) {
    Report("Create temp directory failed\n");
    Die();
  }
  if (ctx->flags.trace_mode == 1) {
    internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/meta.bin",
                      parent_path);
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/meta.bin",
                      path);

    if (CopyFile(src_path, dest_path)) {
      Report("Parent copy meta file failed\n");
      Die();
    }
    internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/trace.bin",
                      parent_path);
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/trace.bin",
                      path);

    if (CopyFile(src_path, dest_path)) {
      Report("Parent copy trace failed\n");
      Die();
    }
  } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
    if (ctx->flags.output_debug) {
      internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/debug", path);
      if (mkdir(dest_path, 0700)) {
        Report("Create temp debug info directory failed\n");
        Die();
      }
    }
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/header", path);
    if (mkdir(dest_path, 0700)) {
      Report("Create temp header directory failed\n");
      Die();
    }
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/metadata",
                      path);
    if (mkdir(dest_path, 0700)) {
      Report("Create temp metadata directory failed\n");
      Die();
    }
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/trace", path);
    if (mkdir(dest_path, 0700)) {
      Report("Create temp trace directory failed\n");
      Die();
    }
    if (ctx->flags.output_debug) {
      internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/debug/%u.bin",
                        parent_path, Maintid);
      internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/debug/%u.bin",
                        path, Maintid);

      if (CopyFile(src_path, dest_path)) {
        Report("Parent copy bin debug info failed\n");
        Die();
      }
    }

    internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/header/%u.bin",
                      parent_path, Maintid);
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/header/%u.bin",
                      path, Maintid);

    if (CopyFile(src_path, dest_path)) {
      Report("Parent copy bin header failed\n");
      Die();
    }
    internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1,
                      "%s/header/modules_%u.txt", parent_path, Maintid);
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1,
                      "%s/header/modules_%u.txt", path, Maintid);
    if (CopyFile(src_path, dest_path)) {
      Report("Parent copy module file failed\n");
      Die();
    }

    internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/metadata/%u.bin",
                      parent_path, Maintid);
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1,
                      "%s/metadata/%u.bin", path, Maintid);
    if (CopyFile(src_path, dest_path)) {
      Report("Parent copy metadata file failed\n");
      Die();
    }
    internal_snprintf(src_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/trace/%u.bin",
                      parent_path, Maintid);
    internal_snprintf(dest_path, 2 * TREC_DIR_PATH_LEN - 1, "%s/trace/%u.bin",
                      path, Maintid);
    if (CopyFile(src_path, dest_path)) {
      Report("Parent copy trace file failed\n");
      Die();
    }
  }
  return;
}

void Context::InheritDir(const char *path, uptr _pid) {
  char dirpath[TREC_DIR_PATH_LEN];
  internal_snprintf(dirpath, TREC_DIR_PATH_LEN - 1, "%s/trec_%llu", trace_dir,
                    _pid);
  if (internal_rename(path, dirpath)) {
    Report("Child inherit directory failed\n");
    Die();
  }
  return;
}

void Context::open_directory(const char *dirpath) {
  // open or create
  char path[TREC_DIR_PATH_LEN];
  internal_snprintf(path, TREC_DIR_PATH_LEN - 1, "%s/trec_%d", dirpath,
                    internal_getpid());

  struct stat _st = {0};
  open_dir_mutex.Lock();
  uptr IS_EXISTS = __sanitizer::internal_stat(path, &_st);
  char filepath[TREC_DIR_PATH_LEN];
  if (IS_EXISTS == 0) {
    open_dir_mutex.Unlock();
    return;
  } else {
    if (mkdir(path, ACCESSPERMS) != 0) {
      Report("Could not create directory at %s, errno=%d, exists=%d, is_dir=%d",
             path, errno, IS_EXISTS, S_ISDIR(_st.st_mode));
      Die();
    }
  }
  if (flags.trace_mode == 2 || flags.trace_mode == 3) {
    internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/%s", path, "trace");
    IS_EXISTS = __sanitizer::internal_stat(filepath, &_st);
    if (IS_EXISTS != 0) {
      if (mkdir(filepath, ACCESSPERMS) != 0) {
        Report("Could not create trace directory at %s, errno=%d\n", filepath,
               errno);
        Die();
      }
    }
    internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/%s", path,
                      "metadata");
    IS_EXISTS = __sanitizer::internal_stat(filepath, &_st);
    if (IS_EXISTS != 0) {
      if (mkdir(filepath, ACCESSPERMS) != 0) {
        Report("Could not create metadata directory at %s, errno=%d\n",
               filepath, errno);
        Die();
      }
    }
    internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/%s", path, "header");
    IS_EXISTS = __sanitizer::internal_stat(filepath, &_st);
    if (IS_EXISTS != 0) {
      if (mkdir(filepath, ACCESSPERMS) != 0) {
        Report("Could not create header directory at %s, errno=%d\n", filepath,
               errno);
        Die();
      }
    }
    if (ctx->flags.output_debug) {
      internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/%s", path,
                        "debug");
      IS_EXISTS = __sanitizer::internal_stat(filepath, &_st);
      if (IS_EXISTS != 0) {
        if (mkdir(filepath, ACCESSPERMS) != 0) {
          Report("Could not create debug directory at %s, errno=%d\n", filepath,
                 errno);
          Die();
        }
      }
    }
    cur_thread()->tctx->flush_module();
  }
  open_dir_mutex.Unlock();
}
void Context::flush_seqc_summary() {
  char filepath[TREC_DIR_PATH_LEN];
  struct stat _st = {0};
  internal_snprintf(filepath, 2 * TREC_DIR_PATH_LEN - 1, "%s/trec_%llu",
                    ctx->trace_dir, internal_getpid());
  uptr IS_EXIST = __sanitizer::internal_stat(filepath, &_st);
  if (IS_EXIST != 0) {
    ctx->open_directory(ctx->trace_dir);
  }

  internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1, "%s/trec_%d/meta.bin.orig",
                    ctx->trace_dir, internal_getpid());
  int fd_summary = internal_open(filepath, O_CREAT | O_WRONLY | O_TRUNC, 0700);

  if (UNLIKELY(fd_summary < 0)) {
    Report("Failed to open meta file at %s\n", filepath);
    Die();
  } else {
    char *buff_pos = (char *)&trace_summary;
    int summary_len = sizeof(trace_summary);
    while (summary_len > 0) {
      uptr write_bytes = internal_write(fd_summary, buff_pos, summary_len);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to flush meta info in %s, errno=%u\n", filepath, errno);
        Die();
      } else {
        summary_len -= write_bytes;
        buff_pos += write_bytes;
      }
    }
  }
  for (int idx = 0; idx < thread_event_cnt.Size(); idx++) {
    __sanitizer::u32 num = thread_event_cnt[idx];
    internal_write(fd_summary, &num, sizeof(num));
  }
  internal_close(fd_summary);
}

void Context::flush_seqc_trace() {
  char filepath[TREC_DIR_PATH_LEN];
  struct stat _st = {0};
  internal_snprintf(filepath, 2 * TREC_DIR_PATH_LEN - 1, "%s/trec_%llu",
                    ctx->trace_dir, internal_getpid());
  uptr IS_EXIST = __sanitizer::internal_stat(filepath, &_st);
  if (IS_EXIST != 0) {
    ctx->open_directory(ctx->trace_dir);
  }
  internal_snprintf(filepath, TREC_DIR_PATH_LEN - 1,
                    "%s/trec_%d/trace.bin.orig", ctx->trace_dir,
                    internal_getpid());
  int fd_trace = internal_open(filepath, O_CREAT | O_WRONLY | O_APPEND, 0700);

  if (UNLIKELY(fd_trace < 0)) {
    Report("Failed to open trace file at %s\n", filepath);
    Die();
  } else if (seqc_trace_buffer != nullptr && seqc_trace_buffer_size > 0) {
    char *buff_pos = (char *)seqc_trace_buffer;
    while (seqc_trace_buffer_size > 0) {
      uptr write_bytes =
          internal_write(fd_trace, buff_pos, seqc_trace_buffer_size);
      if (write_bytes == -1 && errno != EINTR) {
        Report("Failed to flush trace in %s, errno=%u\n", filepath, errno);
        Die();
      } else {
        seqc_trace_buffer_size -= write_bytes;
        buff_pos += write_bytes;
      }
    }
  }
  internal_close(fd_trace);
  return;
}

void Context::put_seqc_trace(void *msg, uptr len) {
  if (seqc_trace_buffer == nullptr) {
    seqc_trace_buffer =
        (char *)internal_alloc(MBlockShadowStack, SEQC_BUFFER_SIZE);
    seqc_trace_buffer_size = 0;
  }
  if (seqc_trace_buffer_size + len >= SEQC_BUFFER_SIZE) {
    flush_seqc_trace();
    flush_seqc_summary();
  }

  internal_memcpy(seqc_trace_buffer + seqc_trace_buffer_size, msg, len);
  seqc_trace_buffer_size += len;
  trace_summary.totNum += 1;
  if (thread_event_cnt.Size() < (cur_thread()->tid + 1))
    thread_event_cnt.PushBack(0);
  thread_event_cnt[cur_thread()->tid] += 1;
}

bool Context::state_restore() {
  struct stat _st = {0};
  char path[2 * TREC_DIR_PATH_LEN];
  internal_snprintf(path, 2 * TREC_DIR_PATH_LEN - 1, "%s/trec_%llu/meta.bin",
                    ctx->trace_dir, internal_getpid());
  uptr IS_EXIST = __sanitizer::internal_stat(path, &_st);
  if (IS_EXIST == 0 && _st.st_size > 0) {
    int summary_fd = internal_open(path, O_RDONLY);
    if (summary_fd < 0) {
      Report("Restore meta file from %s failed\n", path);
      return false;
    } else {
      internal_read(summary_fd, &trace_summary, sizeof(trace_summary));
      return true;
    }
  }
  return false;
}

// The objects are allocated in TLS, so one may rely on zero-initialization.
ThreadState::ThreadState(Context *ctx, int tid, int unique_id)
    : tid(tid), unique_id(unique_id), should_record(true) {}

#if !SANITIZER_GO

static void OnStackUnwind(const SignalContext &sig, const void *,
                          BufferedStackTrace *stack) {
  stack->Unwind(StackTrace::GetNextInstructionPc(sig.pc), sig.bp, sig.context,
                common_flags()->fast_unwind_on_fatal);
}

static void TrecOnDeadlySignal(int signo, void *siginfo, void *context) {
  HandleDeadlySignal(siginfo, context, GetTid(), &OnStackUnwind, nullptr);
}
#endif

void TrecCheckFailed(const char *file, int line, const char *cond, u64 v1,
                     u64 v2) {
  // There is high probability that interceptors will check-fail as well,
  // on the other hand there is no sense in processing interceptors
  // since we are going to die soon.
  ScopedIgnoreInterceptors ignore;
#if !SANITIZER_GO
  cur_thread()->ignore_sync++;
  cur_thread()->ignore_reads_and_writes++;
#endif
  Printf(
      "FATAL: TraceRecorder CHECK failed: "
      "%s:%d \"%s\" (0x%zx, 0x%zx)\n",
      file, line, cond, (uptr)v1, (uptr)v2);
  Die();
}

void Initialize(ThreadState *thr) {
  // Thread safe because done before all threads exist.
  static bool is_initialized = false;
  if (is_initialized)
    return;
  is_initialized = true;
  // We are not ready to handle interceptors yet.
  ScopedIgnoreInterceptors ignore;
  SanitizerToolName = "TraceRecorder";

  ctx = new (ctx_placeholder) Context;
  const char *env_name = SANITIZER_GO ? "GORACE" : "TREC_OPTIONS";
  const char *options = GetEnv(env_name);
  CacheBinaryName();
  CheckASLR();
  InitializeFlags(&ctx->flags, options, env_name);
  AvoidCVE_2016_2143();
  __sanitizer::InitializePlatformEarly();
  __trec::InitializePlatformEarly();

#if !SANITIZER_GO

  InitializeAllocator();
  ReplaceSystemMalloc();
#endif
  Processor *proc = ProcCreate();
  ProcWire(proc, thr);
  InitializeInterceptors();
  InitializePlatform();
#if !SANITIZER_GO
  InitializeAllocatorLate();

  // Do not install SEGV handler
  // InstallDeadlySignalHandlers(TrecOnDeadlySignal);
  if (common_flags()->use_sigaltstack)
    SetAlternateSignalStack();
#endif
  // Setup correct file descriptor for error reports.
  // __sanitizer_set_report_path(common_flags()->log_path);

  VPrintf(1, "***** Running under TraceRecorder v2 (pid %d) *****\n",
          (int)internal_getpid());

  // Initialize thread 0.
  int tid = ThreadCreate(thr, 0, 0, true);
  CHECK_EQ(tid, 0);
  ThreadStart(thr, tid, GetTid(), ThreadType::Regular);
  ctx->initialized = true;

#if !SANITIZER_GO
  Symbolizer::LateInitialize();
#endif
}

int Finalize(ThreadState *thr) {
  if (flags()->atexit_sleep_ms > 0 && ThreadCount(thr) > 1)
    SleepForMillis(flags()->atexit_sleep_ms);

  ThreadFinalize(thr);
  // gyq: always exit with code 0
  return 0;
}

#if !SANITIZER_GO
void ForkBefore(ThreadState *thr, uptr pc) {
  ctx->thread_registry->Lock();
  unsigned int cur_forked_cnt =
      atomic_fetch_add(&ctx->forked_cnt, 1, memory_order_relaxed);
  if (!ctx->temp_dir_path)
    internal_free(ctx->temp_dir_path);
  ctx->temp_dir_path =
      (char *)internal_alloc(MBlockShadowStack, TREC_DIR_PATH_LEN);
  internal_snprintf(ctx->temp_dir_path, TREC_DIR_PATH_LEN - 1, "%s/temp_%d_%d",
                    ctx->trace_dir, internal_getpid(), cur_forked_cnt);
  if (ctx->flags.output_trace &&
      LIKELY(cur_thread()->ignore_interceptors == 0)) {
    if (ctx->flags.trace_mode == 1) {
      ctx->seqc_mtx.Lock();
      ctx->flush_seqc_summary();
      ctx->flush_seqc_trace();
      ctx->seqc_mtx.Unlock();
      if (ctx->seqc_trace_buffer) {
        internal_free(ctx->seqc_trace_buffer);
        ctx->seqc_trace_buffer = nullptr;
      }
      ctx->seqc_trace_buffer_size = 0;
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      thr->tctx->header.StateInc(__trec_header::RecordType::ProcessFork);
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

    ctx->CopyDir(ctx->temp_dir_path, thr->tid);
  }
}

void ForkParentAfter(ThreadState *thr, uptr pc) {
  ctx->thread_registry->Unlock();
}

void ForkChildAfter(ThreadState *thr, uptr pc) {
  ctx->thread_registry->Unlock();

  uptr nthread = 0;
  ctx->thread_registry->GetNumberOfThreads(0, 0, &nthread /* alive threads */);
  VPrintf(1,
          "TraceRecorder: forked new process with pid %d,"
          " parent had %d threads\n",
          (int)internal_getpid(), (int)nthread);
  if (nthread != 1) {
    // We've just forked a multi-threaded process. We cannot reasonably
    // function after that (some mutexes may be locked before fork). So just
    // enable ignores for everything in the hope that we will exec soon.
    ctx->flags.output_trace = false;
    ctx->after_multithreaded_fork = true;
  }
  if (ctx->flags.output_trace && ctx->temp_dir_path &&
      LIKELY(cur_thread()->ignore_interceptors == 0)) {
    ctx->InheritDir(ctx->temp_dir_path, internal_getpid());
    if (ctx->flags.trace_mode == 1) {
      if (!ctx->state_restore()) {
        Report("restore state for pid=%d forked from pid=%d failed\n",
               internal_getpid(), internal_getppid());
        Die();
      }
      ctx->thread_after_fork = true;
    }
    internal_free(ctx->temp_dir_path);
    ctx->temp_dir_path = nullptr;
  }
}
#endif

void UnalignedMemoryAccess(ThreadState *thr, uptr pc, uptr addr, int size,
                           bool kAccessIsWrite, bool kIsAtomic, bool isPtr,
                           uptr val,
                           __trec_metadata::SourceAddressInfo SAI_addr,
                           __trec_metadata::SourceAddressInfo SAI_val) {
  int kAccessSizeLog;
  switch (size) {
    case 1:
      kAccessSizeLog = kSizeLog1;
      break;
    case 2:
      kAccessSizeLog = kSizeLog2;
      break;
    case 4:
      kAccessSizeLog = kSizeLog4;
      break;
    case 8:
      kAccessSizeLog = kSizeLog8;
      break;
    default:
      kAccessSizeLog = kSizeLog1;
  }

  MemoryAccess(thr, pc, addr, kAccessSizeLog, kAccessIsWrite, kIsAtomic, isPtr,
               val, SAI_addr, SAI_val);
}

ALWAYS_INLINE USED void CondBranch(ThreadState *thr, uptr pc,
                                   __sanitizer::u64 cond) {
  if (UNLIKELY(!ctx->flags.record_branch))
    return;
  if (LIKELY(ctx->flags.output_trace) &&
      LIKELY(thr->ignore_interceptors == 0) && LIKELY(thr->should_record)) {
    if (ctx->flags.trace_mode == 1) {
      __seqc_trace::Event e;
      e.type = __seqc_trace::EventType::BRANCH;
      e.eid = thr->tctx->event_cnt++;
      e.iid = pc;
      e.oid = 0;
      e.tid = thr->tid;
      ctx->seqc_mtx.Lock();
      e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
      ctx->put_seqc_trace(&e, sizeof(e));
      ctx->trace_summary.brNum += 1;
      ctx->seqc_mtx.Unlock();
    } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::Branch, thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), cond,
          thr->tctx->dbg_temp_buffer_size ? thr->tctx->metadata_offset : 0, pc);
      if (thr->tctx->dbg_temp_buffer_size) {
        __trec_metadata::BranchMeta meta(thr->tctx->debug_offset);
        thr->tctx->put_debug_info(thr->tctx->dbg_temp_buffer,
                                  thr->tctx->dbg_temp_buffer_size);
        thr->tctx->put_metadata(&meta, sizeof(meta));
      }
      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::Branch);
    }
  }
}

ALWAYS_INLINE USED void FuncParam(ThreadState *thr, u16 param_idx,
                                  uptr src_addr, u16 src_idx, uptr val) {
  if (LIKELY(ctx->flags.output_trace) && LIKELY(ctx->flags.record_func_param) &&
      LIKELY(cur_thread()->ignore_interceptors == 0)) {
    if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      if (param_idx != 0) {
        if (thr->tctx->isFuncEnterMetaVaild) {
          __trec_metadata::FuncParamMeta meta(param_idx, src_idx, src_addr,
                                              val);
          thr->tctx->parammetas.PushBack(meta);
          thr->tctx->entry_meta.parammeta_cnt += 1;
        }
      } else {
        thr->tctx->entry_meta.order = src_addr;
        thr->tctx->entry_meta.arg_size = src_idx;
        thr->tctx->entry_meta.parammeta_cnt = 0;
        thr->tctx->isFuncEnterMetaVaild = true;
        thr->tctx->parammetas.Resize(0);
      }
    }
  }
}

ALWAYS_INLINE USED void FuncExitParam(ThreadState *thr, uptr src_addr,
                                      u16 src_idx, uptr val) {
  if (LIKELY(ctx->flags.output_trace) && LIKELY(ctx->flags.record_func_param) &&
      LIKELY(cur_thread()->ignore_interceptors == 0)) {
    if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      thr->tctx->exit_meta.src_idx = src_idx;
      thr->tctx->exit_meta.src_addr = src_addr;
      thr->tctx->exit_meta.val = val;
      thr->tctx->isFuncExitMetaVaild = src_idx || src_addr;
    }
  }
}

ALWAYS_INLINE USED void MemoryAccess(
    ThreadState *thr, uptr pc, uptr addr, int kAccessSizeLog,
    bool kAccessIsWrite, bool kIsAtomic, bool isPtr, uptr val,
    __trec_metadata::SourceAddressInfo SAI_addr,
    __trec_metadata::SourceAddressInfo SAI_val, bool isMemCpyFlag) {
  if (LIKELY(ctx->flags.output_trace) &&
      LIKELY(thr->ignore_interceptors == 0) && LIKELY(thr->should_record)) {
    if (kAccessIsWrite && LIKELY(ctx->flags.record_write)) {
      if (ctx->flags.trace_mode == 1) {
        __seqc_trace::Event e;
        e.type = __seqc_trace::EventType::WRITE;
        e.eid = thr->tctx->event_cnt++;
        e.iid = pc;
        e.oid = addr;
        e.tid = thr->tid;
        ctx->seqc_mtx.Lock();
        e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
        ctx->put_seqc_trace(&e, sizeof(e));
        ctx->trace_summary.rwNum += 1;
        ctx->seqc_mtx.Unlock();
      } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
        __trec_trace::Event e(
            isPtr ? (val ? __trec_trace::EventType::PtrWrite
                         : __trec_trace::EventType::PtrZeroWrite)
                  : (val ? __trec_trace::EventType::PlainWrite
                         : __trec_trace::EventType::ZeroWrite),
            thr->tid,
            atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
            ((((u64)1) << (kAccessSizeLog + 48)) |
             (addr & ((((u64)1) << 48) - 1))),
            thr->tctx->metadata_offset, pc);

        __trec_metadata::WriteMeta meta(
            val, SAI_addr.idx, SAI_addr.addr, SAI_val.idx, SAI_val.addr,
            isMemCpyFlag,
            thr->tctx->dbg_temp_buffer_size ? thr->tctx->debug_offset : 0);
        thr->tctx->put_metadata(&meta, sizeof(meta));
        if (thr->tctx->dbg_temp_buffer_size)
          thr->tctx->put_debug_info(thr->tctx->dbg_temp_buffer,
                                    thr->tctx->dbg_temp_buffer_size);
        thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
        thr->tctx->header.StateInc(
            isPtr ? (val ? __trec_header::RecordType::PtrWrite
                         : __trec_header::RecordType::PtrZeroWrite)
                  : (val ? __trec_header::RecordType::PlainWrite
                         : __trec_header::RecordType::ZeroWrite));
      }
    } else if (!kAccessIsWrite && LIKELY(ctx->flags.record_read)) {
      if (ctx->flags.trace_mode == 1) {
        __seqc_trace::Event e;
        e.type = __seqc_trace::EventType::READ;
        e.eid = thr->tctx->event_cnt++;
        e.iid = pc;
        e.oid = addr;
        e.tid = thr->tid;
        ctx->seqc_mtx.Lock();
        e.tot = atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed);
        ctx->put_seqc_trace(&e, sizeof(e));
        ctx->trace_summary.rwNum += 1;
        ctx->seqc_mtx.Unlock();
      } else if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
        __trec_trace::Event e(
            isPtr ? __trec_trace::EventType::PtrRead
                  : __trec_trace::EventType::PlainRead,
            thr->tid,
            atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed),
            ((((u64)1) << (kAccessSizeLog + 48)) |
             (addr & ((((u64)1) << 48) - 1))),
            thr->tctx->metadata_offset, pc);

        __trec_metadata::ReadMeta meta(
            val, SAI_addr.idx, SAI_addr.addr, isMemCpyFlag,
            thr->tctx->dbg_temp_buffer_size ? thr->tctx->debug_offset : 0);
        thr->tctx->put_metadata(&meta, sizeof(meta));
        if (thr->tctx->dbg_temp_buffer_size) {
          __sanitizer::u64 prev = thr->tctx->debug_offset;
          thr->tctx->put_debug_info(thr->tctx->dbg_temp_buffer,
                                    thr->tctx->dbg_temp_buffer_size);
        }
        thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
        thr->tctx->header.StateInc(isPtr
                                       ? __trec_header::RecordType::PtrRead
                                       : __trec_header::RecordType::PlainRead);
      }
    }
  }
}

ALWAYS_INLINE USED void RecordFuncEntry(ThreadState *thr, bool &should_record,
                                        const char *name, __sanitizer::u64 pc) {
  if (LIKELY(ctx->flags.output_trace) &&
      LIKELY(ctx->flags.record_func_enter_exit) &&
      LIKELY(thr->ignore_interceptors == 0)) {
    Report("tid=%d name=%s ctx->flags.addr=%p\n", thr->tid, name,
           &(ctx->flags));
           
    if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      should_record = should_record || thr->tctx->isFuncEnterMetaVaild ||
                      thr->tctx->dbg_temp_buffer_size != 0;
      if (should_record && thr->should_record) {
        __sanitizer::u64 oid =
            (((thr->tctx->isFuncEnterMetaVaild ? thr->tctx->entry_meta.order
                                               : 0)
              << 56) |
             ((thr->tctx->isFuncEnterMetaVaild
                   ? thr->tctx->entry_meta.parammeta_cnt
                   : 0)
              << 48) |
             (ctx->flags.output_debug && thr->tctx->dbg_temp_buffer_size
                  ? (((((u64)1) << 48) - 1) & (thr->tctx->debug_offset))
                  : 0));
        __trec_trace::Event e(
            __trec_trace::EventType::FuncEnter, thr->tid,
            atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), oid,
            thr->tctx->parammetas.Size() ? thr->tctx->metadata_offset : 0, pc);
        for (uptr i = 0; i < thr->tctx->parammetas.Size(); i++)
          thr->tctx->put_metadata(&thr->tctx->parammetas[i],
                                  sizeof(__trec_metadata::FuncParamMeta));
        if (ctx->flags.output_debug && thr->tctx->dbg_temp_buffer_size) {
          __trec_debug_info::InstDebugInfo &debug_info =
              (*(__trec_debug_info::InstDebugInfo *)thr->tctx->dbg_temp_buffer);
          if (debug_info.name_len[0] == 0 && internal_strlen(name) != 0) {
            internal_memmove(thr->tctx->dbg_temp_buffer +
                                 sizeof(__trec_debug_info::InstDebugInfo) +
                                 internal_strlen(name),
                             thr->tctx->dbg_temp_buffer +
                                 sizeof(__trec_debug_info::InstDebugInfo),
                             debug_info.name_len[1]);
            internal_memcpy(thr->tctx->dbg_temp_buffer +
                                sizeof(__trec_debug_info::InstDebugInfo),
                            name, internal_strlen(name));
            debug_info.name_len[0] = internal_strlen(name);
            thr->tctx->dbg_temp_buffer_size =
                sizeof(__trec_debug_info::InstDebugInfo) +
                internal_strlen(name) + debug_info.name_len[1];
          }
          thr->tctx->put_debug_info(thr->tctx->dbg_temp_buffer,
                                    thr->tctx->dbg_temp_buffer_size);
        }
        thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
        thr->tctx->header.StateInc(__trec_header::RecordType::FuncEnter);
      }
      thr->tctx->isFuncEnterMetaVaild = false;
      thr->tctx->isFuncExitMetaVaild = false;
      thr->tctx->parammetas.Resize(0);
      thr->tctx->dbg_temp_buffer_size = 0;
    }
  }
}

ALWAYS_INLINE USED void RecordFuncExit(ThreadState *thr, bool &should_record,
                                       const char *name) {
  if (LIKELY(ctx->flags.output_trace) &&
      LIKELY(ctx->flags.record_func_enter_exit) && should_record &&
      thr->should_record && LIKELY(thr->ignore_interceptors == 0)) {
    if (ctx->flags.trace_mode == 2 || ctx->flags.trace_mode == 3) {
      __trec_trace::Event e(
          __trec_trace::EventType::FuncExit, thr->tid,
          atomic_fetch_add(&ctx->global_id, 1, memory_order_relaxed), 0,
          thr->tctx->isFuncExitMetaVaild ? thr->tctx->metadata_offset : 0, 0);
      if (thr->tctx->isFuncExitMetaVaild)
        thr->tctx->put_metadata(&thr->tctx->exit_meta,
                                sizeof(thr->tctx->exit_meta));

      thr->tctx->put_trace(&e, sizeof(__trec_trace::Event));
      thr->tctx->header.StateInc(__trec_header::RecordType::FuncExit);
      thr->tctx->isFuncEnterMetaVaild = false;
      thr->tctx->isFuncExitMetaVaild = false;
      thr->tctx->parammetas.Resize(0);
      thr->tctx->dbg_temp_buffer_size = 0;
    }
  }
}

}  // namespace __trec

#if !SANITIZER_GO
// Must be included in this file to make sure everything is inlined.
#include "trec_interface_inl.h"
#endif

#if !SANITIZER_GO
void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void *context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  uptr top = 0;
  uptr bottom = 0;
  if (StackTrace::WillUseFastUnwind(request_fast)) {
    GetThreadStackTopAndBottom(false, &top, &bottom);
    Unwind(max_depth, pc, bp, nullptr, top, bottom, true);
  } else
    Unwind(max_depth, pc, 0, context, 0, 0, false);
}
#endif  // SANITIZER_GO
