//===-- trec_defs.h ---------------------------------------------*- C++ -*-===//
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

#ifndef TREC_DEFS_H
#define TREC_DEFS_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_libc.h"

// Setup defaults for compile definitions.
#ifndef TREC_NO_HISTORY
#define TREC_NO_HISTORY 0
#endif

#ifndef TREC_BUFFER_SIZE
#define TREC_BUFFER_SIZE (1 << 28)  // default buffer size: 256MB
#endif

#ifndef SEQC_BUFFER_SIZE
#define SEQC_BUFFER_SIZE (1 << 28)  // default buffer size: 256MB
#endif

#ifndef TREC_DIR_PATH_LEN
#define TREC_DIR_PATH_LEN 256
#endif

#ifndef TREC_HAS_128_BIT
#define TREC_HAS_128_BIT 0
#endif

namespace __trec {

const unsigned kMaxTidReuse = (1 << 22) - 1;
const unsigned kMaxTid = (1 << 13) - 1;
const __sanitizer::u16 kInvalidTid = kMaxTid + 1;

template <typename T>
T min(T a, T b) {
  return a < b ? a : b;
}

template <typename T>
T max(T a, T b) {
  return a > b ? a : b;
}

struct Processor;
struct ThreadState;
class ThreadContext;
struct Context;

}  // namespace __trec

namespace __trec_trace {
const __sanitizer::u64 TREC_TRACE_VER = 20221225UL;
enum EventType : __sanitizer::u64 {
  ThreadBegin,
  ThreadEnd,
  PlainRead,
  PlainWrite,
  ZeroWrite,
  PtrRead,
  PtrWrite,
  PtrZeroWrite,
  Branch,
  FuncEnter,
  FuncExit,
  ThreadCreate,
  ThreadJoin,
  MutexLock,
  ReaderLock,
  MutexUnlock,
  ReaderUnlock,
  MemAlloc,
  MemFree,
  MemRangeRead,
  MemRangeWrite,
  CondWait,
  CondSignal,
  CondBroadcast,
  None,
  EventTypeSize,
};
static_assert(EventType::EventTypeSize < 256,
              "ERROR: EventType::EventTypeSize >= 256");
struct Event {
  EventType type : 8;
  __sanitizer::u64 tid : 8;
  __sanitizer::u64 gid : 48;

  /*
   * Read/Write:  size : 16
                  dest address : 48
   * FuncEnter:   order : 8;
                  arg_cnt : 8;
                  debug_offset : 48
   * FuncExit:    0
   * MutexLock/MutexUnlock/ReaderLock/ReaderUnlock:
                  none : 16
                  lock address : 48
   * MemAlloc/MemFree:
                  size : 16
                  address : 48
   * MemRangeRead/MemRangeWrite:
                  size : 16
                  address : 48
   * CondBranch:  0
   * CondWait:    none : 16
                  cond address : 48
   */
  __sanitizer::u64 oid;
  __sanitizer::u64 meta_offset;
  __sanitizer::u64 pc;
  Event(EventType _type, __sanitizer::u64 _tid, __sanitizer::u64 _gid,
        __sanitizer::u64 _oid, __sanitizer::u64 _offset, __sanitizer::u64 _pc)
      : type(_type),
        tid(_tid),
        gid(_gid),
        oid(_oid),
        meta_offset(_offset),
        pc(_pc) {}
};
static_assert(sizeof(Event) == 32, "ERROR: sizeof(Event) != 32");
}  // namespace __trec_trace

namespace __trec_metadata {
const char TREC_METADATA_VER[] = "20221206";
struct SourceAddressInfo {
  // signed bit : 1  如果为0表示addr域是地址，如果为1表示addr域是标号
  // offset     : 15 表示地址偏移量
  __sanitizer::u64 idx : 16;
  __sanitizer::u64 addr : 48;  // source variable's address, zero if not exist
  SourceAddressInfo(__sanitizer::u16 _idx = 0, __sanitizer::u64 _addr = 0)
      : idx((__sanitizer::u64)_idx), addr(_addr) {}
};
static_assert(sizeof(SourceAddressInfo) == 8,
              "ERROR: sizeof(SourceAddressInfo)!=8");

struct ReadMeta {
  __sanitizer::u64 val;
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  __sanitizer::u64 none2 : 16;
  __sanitizer::u64 debug_offset : 48;
  ReadMeta(__sanitizer::u64 v, __sanitizer::u64 si, __sanitizer::u64 sa,
           __sanitizer::u16 isMemCpyFlag, __sanitizer::u64 debug = 0)
      : val(v),
        src_idx(si),
        src_addr(sa),
        none2(isMemCpyFlag),
        debug_offset(debug) {}
};
static_assert(sizeof(ReadMeta) == 24, "ERROR: sizeof(ReadMeta)!=24");

struct WriteMeta {
  __sanitizer::u64 val;
  __sanitizer::u64 addr_src_idx : 16;
  __sanitizer::u64 addr_src_addr : 48;
  __sanitizer::u64 val_src_idx : 16;
  __sanitizer::u64 val_src_addr : 48;
  __sanitizer::u64 none2 : 16;
  __sanitizer::u64 debug_offset : 48;
  WriteMeta(__sanitizer::u64 v, __sanitizer::u64 asi, __sanitizer::u64 asa,
            __sanitizer::u64 vsi, __sanitizer::u64 vsa,
             __sanitizer::u16 isMemCpyFlag, __sanitizer::u64 debug = 0)
      : val(v),
        addr_src_idx(asi),
        addr_src_addr(asa),
        val_src_idx(vsi),
        val_src_addr(vsa),
        none2(isMemCpyFlag),
        debug_offset(debug) {}
};
static_assert(sizeof(WriteMeta) == 32, "ERROR: sizeof(WriteMeta)!=32");

struct BranchMeta {
  __sanitizer::u64 none : 16;
  __sanitizer::u64 debug_offset : 48;
  BranchMeta(__sanitizer::u64 debug) : debug_offset(debug) {}
};
static_assert(sizeof(BranchMeta) == 8, "ERROR: sizeof(BranchMeta) != 8");

struct FuncEnterMeta {
  __sanitizer::u64 order;
  __sanitizer::u64 arg_size;
  __sanitizer::u64 parammeta_cnt;
  FuncEnterMeta(__sanitizer::u64 o, __sanitizer::u32 s, __sanitizer::u32 c)
      : order(o), arg_size(s), parammeta_cnt(c) {}
  FuncEnterMeta() {}
};

struct FuncParamMeta {
  __sanitizer::u64 id;
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  __sanitizer::u64 val;
  FuncParamMeta(__sanitizer::u64 _id, __sanitizer::u16 si, __sanitizer::u64 sa,
                __sanitizer::u64 v)
      : id(_id), src_idx(si), src_addr(sa), val(v) {}
};
static_assert(sizeof(FuncParamMeta) == 24, "ERROR: sizeof(FuncParamMeta)!=24");

struct FuncExitMeta {
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  __sanitizer::u64 val;
  FuncExitMeta(__sanitizer::u16 idx, __sanitizer::u64 addr, __sanitizer::u64 v)
      : src_idx(idx), src_addr(addr), val(v) {}
  FuncExitMeta() {}
};
static_assert(sizeof(FuncExitMeta) == 16, "ERROR: sizeof(FuncExitmMeta)!=16");

struct MemFreeMeta {
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  MemFreeMeta(__sanitizer::u64 si, __sanitizer::u64 sa)
      : src_idx(si), src_addr(sa) {}
};
static_assert(sizeof(MemFreeMeta) == 8, "ERROR: sizeof(MemFreeMeta) != 8");

struct MutexMeta {
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  MutexMeta(__sanitizer::u64 si, __sanitizer::u64 sa)
      : src_idx(si), src_addr(sa) {}
};
static_assert(sizeof(MutexMeta) == 8, "ERROR: sizeof(MemFreeMeta) != 8");

struct CondWaitMeta {
  __sanitizer::u64 none : 16;
  __sanitizer::u64 mutex : 48;
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;

  __sanitizer::u64 mutex_src_idx : 16;
  __sanitizer::u64 mutex_src_addr : 48;
  CondWaitMeta(__sanitizer::u64 m, __sanitizer::u64 si, __sanitizer::u64 sa,
               __sanitizer::u64 msi, __sanitizer::u64 msa)
      : mutex(m),
        src_idx(si),
        src_addr(sa),
        mutex_src_idx(msi),
        mutex_src_addr(msa) {}
};
static_assert(sizeof(CondWaitMeta) == 24, "ERROR: sizeof(CondWaitMeta) != 24");

struct CondSignalMeta {
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  CondSignalMeta(__sanitizer::u64 si, __sanitizer::u64 sa)
      : src_idx(si), src_addr(sa) {}
};
static_assert(sizeof(CondSignalMeta) == 8,
              "ERROR: sizeof(CondSignalMeta) != 8");

struct MemRangeMeta {
  __sanitizer::u64 src_idx : 16;
  __sanitizer::u64 src_addr : 48;
  MemRangeMeta(__sanitizer::u64 si, __sanitizer::u64 sa)
      : src_idx(si), src_addr(sa) {}
};

static_assert(sizeof(MemRangeMeta) == 8, "ERROR: sizeof(MemRangeMeta) != 8");

}  // namespace __trec_metadata

namespace __trec_header {
const char TREC_HEADER_VER[] = "20221206";
enum RecordType : __sanitizer::u32 {
  // Event type count
  PlainRead,
  PlainWrite,
  ZeroWrite,
  PtrRead,
  PtrWrite,
  PtrZeroWrite,
  Branch,
  FuncEnter,
  FuncExit,
  ThreadCreate,
  ThreadJoin,
  MutexLock,
  ReaderLock,
  MutexUnlock,
  ReaderUnlock,
  MemAlloc,
  MemFree,
  MemRangeRead,
  MemRangeWrite,
  CondWait,
  CondSignal,
  CondBroadcast,
  EventTypeCnt,

  // trace information
  Tid,
  TotalEventCnt,
  MetadataFileLen,
  DebugFileLen,
  ProcessFork,

  RecordTypeCnt,
};

struct TraceHeader {
  __sanitizer::u64 state[RecordType::RecordTypeCnt];
  char binary_path[512];
  char cmd[1024];
  char version[9];
  TraceHeader(__sanitizer::u64 tid) {
    __sanitizer::internal_memset(state, 0, sizeof(state));
    state[RecordType::Tid] = tid;
    __sanitizer::internal_memset(binary_path, 0, sizeof(binary_path));
    __sanitizer::internal_memset(cmd, 0, sizeof(cmd));
    __sanitizer::internal_strlcpy(version, TREC_HEADER_VER, sizeof(version));
  }
  void StateInc(RecordType type) { state[type] += 1; }
};
}  // namespace __trec_header

namespace __trec_debug_info {
const char TREC_DEBUG_VER[] = "20221206";
struct InstDebugInfo {
  __sanitizer::u32 line;
  __sanitizer::u16 column;

  // val_name_len, addr_name_len for read/write events
  // func_name_len,file_name_len for funcEntry events
  __sanitizer::u8 name_len[2];
  InstDebugInfo(__sanitizer::u32 _l, __sanitizer::u16 _c,
                __sanitizer::u8 name_len_1 = 0, __sanitizer::u8 name_len_2 = 0)
      : line(_l), column(_c) {
    name_len[0] = name_len_1;
    name_len[1] = name_len_2;
  }
};
static_assert(sizeof(InstDebugInfo) == 8, "ERROR: sizeof(InstDebugInfo)!=8");

}  // namespace __trec_debug_info

namespace __seqc_trace {
enum EventType : __sanitizer::u16 {
  FORK,
  JOIN,
  BEGIN,
  END,
  BRANCH,
  READ,
  WRITE,
  REQUEST,
  ACQUIRE,
  RELEASE
};

struct Event {
  EventType type;
  __sanitizer::u16 tid;
  __sanitizer::u32 eid, oid, tot, iid;
};
static_assert(sizeof(Event) == 20, "Error: sizeof(Event)!=20");
}  // namespace __seqc_trace

namespace __seqc_summary {
struct TraceInfo {
  __sanitizer::u32 tNum;
  __sanitizer::u32 mNum;
  __sanitizer::u32 lNum;
  __sanitizer::u32 totNum;
  __sanitizer::u32 brNum;
  __sanitizer::u32 rwNum;
  __sanitizer::u32 arNum;
};
static_assert(sizeof(TraceInfo) == 28, "Error: sizeof(TraceInfo)!=28");
}  // namespace __seqc_summary
#endif  // TREC_DEFS_H
