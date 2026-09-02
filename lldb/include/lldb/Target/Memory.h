//===-- Memory.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_MEMORY_H
#define LLDB_TARGET_MEMORY_H

#include "lldb/Utility/RangeMap.h"
#include "lldb/lldb-private.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace lldb_private {

/// A set of whole, aligned cache lines, keyed by line index.  A key names a
/// whole line, so no entry can be partial or unaligned and no length is
/// stored per entry.
class LineCache {
  using Collection = llvm::DenseMap<uint64_t, std::unique_ptr<uint8_t[]>>;

public:
  explicit LineCache(uint32_t line_byte_size)
      : m_line_byte_size(line_byte_size) {}

  uint32_t GetLineByteSize() const { return m_line_byte_size; }

  /// The cached bytes from \a addr to the end of the line holding it, empty if
  /// that line is not resident.
  llvm::ArrayRef<uint8_t> Lookup(lldb::addr_t addr) const;

  bool Holds(lldb::addr_t addr) const {
    return m_lines.contains(IndexOf(addr));
  }

  /// Add one whole line.  \a addr must be line aligned and \a src must hold a
  /// whole line.
  void Insert(lldb::addr_t addr, llvm::ArrayRef<uint8_t> src);

  /// Drop every line that intersects [addr, addr+size).
  void EraseRange(lldb::addr_t addr, lldb::addr_t size);

  void Clear(uint32_t new_line_byte_size) {
    m_lines.clear();
    m_line_byte_size = new_line_byte_size;
  }

  size_t GetSize() const { return m_lines.size(); }

  /// Iteration yields a line index and its bytes, in unspecified order.
  using const_iterator = Collection::const_iterator;
  const_iterator begin() const { return m_lines.begin(); }
  const_iterator end() const { return m_lines.end(); }

private:
  uint64_t IndexOf(lldb::addr_t addr) const { return addr / m_line_byte_size; }

  Collection m_lines;
  uint32_t m_line_byte_size;
};

/// A set of non-overlapping byte ranges at arbitrary addresses.  Lengths vary,
/// so every chunk carries its own.
class ChunkCache {
  using Collection = std::map<lldb::addr_t, std::vector<uint8_t>>;

public:
  /// The cached bytes from \a addr to the end of the chunk holding it, empty
  /// if no chunk holds it.
  llvm::ArrayRef<uint8_t> Lookup(lldb::addr_t addr) const;

  bool Holds(lldb::addr_t addr) const { return !Lookup(addr).empty(); }

  /// Add the bytes of [addr, addr+src.size()) that no chunk holds yet.  Bytes
  /// already held are kept: which read produced a byte does not matter.
  void InsertMissing(lldb::addr_t addr, llvm::ArrayRef<uint8_t> src);

  /// Drop every chunk that intersects [addr, addr+size).
  void EraseRange(lldb::addr_t addr, lldb::addr_t size);

  void Clear() { m_chunks.clear(); }

  size_t GetSize() const { return m_chunks.size(); }

  /// Iteration yields a chunk's start address and its bytes, in address order.
  using const_iterator = Collection::const_iterator;
  const_iterator begin() const { return m_chunks.begin(); }
  const_iterator end() const { return m_chunks.end(); }

private:
  /// The chunk holding \a addr, or end().  Chunks never overlap, so only the
  /// one starting at or below \a addr can hold it.
  Collection::const_iterator FindChunkContaining(lldb::addr_t addr) const;

  Collection m_chunks;
};

// A class to track memory that was read from a live process between
// runs.
class MemoryCache {
public:
  // Constructors and Destructors
  MemoryCache(Process &process);

  ~MemoryCache();

  void Clear(bool clear_invalid_ranges = false);

  void Flush(lldb::addr_t addr, size_t size);

  size_t Read(lldb::addr_t addr, void *dst, size_t dst_len, Status &error);

  /// Reads memory ranges, serving hits from the cache and batching misses
  /// through Process::DoReadMemoryRanges.  Matches Process::ReadMemoryRanges.
  llvm::SmallVector<llvm::MutableArrayRef<uint8_t>>
  ReadRanges(llvm::ArrayRef<Range<lldb::addr_t, size_t>> ranges,
             llvm::MutableArrayRef<uint8_t> buffer);

  uint32_t GetMemoryCacheLineSize() const {
    return m_L2_cache.GetLineByteSize();
  }

  void AddInvalidRange(lldb::addr_t base_addr, lldb::addr_t byte_size);

  bool RemoveInvalidRange(lldb::addr_t base_addr, lldb::addr_t byte_size);

  /// Allow external sources to populate data into the memory cache.
  void AddCacheData(lldb::addr_t addr, const void *src, size_t src_len);

  void AddCacheData(lldb::addr_t addr, llvm::ArrayRef<uint8_t> src) {
    if (!src.empty())
      AddCacheData(addr, src.data(), src.size());
  }

  void AddCacheData(lldb::addr_t addr,
                    const lldb::DataBufferSP &data_buffer_sp);

protected:
  typedef RangeVector<lldb::addr_t, lldb::addr_t, 4> InvalidRanges;
  typedef Range<lldb::addr_t, lldb::addr_t> AddrRange;
  // Classes that inherit from MemoryCache can see and modify these
  std::recursive_mutex m_mutex;
  // L1 and L2 partition the cache.  An address is held by at most one.  L2
  // holds whole, aligned lines; L1 holds smaller, non-overlapping pieces.
  ChunkCache m_L1_cache; // Chunks smaller than a cache line.
  LineCache m_L2_cache;  // Whole cache lines.
  InvalidRanges m_invalid_ranges;
  Process &m_process;

private:
  MemoryCache(const MemoryCache &) = delete;
  const MemoryCache &operator=(const MemoryCache &) = delete;

  // Add a whole cache line to L2 and drop the L1 entries it supersedes.
  // Caller must hold m_mutex.
  void InsertWholeLine(lldb::addr_t line_base_addr,
                       llvm::ArrayRef<uint8_t> src);

  // Add the bytes of [addr, addr+src.size()) that no entry holds yet to L1.
  // The range must lie within one cache line.  Caller must hold m_mutex.
  void InsertPartialLine(lldb::addr_t addr, llvm::ArrayRef<uint8_t> src);

  // Split [addr, addr+src.size()) at cache line boundaries: whole lines to L2,
  // shorter pieces to L1.  Takes m_mutex.
  void InsertData(lldb::addr_t addr, llvm::ArrayRef<uint8_t> src);

  // Copy the cached bytes of [addr, addr+len), stopping at the first miss,
  // and return the count.  Never reads from the inferior; caller holds m_mutex.
  size_t ReadFromCaches(lldb::addr_t addr, void *dst, size_t len) const;

  // The range to fetch for a read that ends at caller_end and whose first
  // bytes_filled bytes the caches supplied, so read_addr is the first byte
  // none of them holds.  Grown to whole cache lines where that costs nothing,
  // and clipped at an invalid range.  Caller must hold m_mutex.
  AddrRange GrowReadRange(lldb::addr_t read_addr, lldb::addr_t caller_end,
                          size_t bytes_filled) const;
};

    

class AllocatedBlock {
public:
  AllocatedBlock(lldb::addr_t addr, uint32_t byte_size, uint32_t permissions,
                 uint32_t chunk_size);

  ~AllocatedBlock();

  lldb::addr_t ReserveBlock(uint32_t size);

  bool FreeBlock(lldb::addr_t addr);

  lldb::addr_t GetBaseAddress() const { return m_range.GetRangeBase(); }

  uint32_t GetByteSize() const { return m_range.GetByteSize(); }

  uint32_t GetPermissions() const { return m_permissions; }

  uint32_t GetChunkSize() const { return m_chunk_size; }

  bool Contains(lldb::addr_t addr) const {
    return m_range.Contains(addr);
  }

protected:
  uint32_t TotalChunks() const { return GetByteSize() / GetChunkSize(); }

  uint32_t CalculateChunksNeededForSize(uint32_t size) const {
    return (size + m_chunk_size - 1) / m_chunk_size;
  }
  // Base address of this block of memory 4GB of chunk should be enough.
  Range<lldb::addr_t, uint32_t> m_range;
  // Permissions for this memory (logical OR of lldb::Permissions bits)
  const uint32_t m_permissions;
  // The size of chunks that the memory at m_addr is divied up into.
  const uint32_t m_chunk_size;
  // A sorted list of free address ranges.
  RangeVector<lldb::addr_t, uint32_t> m_free_blocks;
  // A sorted list of reserved address.
  RangeVector<lldb::addr_t, uint32_t> m_reserved_blocks;
};

// A class that can track allocated memory and give out allocated memory
// without us having to make an allocate/deallocate call every time we need
// some memory in a process that is being debugged.
class AllocatedMemoryCache {
public:
  // Constructors and Destructors
  AllocatedMemoryCache(Process &process);

  ~AllocatedMemoryCache();

  void Clear(bool deallocate_memory);

  lldb::addr_t AllocateMemory(size_t byte_size, uint32_t permissions,
                              Status &error);

  bool DeallocateMemory(lldb::addr_t ptr);

  bool IsInCache(lldb::addr_t addr) const;

protected:
  typedef std::shared_ptr<AllocatedBlock> AllocatedBlockSP;

  AllocatedBlockSP AllocatePage(uint32_t byte_size, uint32_t permissions,
                                uint32_t chunk_size, Status &error);

  // Classes that inherit from MemoryCache can see and modify these
  Process &m_process;
  mutable std::recursive_mutex m_mutex;
  typedef std::multimap<uint32_t, AllocatedBlockSP> PermissionsToBlockMap;
  PermissionsToBlockMap m_memory_map;

private:
  AllocatedMemoryCache(const AllocatedMemoryCache &) = delete;
  const AllocatedMemoryCache &operator=(const AllocatedMemoryCache &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_MEMORY_H
