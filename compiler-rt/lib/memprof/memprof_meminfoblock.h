#ifndef MEMPROF_MEMINFOBLOCK_H_
#define MEMPROF_MEMINFOBLOCK_H_

#include "memprof_interface_internal.h" // For u32, u64 TODO: Move these out of the internal header.
#include "sanitizer_common/sanitizer_common.h"

namespace __memprof {

using __sanitizer::Printf;

struct MemInfoBlock {
  u32 alloc_count;
  u64 total_access_count, min_access_count, max_access_count;
  u64 total_size;
  u32 min_size, max_size;
  u32 alloc_timestamp, dealloc_timestamp;
  u64 total_lifetime;
  u32 min_lifetime, max_lifetime;
  u32 alloc_cpu_id, dealloc_cpu_id;
  u32 num_migrated_cpu;

  // Only compared to prior deallocated object currently.
  u32 num_lifetime_overlaps;
  u32 num_same_alloc_cpu;
  u32 num_same_dealloc_cpu;

  u64 data_type_id; // TODO: hash of type name

  MemInfoBlock() : alloc_count(0) {}

  MemInfoBlock(u32 size, u64 access_count, u32 alloc_timestamp,
               u32 dealloc_timestamp, u32 alloc_cpu, u32 dealloc_cpu)
      : alloc_count(1), total_access_count(access_count),
        min_access_count(access_count), max_access_count(access_count),
        total_size(size), min_size(size), max_size(size),
        alloc_timestamp(alloc_timestamp), dealloc_timestamp(dealloc_timestamp),
        total_lifetime(dealloc_timestamp - alloc_timestamp),
        min_lifetime(total_lifetime), max_lifetime(total_lifetime),
        alloc_cpu_id(alloc_cpu), dealloc_cpu_id(dealloc_cpu),
        num_lifetime_overlaps(0), num_same_alloc_cpu(0),
        num_same_dealloc_cpu(0) {
    num_migrated_cpu = alloc_cpu_id != dealloc_cpu_id;
  }

  void Print(u64 id, bool print_terse) const {
    u64 p;

    if (print_terse) {
      p = total_size * 100 / alloc_count;
      Printf("MIB:%llu/%u/%llu.%02llu/%u/%u/", id, alloc_count, p / 100,
             p % 100, min_size, max_size);
      p = total_access_count * 100 / alloc_count;
      Printf("%llu.%02llu/%llu/%llu/", p / 100, p % 100, min_access_count,
             max_access_count);
      p = total_lifetime * 100 / alloc_count;
      Printf("%llu.%02llu/%u/%u/", p / 100, p % 100, min_lifetime,
             max_lifetime);
      Printf("%u/%u/%u/%u\n", num_migrated_cpu, num_lifetime_overlaps,
             num_same_alloc_cpu, num_same_dealloc_cpu);
    } else {
      p = total_size * 100 / alloc_count;
      Printf("Memory allocation stack id = %llu\n", id);
      Printf("\talloc_count %u, size (ave/min/max) %llu.%02llu / %u / %u\n",
             alloc_count, p / 100, p % 100, min_size, max_size);
      p = total_access_count * 100 / alloc_count;
      Printf("\taccess_count (ave/min/max): %llu.%02llu / %llu / %llu\n",
             p / 100, p % 100, min_access_count, max_access_count);
      p = total_lifetime * 100 / alloc_count;
      Printf("\tlifetime (ave/min/max): %llu.%02llu / %u / %u\n", p / 100,
             p % 100, min_lifetime, max_lifetime);
      Printf("\tnum migrated: %u, num lifetime overlaps: %u, num same alloc "
             "cpu: %u, num same dealloc_cpu: %u\n",
             num_migrated_cpu, num_lifetime_overlaps, num_same_alloc_cpu,
             num_same_dealloc_cpu);
    }
  }

  static void printHeader() {
    Printf("MIB:StackID/AllocCount/AveSize/MinSize/MaxSize/AveAccessCount/"
           "MinAccessCount/MaxAccessCount/AveLifetime/MinLifetime/MaxLifetime/"
           "NumMigratedCpu/NumLifetimeOverlaps/NumSameAllocCpu/"
           "NumSameDeallocCpu\n");
  }

  void Merge(const MemInfoBlock &newMIB) {
    alloc_count += newMIB.alloc_count;

    total_access_count += newMIB.total_access_count;
    min_access_count = Min(min_access_count, newMIB.min_access_count);
    max_access_count = Max(max_access_count, newMIB.max_access_count);

    total_size += newMIB.total_size;
    min_size = Min(min_size, newMIB.min_size);
    max_size = Max(max_size, newMIB.max_size);

    total_lifetime += newMIB.total_lifetime;
    min_lifetime = Min(min_lifetime, newMIB.min_lifetime);
    max_lifetime = Max(max_lifetime, newMIB.max_lifetime);

    // We know newMIB was deallocated later, so just need to check if it was
    // allocated before last one deallocated.
    num_lifetime_overlaps += newMIB.alloc_timestamp < dealloc_timestamp;
    alloc_timestamp = newMIB.alloc_timestamp;
    dealloc_timestamp = newMIB.dealloc_timestamp;

    num_same_alloc_cpu += alloc_cpu_id == newMIB.alloc_cpu_id;
    num_same_dealloc_cpu += dealloc_cpu_id == newMIB.dealloc_cpu_id;
    alloc_cpu_id = newMIB.alloc_cpu_id;
    dealloc_cpu_id = newMIB.dealloc_cpu_id;
  }

} __attribute__((packed));

} // namespace __memprof

#endif // MEMPROF_MEMINFOBLOCK_H_
