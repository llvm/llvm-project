//===-- sanitizer_common.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
// It declares common functions and classes that are used in both runtimes.
// Implementation of some functions are provided in sanitizer_common, while
// others must be defined by run-time library itself.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_COMMON_H
#define SANITIZER_COMMON_H

#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_mutex.h"

namespace __sanitizer {
struct StackTrace;

// Constants.
const uptr kWordSize = SANITIZER_WORDSIZE / 8;
const uptr kWordSizeInBits = 8 * kWordSize;

#if defined(__powerpc__) || defined(__powerpc64__)
const uptr kCacheLineSize = 128;
#else
const uptr kCacheLineSize = 64;
#endif

const uptr kMaxPathLength = 512;

extern const char *SanitizerToolName;  // Can be changed by the tool.

uptr GetPageSize();
uptr GetPageSizeCached();
uptr GetMmapGranularity();
uptr GetMaxVirtualAddress();
// Threads
uptr GetTid();
uptr GetThreadSelf();
void GetThreadStackTopAndBottom(bool at_initialization, uptr *stack_top,
                                uptr *stack_bottom);
void GetThreadStackAndTls(bool main, uptr *stk_addr, uptr *stk_size,
                          uptr *tls_addr, uptr *tls_size);

// Memory management
void *MmapOrDie(uptr size, const char *mem_type);
void UnmapOrDie(void *addr, uptr size);
void *MmapFixedNoReserve(uptr fixed_addr, uptr size);
void *MmapFixedOrDie(uptr fixed_addr, uptr size);
void *Mprotect(uptr fixed_addr, uptr size);
// Map aligned chunk of address space; size and alignment are powers of two.
void *MmapAlignedOrDie(uptr size, uptr alignment, const char *mem_type);
// Used to check if we can map shadow memory to a fixed location.
bool MemoryRangeIsAvailable(uptr range_start, uptr range_end);
void FlushUnneededShadowMemory(uptr addr, uptr size);

// InternalScopedBuffer can be used instead of large stack arrays to
// keep frame size low.
// FIXME: use InternalAlloc instead of MmapOrDie once
// InternalAlloc is made libc-free.
template<typename T>
class InternalScopedBuffer {
 public:
  explicit InternalScopedBuffer(uptr cnt) {
    cnt_ = cnt;
    ptr_ = (T*)MmapOrDie(cnt * sizeof(T), "InternalScopedBuffer");
  }
  ~InternalScopedBuffer() {
    UnmapOrDie(ptr_, cnt_ * sizeof(T));
  }
  T &operator[](uptr i) { return ptr_[i]; }
  T *data() { return ptr_; }
  uptr size() { return cnt_ * sizeof(T); }

 private:
  T *ptr_;
  uptr cnt_;
  // Disallow evil constructors.
  InternalScopedBuffer(const InternalScopedBuffer&);
  void operator=(const InternalScopedBuffer&);
};

class InternalScopedString : public InternalScopedBuffer<char> {
 public:
  explicit InternalScopedString(uptr max_length)
      : InternalScopedBuffer<char>(max_length), length_(0) {
    (*this)[0] = '\0';
  }
  uptr length() { return length_; }
  void clear() {
    (*this)[0] = '\0';
    length_ = 0;
  }
  void append(const char *format, ...);

 private:
  uptr length_;
};

// Simple low-level (mmap-based) allocator for internal use. Doesn't have
// constructor, so all instances of LowLevelAllocator should be
// linker initialized.
class LowLevelAllocator {
 public:
  // Requires an external lock.
  void *Allocate(uptr size);
 private:
  char *allocated_end_;
  char *allocated_current_;
};
typedef void (*LowLevelAllocateCallback)(uptr ptr, uptr size);
// Allows to register tool-specific callbacks for LowLevelAllocator.
// Passing NULL removes the callback.
void SetLowLevelAllocateCallback(LowLevelAllocateCallback callback);

// IO
void RawWrite(const char *buffer);
bool PrintsToTty();
// Caching version of PrintsToTty(). Not thread-safe.
bool PrintsToTtyCached();
void Printf(const char *format, ...);
void Report(const char *format, ...);
void SetPrintfAndReportCallback(void (*callback)(const char *));

// Can be used to prevent mixing error reports from different sanitizers.
extern StaticSpinMutex CommonSanitizerReportMutex;
void MaybeOpenReportFile();
extern fd_t report_fd;
extern bool log_to_file;
extern char report_path_prefix[4096];
extern uptr report_fd_pid;

uptr OpenFile(const char *filename, bool write);
// Opens the file 'file_name" and reads up to 'max_len' bytes.
// The resulting buffer is mmaped and stored in '*buff'.
// The size of the mmaped region is stored in '*buff_size',
// Returns the number of read bytes or 0 if file can not be opened.
uptr ReadFileToBuffer(const char *file_name, char **buff,
                      uptr *buff_size, uptr max_len);
// Maps given file to virtual memory, and returns pointer to it
// (or NULL if the mapping failes). Stores the size of mmaped region
// in '*buff_size'.
void *MapFileToMemory(const char *file_name, uptr *buff_size);

// Error report formatting.
const char *StripPathPrefix(const char *filepath,
                            const char *strip_file_prefix);
void PrintSourceLocation(InternalScopedString *buffer, const char *file,
                         int line, int column);
void PrintModuleAndOffset(InternalScopedString *buffer,
                          const char *module, uptr offset);

// OS
void DisableCoreDumper();
void DumpProcessMap();
bool FileExists(const char *filename);
const char *GetEnv(const char *name);
bool SetEnv(const char *name, const char *value);
const char *GetPwd();
char *FindPathToBinary(const char *name);
u32 GetUid();
void ReExec();
bool StackSizeIsUnlimited();
void SetStackSizeLimitInBytes(uptr limit);
void PrepareForSandboxing();

void InitTlsSize();
uptr GetTlsSize();

// Other
void SleepForSeconds(int seconds);
void SleepForMillis(int millis);
u64 NanoTime();
int Atexit(void (*function)(void));
void SortArray(uptr *array, uptr size);
// Strip the directories from the module name, return a new string allocated
// with internal_strdup.
char *StripModuleName(const char *module);

// Exit
void NORETURN Abort();
void NORETURN Die();
void NORETURN
CheckFailed(const char *file, int line, const char *cond, u64 v1, u64 v2);

// Set the name of the current thread to 'name', return true on succees.
// The name may be truncated to a system-dependent limit.
bool SanitizerSetThreadName(const char *name);
// Get the name of the current thread (no more than max_len bytes),
// return true on succees. name should have space for at least max_len+1 bytes.
bool SanitizerGetThreadName(char *name, int max_len);

// Specific tools may override behavior of "Die" and "CheckFailed" functions
// to do tool-specific job.
typedef void (*DieCallbackType)(void);
void SetDieCallback(DieCallbackType);
DieCallbackType GetDieCallback();
typedef void (*CheckFailedCallbackType)(const char *, int, const char *,
                                       u64, u64);
void SetCheckFailedCallback(CheckFailedCallbackType callback);

// We don't want a summary too long.
const int kMaxSummaryLength = 1024;
// Construct a one-line string:
//   SUMMARY: SanitizerToolName: error_message
// and pass it to __sanitizer_report_error_summary.
void ReportErrorSummary(const char *error_message);
// Same as above, but construct error_message as:
//   error_type: file:line function
void ReportErrorSummary(const char *error_type, const char *file,
                        int line, const char *function);
void ReportErrorSummary(const char *error_type, StackTrace *trace);

// Math
#if SANITIZER_WINDOWS && !defined(__clang__) && !defined(__GNUC__)
extern "C" {
unsigned char _BitScanForward(unsigned long *index, unsigned long mask);  // NOLINT
unsigned char _BitScanReverse(unsigned long *index, unsigned long mask);  // NOLINT
#if defined(_WIN64)
unsigned char _BitScanForward64(unsigned long *index, unsigned __int64 mask);  // NOLINT
unsigned char _BitScanReverse64(unsigned long *index, unsigned __int64 mask);  // NOLINT
#endif
}
#endif

INLINE uptr MostSignificantSetBitIndex(uptr x) {
  CHECK_NE(x, 0U);
  unsigned long up;  // NOLINT
#if !SANITIZER_WINDOWS || defined(__clang__) || defined(__GNUC__)
  up = SANITIZER_WORDSIZE - 1 - __builtin_clzl(x);
#elif defined(_WIN64)
  _BitScanReverse64(&up, x);
#else
  _BitScanReverse(&up, x);
#endif
  return up;
}

INLINE bool IsPowerOfTwo(uptr x) {
  return (x & (x - 1)) == 0;
}

INLINE uptr RoundUpToPowerOfTwo(uptr size) {
  CHECK(size);
  if (IsPowerOfTwo(size)) return size;

  uptr up = MostSignificantSetBitIndex(size);
  CHECK(size < (1ULL << (up + 1)));
  CHECK(size > (1ULL << up));
  return 1UL << (up + 1);
}

INLINE uptr RoundUpTo(uptr size, uptr boundary) {
  CHECK(IsPowerOfTwo(boundary));
  return (size + boundary - 1) & ~(boundary - 1);
}

INLINE uptr RoundDownTo(uptr x, uptr boundary) {
  return x & ~(boundary - 1);
}

INLINE bool IsAligned(uptr a, uptr alignment) {
  return (a & (alignment - 1)) == 0;
}

INLINE uptr Log2(uptr x) {
  CHECK(IsPowerOfTwo(x));
#if !SANITIZER_WINDOWS || defined(__clang__) || defined(__GNUC__)
  return __builtin_ctzl(x);
#elif defined(_WIN64)
  unsigned long ret;  // NOLINT
  _BitScanForward64(&ret, x);
  return ret;
#else
  unsigned long ret;  // NOLINT
  _BitScanForward(&ret, x);
  return ret;
#endif
}

// Don't use std::min, std::max or std::swap, to minimize dependency
// on libstdc++.
template<class T> T Min(T a, T b) { return a < b ? a : b; }
template<class T> T Max(T a, T b) { return a > b ? a : b; }
template<class T> void Swap(T& a, T& b) {
  T tmp = a;
  a = b;
  b = tmp;
}

// Char handling
INLINE bool IsSpace(int c) {
  return (c == ' ') || (c == '\n') || (c == '\t') ||
         (c == '\f') || (c == '\r') || (c == '\v');
}
INLINE bool IsDigit(int c) {
  return (c >= '0') && (c <= '9');
}
INLINE int ToLower(int c) {
  return (c >= 'A' && c <= 'Z') ? (c + 'a' - 'A') : c;
}

#if SANITIZER_WORDSIZE == 64
# define FIRST_32_SECOND_64(a, b) (b)
#else
# define FIRST_32_SECOND_64(a, b) (a)
#endif

// A low-level vector based on mmap. May incur a significant memory overhead for
// small vectors.
// WARNING: The current implementation supports only POD types.
template<typename T>
class InternalMmapVector {
 public:
  explicit InternalMmapVector(uptr initial_capacity) {
    CHECK_GT(initial_capacity, 0);
    capacity_ = initial_capacity;
    size_ = 0;
    data_ = (T *)MmapOrDie(capacity_ * sizeof(T), "InternalMmapVector");
  }
  ~InternalMmapVector() {
    UnmapOrDie(data_, capacity_ * sizeof(T));
  }
  T &operator[](uptr i) {
    CHECK_LT(i, size_);
    return data_[i];
  }
  const T &operator[](uptr i) const {
    CHECK_LT(i, size_);
    return data_[i];
  }
  void push_back(const T &element) {
    CHECK_LE(size_, capacity_);
    if (size_ == capacity_) {
      uptr new_capacity = RoundUpToPowerOfTwo(size_ + 1);
      Resize(new_capacity);
    }
    data_[size_++] = element;
  }
  T &back() {
    CHECK_GT(size_, 0);
    return data_[size_ - 1];
  }
  void pop_back() {
    CHECK_GT(size_, 0);
    size_--;
  }
  uptr size() const {
    return size_;
  }
  const T *data() const {
    return data_;
  }
  uptr capacity() const {
    return capacity_;
  }

  void clear() { size_ = 0; }

 private:
  void Resize(uptr new_capacity) {
    CHECK_GT(new_capacity, 0);
    CHECK_LE(size_, new_capacity);
    T *new_data = (T *)MmapOrDie(new_capacity * sizeof(T),
                                 "InternalMmapVector");
    internal_memcpy(new_data, data_, size_ * sizeof(T));
    T *old_data = data_;
    data_ = new_data;
    UnmapOrDie(old_data, capacity_ * sizeof(T));
    capacity_ = new_capacity;
  }
  // Disallow evil constructors.
  InternalMmapVector(const InternalMmapVector&);
  void operator=(const InternalMmapVector&);

  T *data_;
  uptr capacity_;
  uptr size_;
};

// HeapSort for arrays and InternalMmapVector.
template<class Container, class Compare>
void InternalSort(Container *v, uptr size, Compare comp) {
  if (size < 2)
    return;
  // Stage 1: insert elements to the heap.
  for (uptr i = 1; i < size; i++) {
    uptr j, p;
    for (j = i; j > 0; j = p) {
      p = (j - 1) / 2;
      if (comp((*v)[p], (*v)[j]))
        Swap((*v)[j], (*v)[p]);
      else
        break;
    }
  }
  // Stage 2: swap largest element with the last one,
  // and sink the new top.
  for (uptr i = size - 1; i > 0; i--) {
    Swap((*v)[0], (*v)[i]);
    uptr j, max_ind;
    for (j = 0; j < i; j = max_ind) {
      uptr left = 2 * j + 1;
      uptr right = 2 * j + 2;
      max_ind = j;
      if (left < i && comp((*v)[max_ind], (*v)[left]))
        max_ind = left;
      if (right < i && comp((*v)[max_ind], (*v)[right]))
        max_ind = right;
      if (max_ind != j)
        Swap((*v)[j], (*v)[max_ind]);
      else
        break;
    }
  }
}

template<class Container, class Value, class Compare>
uptr InternalBinarySearch(const Container &v, uptr first, uptr last,
                          const Value &val, Compare comp) {
  uptr not_found = last + 1;
  while (last >= first) {
    uptr mid = (first + last) / 2;
    if (comp(v[mid], val))
      first = mid + 1;
    else if (comp(val, v[mid]))
      last = mid - 1;
    else
      return mid;
  }
  return not_found;
}

// Represents a binary loaded into virtual memory (e.g. this can be an
// executable or a shared object).
class LoadedModule {
 public:
  LoadedModule(const char *module_name, uptr base_address);
  void addAddressRange(uptr beg, uptr end);
  bool containsAddress(uptr address) const;

  const char *full_name() const { return full_name_; }
  uptr base_address() const { return base_address_; }

 private:
  struct AddressRange {
    uptr beg;
    uptr end;
  };
  char *full_name_;
  uptr base_address_;
  static const uptr kMaxNumberOfAddressRanges = 6;
  AddressRange ranges_[kMaxNumberOfAddressRanges];
  uptr n_ranges_;
};

// OS-dependent function that fills array with descriptions of at most
// "max_modules" currently loaded modules. Returns the number of
// initialized modules. If filter is nonzero, ignores modules for which
// filter(full_name) is false.
typedef bool (*string_predicate_t)(const char *);
uptr GetListOfModules(LoadedModule *modules, uptr max_modules,
                      string_predicate_t filter);

#if SANITIZER_POSIX
const uptr kPthreadDestructorIterations = 4;
#else
// Unused on Windows.
const uptr kPthreadDestructorIterations = 0;
#endif

// Callback type for iterating over a set of memory ranges.
typedef void (*RangeIteratorCallback)(uptr begin, uptr end, void *arg);
}  // namespace __sanitizer

inline void *operator new(__sanitizer::operator_new_size_type size,
                          __sanitizer::LowLevelAllocator &alloc) {
  return alloc.Allocate(size);
}

#endif  // SANITIZER_COMMON_H
