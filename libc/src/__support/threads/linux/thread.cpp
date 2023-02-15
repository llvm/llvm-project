//===--- Implementation of a Linux thread class -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/thread.h"
#include "config/linux/app.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/CPP/stringstream.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/threads/linux/futex_word.h" // For FutexWordType

#ifdef LIBC_TARGET_ARCH_IS_AARCH64
#include <arm_acle.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <linux/futex.h>
#include <linux/prctl.h> // For PR_SET_NAME
#include <linux/sched.h> // For CLONE_* flags.
#include <stdint.h>
#include <sys/mman.h>    // For PROT_* and MAP_* definitions.
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

#ifdef SYS_mmap2
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "SYS_mmap or SYS_mmap2 not available on the target platform"
#endif

static constexpr size_t NAME_SIZE_MAX = 16; // Includes the null terminator
static constexpr size_t DEFAULT_STACK_SIZE = (1 << 16); // 64KB
static constexpr uint32_t CLEAR_TID_VALUE = 0xABCD1234;
static constexpr unsigned CLONE_SYSCALL_FLAGS =
    CLONE_VM        // Share the memory space with the parent.
    | CLONE_FS      // Share the file system with the parent.
    | CLONE_FILES   // Share the files with the parent.
    | CLONE_SIGHAND // Share the signal handlers with the parent.
    | CLONE_THREAD  // Same thread group as the parent.
    | CLONE_SYSVSEM // Share a single list of System V semaphore adjustment
                    // values
    | CLONE_PARENT_SETTID  // Set child thread ID in |ptid| of the parent.
    | CLONE_CHILD_CLEARTID // Let the kernel clear the tid address
                           // wake the joining thread.
    | CLONE_SETTLS;        // Setup the thread pointer of the new thread.

LIBC_INLINE ErrorOr<void *> alloc_stack(size_t size) {
  long mmap_result =
      __llvm_libc::syscall_impl(MMAP_SYSCALL_NUMBER,
                                0, // No special address
                                size,
                                PROT_READ | PROT_WRITE, // Read and write stack
                                MAP_ANONYMOUS | MAP_PRIVATE, // Process private
                                -1, // Not backed by any file
                                0   // No offset
      );
  if (mmap_result < 0 && (uintptr_t(mmap_result) >= UINTPTR_MAX - size))
    return Error{int(-mmap_result)};
  return reinterpret_cast<void *>(mmap_result);
}

LIBC_INLINE void free_stack(void *stack, size_t size) {
  __llvm_libc::syscall_impl(SYS_munmap, stack, size);
}

struct Thread;

// We align the start args to 16-byte boundary as we adjust the allocated
// stack memory with its size. We want the adjusted address to be at a
// 16-byte boundary to satisfy the x86_64 and aarch64 ABI requirements.
// If different architecture in future requires higher alignment, then we
// can add a platform specific alignment spec.
struct alignas(STACK_ALIGNMENT) StartArgs {
  ThreadAttributes *thread_attrib;
  ThreadRunner runner;
  void *arg;
};

static void cleanup_thread_resources(ThreadAttributes *attrib) {
  // Cleanup the TLS before the stack as the TLS information is stored on
  // the stack.
  cleanup_tls(attrib->tls, attrib->tls_size);
  if (attrib->owned_stack)
    free_stack(attrib->stack, attrib->stack_size);
}

__attribute__((always_inline)) inline uintptr_t get_start_args_addr() {
// NOTE: For __builtin_frame_address to work reliably across compilers,
// architectures and various optimization levels, the TU including this file
// should be compiled with -fno-omit-frame-pointer.
#ifdef LIBC_TARGET_ARCH_IS_X86_64
  return reinterpret_cast<uintptr_t>(__builtin_frame_address(0))
         // The x86_64 call instruction pushes resume address on to the stack.
         // Next, The x86_64 SysV ABI requires that the frame pointer be pushed
         // on to the stack. So, we have to step past two 64-bit values to get
         // to the start args.
         + sizeof(uintptr_t) * 2;
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  // The frame pointer after cloning the new thread in the Thread::run method
  // is set to the stack pointer where start args are stored. So, we fetch
  // from there.
  return reinterpret_cast<uintptr_t>(__builtin_frame_address(1));
#endif
}

__attribute__((noinline)) static void start_thread() {
  auto *start_args = reinterpret_cast<StartArgs *>(get_start_args_addr());
  auto *attrib = start_args->thread_attrib;
  self.attrib = attrib;
  self.attrib->atexit_callback_mgr = internal::get_thread_atexit_callback_mgr();

  if (attrib->style == ThreadStyle::POSIX) {
    attrib->retval.posix_retval =
        start_args->runner.posix_runner(start_args->arg);
    thread_exit(ThreadReturnValue(attrib->retval.posix_retval),
                ThreadStyle::POSIX);
  } else {
    attrib->retval.stdc_retval =
        start_args->runner.stdc_runner(start_args->arg);
    thread_exit(ThreadReturnValue(attrib->retval.stdc_retval),
                ThreadStyle::STDC);
  }
}

int Thread::run(ThreadStyle style, ThreadRunner runner, void *arg, void *stack,
                size_t size, bool detached) {
  bool owned_stack = false;
  if (stack == nullptr) {
    if (size == 0)
      size = DEFAULT_STACK_SIZE;
    auto alloc = alloc_stack(size);
    if (!alloc)
      return alloc.error();
    else
      stack = alloc.value();
    owned_stack = true;
  }

  TLSDescriptor tls;
  init_tls(tls);

  // When the new thread is spawned by the kernel, the new thread gets the
  // stack we pass to the clone syscall. However, this stack is empty and does
  // not have any local vars present in this function. Hence, one cannot
  // pass arguments to the thread start function, or use any local vars from
  // here. So, we pack them into the new stack from where the thread can sniff
  // them out.
  //
  // Likewise, the actual thread state information is also stored on the
  // stack memory.
  uintptr_t adjusted_stack = reinterpret_cast<uintptr_t>(stack) + size -
                             sizeof(StartArgs) - sizeof(ThreadAttributes) -
                             sizeof(cpp::Atomic<FutexWordType>);
  adjusted_stack &= ~(uintptr_t(STACK_ALIGNMENT) - 1);

  auto *start_args = reinterpret_cast<StartArgs *>(adjusted_stack);

  attrib =
      reinterpret_cast<ThreadAttributes *>(adjusted_stack + sizeof(StartArgs));
  attrib->style = style;
  attrib->detach_state =
      uint32_t(detached ? DetachState::DETACHED : DetachState::JOINABLE);
  attrib->stack = stack;
  attrib->stack_size = size;
  attrib->owned_stack = owned_stack;
  attrib->tls = tls.addr;
  attrib->tls_size = tls.size;

  start_args->thread_attrib = attrib;
  start_args->runner = runner;
  start_args->arg = arg;

  auto clear_tid = reinterpret_cast<cpp::Atomic<FutexWordType> *>(
      adjusted_stack + sizeof(StartArgs) + sizeof(ThreadAttributes));
  clear_tid->val = CLEAR_TID_VALUE;
  attrib->platform_data = clear_tid;

  // The clone syscall takes arguments in an architecture specific order.
  // Also, we want the result of the syscall to be in a register as the child
  // thread gets a completely different stack after it is created. The stack
  // variables from this function will not be availalbe to the child thread.
#ifdef LIBC_TARGET_ARCH_IS_X86_64
  long register clone_result asm("rax");
  clone_result = __llvm_libc::syscall_impl(
      SYS_clone, CLONE_SYSCALL_FLAGS, adjusted_stack,
      &attrib->tid,    // The address where the child tid is written
      &clear_tid->val, // The futex where the child thread status is signalled
      tls.tp           // The thread pointer value for the new thread.
  );
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  long register clone_result asm("x0");
  clone_result = __llvm_libc::syscall_impl(
      SYS_clone, CLONE_SYSCALL_FLAGS, adjusted_stack,
      &attrib->tid,   // The address where the child tid is written
      tls.tp,         // The thread pointer value for the new thread.
      &clear_tid->val // The futex where the child thread status is signalled
  );
#else
#error "Unsupported architecture for the clone syscall."
#endif

  if (clone_result == 0) {
#ifdef LIBC_TARGET_ARCH_IS_AARCH64
    // We set the frame pointer to be the same as the "sp" so that start args
    // can be sniffed out from start_thread.
    __arm_wsr64("x29", __arm_rsr64("sp"));
#endif
    start_thread();
  } else if (clone_result < 0) {
    cleanup_thread_resources(attrib);
    return -clone_result;
  }

  return 0;
}

int Thread::join(ThreadReturnValue &retval) {
  wait();

  if (attrib->style == ThreadStyle::POSIX)
    retval.posix_retval = attrib->retval.posix_retval;
  else
    retval.stdc_retval = attrib->retval.stdc_retval;

  cleanup_thread_resources(attrib);

  return 0;
}

int Thread::detach() {
  uint32_t joinable_state = uint32_t(DetachState::JOINABLE);
  if (attrib->detach_state.compare_exchange_strong(
          joinable_state, uint32_t(DetachState::DETACHED))) {
    return int(DetachType::SIMPLE);
  }

  // If the thread was already detached, then the detach method should not
  // be called at all. If the thread is exiting, then we wait for it to exit
  // and free up resources.
  wait();

  cleanup_thread_resources(attrib);

  return int(DetachType::CLEANUP);
}

void Thread::wait() {
  // The kernel should set the value at the clear tid address to zero.
  // If not, it is a spurious wake and we should continue to wait on
  // the futex.
  auto *clear_tid =
      reinterpret_cast<cpp::Atomic<FutexWordType> *>(attrib->platform_data);
  while (clear_tid->load() != 0) {
    // We cannot do a FUTEX_WAIT_PRIVATE here as the kernel does a
    // FUTEX_WAKE and not a FUTEX_WAKE_PRIVATE.
    __llvm_libc::syscall_impl(SYS_futex, &clear_tid->val, FUTEX_WAIT,
                              CLEAR_TID_VALUE, nullptr);
  }
}

bool Thread::operator==(const Thread &thread) const {
  return attrib->tid == thread.attrib->tid;
}

static constexpr cpp::string_view THREAD_NAME_PATH_PREFIX("/proc/self/task/");
static constexpr size_t THREAD_NAME_PATH_SIZE =
    THREAD_NAME_PATH_PREFIX.size() +
    IntegerToString::dec_bufsize<int>() + // Size of tid
    1 +                                   // For '/' character
    5; // For the file name "comm" and the nullterminator.

static void construct_thread_name_file_path(cpp::StringStream &stream,
                                            int tid) {
  stream << THREAD_NAME_PATH_PREFIX << tid << '/' << cpp::string_view("comm")
         << cpp::StringStream::ENDS;
}

int Thread::set_name(const cpp::string_view &name) {
  if (name.size() >= NAME_SIZE_MAX)
    return ERANGE;

  if (*this == self) {
    // If we are setting the name of the current thread, then we can
    // use the syscall to set the name.
    int retval = __llvm_libc::syscall_impl(SYS_prctl, PR_SET_NAME, name.data());
    if (retval < 0)
      return -retval;
    else
      return 0;
  }

  char path_name_buffer[THREAD_NAME_PATH_SIZE];
  cpp::StringStream path_stream(path_name_buffer);
  construct_thread_name_file_path(path_stream, attrib->tid);
#ifdef SYS_open
  int fd = __llvm_libc::syscall_impl(SYS_open, path_name_buffer, O_RDWR);
#else
  int fd =
      __llvm_libc::syscall_impl(SYS_openat, AT_FDCWD, path_name_buffer, O_RDWR);
#endif
  if (fd < 0)
    return -fd;

  int retval =
      __llvm_libc::syscall_impl(SYS_write, fd, name.data(), name.size());
  __llvm_libc::syscall_impl(SYS_close, fd);

  if (retval < 0)
    return -retval;
  else if (retval != int(name.size()))
    return EIO;
  else
    return 0;
}

int Thread::get_name(cpp::StringStream &name) const {
  if (name.bufsize() < NAME_SIZE_MAX)
    return ERANGE;

  char name_buffer[NAME_SIZE_MAX];

  if (*this == self) {
    // If we are getting the name of the current thread, then we can
    // use the syscall to get the name.
    int retval = __llvm_libc::syscall_impl(SYS_prctl, PR_GET_NAME, name_buffer);
    if (retval < 0)
      return -retval;
    name << name_buffer;
    return 0;
  }

  char path_name_buffer[THREAD_NAME_PATH_SIZE];
  cpp::StringStream path_stream(path_name_buffer);
  construct_thread_name_file_path(path_stream, attrib->tid);
#ifdef SYS_open
  int fd = __llvm_libc::syscall_impl(SYS_open, path_name_buffer, O_RDONLY);
#else
  int fd = __llvm_libc::syscall_impl(SYS_openat, AT_FDCWD, path_name_buffer,
                                     O_RDONLY);
#endif
  if (fd < 0)
    return -fd;

  int retval =
      __llvm_libc::syscall_impl(SYS_read, fd, name_buffer, NAME_SIZE_MAX);
  __llvm_libc::syscall_impl(SYS_close, fd);
  if (retval < 0)
    return -retval;
  if (retval == NAME_SIZE_MAX)
    return ERANGE;
  if (name_buffer[retval - 1] == '\n')
    name_buffer[retval - 1] = '\0';
  else
    name_buffer[retval] = '\0';
  name << name_buffer;
  return 0;
}

void thread_exit(ThreadReturnValue retval, ThreadStyle style) {
  auto attrib = self.attrib;

  // The very first thing we do is to call the thread's atexit callbacks.
  // These callbacks could be the ones registered by the language runtimes,
  // for example, the destructors of thread local objects. They can also
  // be destructors of the TSS objects set using API like pthread_setspecific.
  // NOTE: We cannot call the atexit callbacks as part of the
  // cleanup_thread_resources function as that function can be called from a
  // different thread. The destructors of thread local and TSS objects should
  // be called by the thread which owns them.
  internal::call_atexit_callbacks(attrib);

  uint32_t joinable_state = uint32_t(DetachState::JOINABLE);
  if (!attrib->detach_state.compare_exchange_strong(
          joinable_state, uint32_t(DetachState::EXITING))) {
    // Thread is detached so cleanup the resources.
    cleanup_thread_resources(attrib);

    // Set the CLEAR_TID address to nullptr to prevent the kernel
    // from signalling at a non-existent futex location.
    __llvm_libc::syscall_impl(SYS_set_tid_address, 0);
  }

  if (style == ThreadStyle::POSIX)
    __llvm_libc::syscall_impl(SYS_exit, retval.posix_retval);
  else
    __llvm_libc::syscall_impl(SYS_exit, retval.stdc_retval);
}

} // namespace __llvm_libc
