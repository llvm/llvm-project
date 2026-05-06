#include "ubsan_minimal_common.h"

#define __gpu_constant __attribute__((opencl_constant))

using size_t3 = size_t [[clang::ext_vector_type(3)]];
struct __ubsan_abort_info_t {
  __gpu_constant const char *fmt;
  const char *kind;
  uintptr_t caller;
  size_t3 gid;
  size_t3 lid;
};

// OpenCL printf maps to OpExtInst printf (OpenCL extended instruction set).
extern "C" int printf(__gpu_constant const char *fmt, ...);

// SPV_KHR_abort: OpAbortKHR terminates the invocation and passes a message
// to the client API. The message is passed as a typed value.
[[noreturn]] void __spirv_AbortKHR(__ubsan_abort_info_t info);

static __gpu_constant const char ubsan_msg_simple[] = "%s";
static __gpu_constant const char ubsan_msg_fmt[] =
    "ubsan: %s by 0x%lx at gid=[%v3lu] lid=[%v3lu]\n";

void __ubsan_message(const char *msg) { printf(ubsan_msg_simple, msg); }

void __ubsan_message(const char *kind, uintptr_t caller) {
  size_t3 gid = {__builtin_spirv_workgroup_id(0),
                 __builtin_spirv_workgroup_id(1),
                 __builtin_spirv_workgroup_id(2)};
  size_t3 lid = {__builtin_spirv_local_invocation_id(0),
                 __builtin_spirv_local_invocation_id(1),
                 __builtin_spirv_local_invocation_id(2)};

  printf(ubsan_msg_fmt, kind, caller, gid, lid);
}

[[noreturn]] void __ubsan_abort() { __ubsan_abort_with_message("abort", 0); }

[[noreturn]] void __ubsan_abort_with_message(const char *kind,
                                             uintptr_t caller) {
  size_t3 gid = {__builtin_spirv_workgroup_id(0),
                 __builtin_spirv_workgroup_id(1),
                 __builtin_spirv_workgroup_id(2)};
  size_t3 lid = {__builtin_spirv_local_invocation_id(0),
                 __builtin_spirv_local_invocation_id(1),
                 __builtin_spirv_local_invocation_id(2)};

  __ubsan_abort_info_t info = {ubsan_msg_fmt, kind, caller, gid, lid};
  __spirv_AbortKHR(info);
}
