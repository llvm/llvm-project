#include "ubsan_minimal_common.h"

using uint3 = unsigned __attribute__((ext_vector_type(3)));

struct __ubsan_abort_info_t {
  __attribute__((opencl_constant)) const char *fmt;
  const char *kind;
  uintptr_t caller;
  uint3 gid;
  uint3 lid;
};

// OpenCL printf maps to OpExtInst printf (OpenCL extended instruction set).
extern "C" int printf(__attribute__((opencl_constant)) const char *fmt, ...);

// OpenCL work-item builtins map to SPIR-V BuiltIn variables.
extern "C" unsigned get_group_id(unsigned dim);
extern "C" unsigned get_local_id(unsigned dim);

static __attribute__((opencl_constant)) const char ubsan_msg_simple[] = "%s";
static __attribute__((opencl_constant)) const char ubsan_msg_fmt[] =
    "ubsan: %s by 0x%lx at gid=[%v3u] lid=[%v3u]\n";

void __ubsan_message(const char *msg) { printf(ubsan_msg_simple, msg); }

void __ubsan_message(const char *kind, uintptr_t caller) {
  uint3 gid = {get_group_id(0), get_group_id(1), get_group_id(2)};
  uint3 lid = {get_local_id(0), get_local_id(1), get_local_id(2)};

  printf(ubsan_msg_fmt, kind, caller, gid, lid);
}

// SPV_KHR_abort: OpAbortKHR terminates the invocation and passes a message
// to the client API. The message is passed as a typed value.
[[noreturn]] void __spirv_AbortKHR(__ubsan_abort_info_t info);

[[noreturn]] void __ubsan_abort() { __ubsan_abort_with_message("abort", 0); }

[[noreturn]] void __ubsan_abort_with_message(const char *kind,
                                             uintptr_t caller) {
  uint3 gid = {get_group_id(0), get_group_id(1), get_group_id(2)};
  uint3 lid = {get_local_id(0), get_local_id(1), get_local_id(2)};
  __ubsan_abort_info_t info = {ubsan_msg_fmt, kind, caller, gid, lid};
  __spirv_AbortKHR(info);
}
