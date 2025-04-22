#ifndef LLVM_OpenACC_INC
#define LLVM_OpenACC_INC

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>

namespace llvm {
class StringRef;
namespace acc {

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

enum class Association {
  Block,
  Declaration,
  Delimited,
  Loop,
  None,
  Separating,
};

static constexpr std::size_t Association_enumSize = 6;

enum class Category {
  Declarative,
  Executable,
  Informational,
  Meta,
  Subsidiary,
  Utility,
};

static constexpr std::size_t Category_enumSize = 6;

enum class Directive {
  ACCD_atomic,
  ACCD_cache,
  ACCD_data,
  ACCD_declare,
  ACCD_enter_data,
  ACCD_exit_data,
  ACCD_host_data,
  ACCD_init,
  ACCD_kernels,
  ACCD_kernels_loop,
  ACCD_loop,
  ACCD_parallel,
  ACCD_parallel_loop,
  ACCD_routine,
  ACCD_serial,
  ACCD_serial_loop,
  ACCD_set,
  ACCD_shutdown,
  ACCD_unknown,
  ACCD_update,
  ACCD_wait,
};

static constexpr std::size_t Directive_enumSize = 21;

constexpr auto ACCD_atomic = llvm::acc::Directive::ACCD_atomic;
constexpr auto ACCD_cache = llvm::acc::Directive::ACCD_cache;
constexpr auto ACCD_data = llvm::acc::Directive::ACCD_data;
constexpr auto ACCD_declare = llvm::acc::Directive::ACCD_declare;
constexpr auto ACCD_enter_data = llvm::acc::Directive::ACCD_enter_data;
constexpr auto ACCD_exit_data = llvm::acc::Directive::ACCD_exit_data;
constexpr auto ACCD_host_data = llvm::acc::Directive::ACCD_host_data;
constexpr auto ACCD_init = llvm::acc::Directive::ACCD_init;
constexpr auto ACCD_kernels = llvm::acc::Directive::ACCD_kernels;
constexpr auto ACCD_kernels_loop = llvm::acc::Directive::ACCD_kernels_loop;
constexpr auto ACCD_loop = llvm::acc::Directive::ACCD_loop;
constexpr auto ACCD_parallel = llvm::acc::Directive::ACCD_parallel;
constexpr auto ACCD_parallel_loop = llvm::acc::Directive::ACCD_parallel_loop;
constexpr auto ACCD_routine = llvm::acc::Directive::ACCD_routine;
constexpr auto ACCD_serial = llvm::acc::Directive::ACCD_serial;
constexpr auto ACCD_serial_loop = llvm::acc::Directive::ACCD_serial_loop;
constexpr auto ACCD_set = llvm::acc::Directive::ACCD_set;
constexpr auto ACCD_shutdown = llvm::acc::Directive::ACCD_shutdown;
constexpr auto ACCD_unknown = llvm::acc::Directive::ACCD_unknown;
constexpr auto ACCD_update = llvm::acc::Directive::ACCD_update;
constexpr auto ACCD_wait = llvm::acc::Directive::ACCD_wait;

enum class Clause {
  ACCC_async,
  ACCC_attach,
  ACCC_auto,
  ACCC_bind,
  ACCC_capture,
  ACCC_collapse,
  ACCC_copy,
  ACCC_copyin,
  ACCC_copyout,
  ACCC_create,
  ACCC_default,
  ACCC_default_async,
  ACCC_delete,
  ACCC_detach,
  ACCC_device,
  ACCC_device_num,
  ACCC_deviceptr,
  ACCC_device_resident,
  ACCC_device_type,
  ACCC_finalize,
  ACCC_firstprivate,
  ACCC_gang,
  ACCC_host,
  ACCC_if,
  ACCC_if_present,
  ACCC_independent,
  ACCC_link,
  ACCC_no_create,
  ACCC_nohost,
  ACCC_num_gangs,
  ACCC_num_workers,
  ACCC_present,
  ACCC_private,
  ACCC_read,
  ACCC_reduction,
  ACCC_self,
  ACCC_seq,
  ACCC_shortloop,
  ACCC_tile,
  ACCC_unknown,
  ACCC_use_device,
  ACCC_vector,
  ACCC_vector_length,
  ACCC_wait,
  ACCC_worker,
  ACCC_write,
};

static constexpr std::size_t Clause_enumSize = 46;

constexpr auto ACCC_async = llvm::acc::Clause::ACCC_async;
constexpr auto ACCC_attach = llvm::acc::Clause::ACCC_attach;
constexpr auto ACCC_auto = llvm::acc::Clause::ACCC_auto;
constexpr auto ACCC_bind = llvm::acc::Clause::ACCC_bind;
constexpr auto ACCC_capture = llvm::acc::Clause::ACCC_capture;
constexpr auto ACCC_collapse = llvm::acc::Clause::ACCC_collapse;
constexpr auto ACCC_copy = llvm::acc::Clause::ACCC_copy;
constexpr auto ACCC_copyin = llvm::acc::Clause::ACCC_copyin;
constexpr auto ACCC_copyout = llvm::acc::Clause::ACCC_copyout;
constexpr auto ACCC_create = llvm::acc::Clause::ACCC_create;
constexpr auto ACCC_default = llvm::acc::Clause::ACCC_default;
constexpr auto ACCC_default_async = llvm::acc::Clause::ACCC_default_async;
constexpr auto ACCC_delete = llvm::acc::Clause::ACCC_delete;
constexpr auto ACCC_detach = llvm::acc::Clause::ACCC_detach;
constexpr auto ACCC_device = llvm::acc::Clause::ACCC_device;
constexpr auto ACCC_device_num = llvm::acc::Clause::ACCC_device_num;
constexpr auto ACCC_deviceptr = llvm::acc::Clause::ACCC_deviceptr;
constexpr auto ACCC_device_resident = llvm::acc::Clause::ACCC_device_resident;
constexpr auto ACCC_device_type = llvm::acc::Clause::ACCC_device_type;
constexpr auto ACCC_finalize = llvm::acc::Clause::ACCC_finalize;
constexpr auto ACCC_firstprivate = llvm::acc::Clause::ACCC_firstprivate;
constexpr auto ACCC_gang = llvm::acc::Clause::ACCC_gang;
constexpr auto ACCC_host = llvm::acc::Clause::ACCC_host;
constexpr auto ACCC_if = llvm::acc::Clause::ACCC_if;
constexpr auto ACCC_if_present = llvm::acc::Clause::ACCC_if_present;
constexpr auto ACCC_independent = llvm::acc::Clause::ACCC_independent;
constexpr auto ACCC_link = llvm::acc::Clause::ACCC_link;
constexpr auto ACCC_no_create = llvm::acc::Clause::ACCC_no_create;
constexpr auto ACCC_nohost = llvm::acc::Clause::ACCC_nohost;
constexpr auto ACCC_num_gangs = llvm::acc::Clause::ACCC_num_gangs;
constexpr auto ACCC_num_workers = llvm::acc::Clause::ACCC_num_workers;
constexpr auto ACCC_present = llvm::acc::Clause::ACCC_present;
constexpr auto ACCC_private = llvm::acc::Clause::ACCC_private;
constexpr auto ACCC_read = llvm::acc::Clause::ACCC_read;
constexpr auto ACCC_reduction = llvm::acc::Clause::ACCC_reduction;
constexpr auto ACCC_self = llvm::acc::Clause::ACCC_self;
constexpr auto ACCC_seq = llvm::acc::Clause::ACCC_seq;
constexpr auto ACCC_shortloop = llvm::acc::Clause::ACCC_shortloop;
constexpr auto ACCC_tile = llvm::acc::Clause::ACCC_tile;
constexpr auto ACCC_unknown = llvm::acc::Clause::ACCC_unknown;
constexpr auto ACCC_use_device = llvm::acc::Clause::ACCC_use_device;
constexpr auto ACCC_vector = llvm::acc::Clause::ACCC_vector;
constexpr auto ACCC_vector_length = llvm::acc::Clause::ACCC_vector_length;
constexpr auto ACCC_wait = llvm::acc::Clause::ACCC_wait;
constexpr auto ACCC_worker = llvm::acc::Clause::ACCC_worker;
constexpr auto ACCC_write = llvm::acc::Clause::ACCC_write;

enum class DefaultValue {
  ACC_Default_present=0,
  ACC_Default_none=1,
};

constexpr auto ACC_Default_present = llvm::acc::DefaultValue::ACC_Default_present;
constexpr auto ACC_Default_none = llvm::acc::DefaultValue::ACC_Default_none;

// Enumeration helper functions
LLVM_ABI Directive getOpenACCDirectiveKind(llvm::StringRef Str);

LLVM_ABI llvm::StringRef getOpenACCDirectiveName(Directive D);

LLVM_ABI Clause getOpenACCClauseKind(llvm::StringRef Str);

LLVM_ABI llvm::StringRef getOpenACCClauseName(Clause C);

/// Return true if \p C is a valid clause for \p D in version \p Version.
LLVM_ABI bool isAllowedClauseForDirective(Directive D, Clause C, unsigned Version);

constexpr std::size_t getMaxLeafCount() { return 2; }
LLVM_ABI Association getDirectiveAssociation(Directive D);
LLVM_ABI Category getDirectiveCategory(Directive D);
LLVM_ABI DefaultValue getDefaultValue(StringRef);
LLVM_ABI llvm::StringRef getOpenACCDefaultValueName(DefaultValue);

} // namespace acc
} // namespace llvm
#endif // LLVM_OpenACC_INC
