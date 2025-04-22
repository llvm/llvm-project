#ifndef LLVM_OpenMP_INC
#define LLVM_OpenMP_INC

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/Support/Compiler.h"
#include <cstddef>

namespace llvm {
class StringRef;
namespace omp {

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
  OMPD_allocate,
  OMPD_allocators,
  OMPD_assume,
  OMPD_assumes,
  OMPD_atomic,
  OMPD_barrier,
  OMPD_begin_assumes,
  OMPD_begin_declare_target,
  OMPD_begin_declare_variant,
  OMPD_cancel,
  OMPD_cancellation_point,
  OMPD_critical,
  OMPD_declare_mapper,
  OMPD_declare_reduction,
  OMPD_declare_simd,
  OMPD_declare_target,
  OMPD_declare_variant,
  OMPD_depobj,
  OMPD_distribute,
  OMPD_distribute_parallel_do,
  OMPD_distribute_parallel_do_simd,
  OMPD_distribute_parallel_for,
  OMPD_distribute_parallel_for_simd,
  OMPD_distribute_simd,
  OMPD_do,
  OMPD_do_simd,
  OMPD_end_assumes,
  OMPD_end_declare_target,
  OMPD_end_declare_variant,
  OMPD_end_do,
  OMPD_end_do_simd,
  OMPD_end_scope,
  OMPD_end_sections,
  OMPD_end_single,
  OMPD_end_workshare,
  OMPD_error,
  OMPD_flush,
  OMPD_for,
  OMPD_for_simd,
  OMPD_interchange,
  OMPD_masked_taskloop,
  OMPD_masked_taskloop_simd,
  OMPD_master,
  OMPD_master_taskloop,
  OMPD_master_taskloop_simd,
  OMPD_metadirective,
  OMPD_nothing,
  OMPD_ordered,
  OMPD_parallel,
  OMPD_parallel_do,
  OMPD_parallel_do_simd,
  OMPD_parallel_for,
  OMPD_parallel_for_simd,
  OMPD_parallel_masked,
  OMPD_parallel_masked_taskloop,
  OMPD_parallel_masked_taskloop_simd,
  OMPD_parallel_master,
  OMPD_parallel_master_taskloop,
  OMPD_parallel_master_taskloop_simd,
  OMPD_parallel_sections,
  OMPD_parallel_workshare,
  OMPD_requires,
  OMPD_reverse,
  OMPD_scan,
  OMPD_scope,
  OMPD_section,
  OMPD_sections,
  OMPD_simd,
  OMPD_single,
  OMPD_stripe,
  OMPD_target,
  OMPD_target_data,
  OMPD_target_enter_data,
  OMPD_target_exit_data,
  OMPD_target_parallel,
  OMPD_target_parallel_do,
  OMPD_target_parallel_do_simd,
  OMPD_target_parallel_for,
  OMPD_target_parallel_for_simd,
  OMPD_target_simd,
  OMPD_target_teams,
  OMPD_target_teams_distribute,
  OMPD_target_teams_distribute_parallel_do,
  OMPD_target_teams_distribute_parallel_do_simd,
  OMPD_target_teams_distribute_parallel_for,
  OMPD_target_teams_distribute_parallel_for_simd,
  OMPD_target_teams_distribute_simd,
  OMPD_target_update,
  OMPD_task,
  OMPD_taskgroup,
  OMPD_taskloop,
  OMPD_taskloop_simd,
  OMPD_taskwait,
  OMPD_taskyield,
  OMPD_teams,
  OMPD_teams_distribute,
  OMPD_teams_distribute_parallel_do,
  OMPD_teams_distribute_parallel_do_simd,
  OMPD_teams_distribute_parallel_for,
  OMPD_teams_distribute_parallel_for_simd,
  OMPD_teams_distribute_simd,
  OMPD_threadprivate,
  OMPD_tile,
  OMPD_unknown,
  OMPD_unroll,
  OMPD_workshare,
  OMPD_dispatch,
  OMPD_interop,
  OMPD_loop,
  OMPD_masked,
  OMPD_parallel_loop,
  OMPD_target_loop,
  OMPD_target_parallel_loop,
  OMPD_target_teams_loop,
  OMPD_teams_loop,
};

static constexpr std::size_t Directive_enumSize = 115;

constexpr auto OMPD_allocate = llvm::omp::Directive::OMPD_allocate;
constexpr auto OMPD_allocators = llvm::omp::Directive::OMPD_allocators;
constexpr auto OMPD_assume = llvm::omp::Directive::OMPD_assume;
constexpr auto OMPD_assumes = llvm::omp::Directive::OMPD_assumes;
constexpr auto OMPD_atomic = llvm::omp::Directive::OMPD_atomic;
constexpr auto OMPD_barrier = llvm::omp::Directive::OMPD_barrier;
constexpr auto OMPD_begin_assumes = llvm::omp::Directive::OMPD_begin_assumes;
constexpr auto OMPD_begin_declare_target = llvm::omp::Directive::OMPD_begin_declare_target;
constexpr auto OMPD_begin_declare_variant = llvm::omp::Directive::OMPD_begin_declare_variant;
constexpr auto OMPD_cancel = llvm::omp::Directive::OMPD_cancel;
constexpr auto OMPD_cancellation_point = llvm::omp::Directive::OMPD_cancellation_point;
constexpr auto OMPD_critical = llvm::omp::Directive::OMPD_critical;
constexpr auto OMPD_declare_mapper = llvm::omp::Directive::OMPD_declare_mapper;
constexpr auto OMPD_declare_reduction = llvm::omp::Directive::OMPD_declare_reduction;
constexpr auto OMPD_declare_simd = llvm::omp::Directive::OMPD_declare_simd;
constexpr auto OMPD_declare_target = llvm::omp::Directive::OMPD_declare_target;
constexpr auto OMPD_declare_variant = llvm::omp::Directive::OMPD_declare_variant;
constexpr auto OMPD_depobj = llvm::omp::Directive::OMPD_depobj;
constexpr auto OMPD_distribute = llvm::omp::Directive::OMPD_distribute;
constexpr auto OMPD_distribute_parallel_do = llvm::omp::Directive::OMPD_distribute_parallel_do;
constexpr auto OMPD_distribute_parallel_do_simd = llvm::omp::Directive::OMPD_distribute_parallel_do_simd;
constexpr auto OMPD_distribute_parallel_for = llvm::omp::Directive::OMPD_distribute_parallel_for;
constexpr auto OMPD_distribute_parallel_for_simd = llvm::omp::Directive::OMPD_distribute_parallel_for_simd;
constexpr auto OMPD_distribute_simd = llvm::omp::Directive::OMPD_distribute_simd;
constexpr auto OMPD_do = llvm::omp::Directive::OMPD_do;
constexpr auto OMPD_do_simd = llvm::omp::Directive::OMPD_do_simd;
constexpr auto OMPD_end_assumes = llvm::omp::Directive::OMPD_end_assumes;
constexpr auto OMPD_end_declare_target = llvm::omp::Directive::OMPD_end_declare_target;
constexpr auto OMPD_end_declare_variant = llvm::omp::Directive::OMPD_end_declare_variant;
constexpr auto OMPD_end_do = llvm::omp::Directive::OMPD_end_do;
constexpr auto OMPD_end_do_simd = llvm::omp::Directive::OMPD_end_do_simd;
constexpr auto OMPD_end_scope = llvm::omp::Directive::OMPD_end_scope;
constexpr auto OMPD_end_sections = llvm::omp::Directive::OMPD_end_sections;
constexpr auto OMPD_end_single = llvm::omp::Directive::OMPD_end_single;
constexpr auto OMPD_end_workshare = llvm::omp::Directive::OMPD_end_workshare;
constexpr auto OMPD_error = llvm::omp::Directive::OMPD_error;
constexpr auto OMPD_flush = llvm::omp::Directive::OMPD_flush;
constexpr auto OMPD_for = llvm::omp::Directive::OMPD_for;
constexpr auto OMPD_for_simd = llvm::omp::Directive::OMPD_for_simd;
constexpr auto OMPD_interchange = llvm::omp::Directive::OMPD_interchange;
constexpr auto OMPD_masked_taskloop = llvm::omp::Directive::OMPD_masked_taskloop;
constexpr auto OMPD_masked_taskloop_simd = llvm::omp::Directive::OMPD_masked_taskloop_simd;
constexpr auto OMPD_master = llvm::omp::Directive::OMPD_master;
constexpr auto OMPD_master_taskloop = llvm::omp::Directive::OMPD_master_taskloop;
constexpr auto OMPD_master_taskloop_simd = llvm::omp::Directive::OMPD_master_taskloop_simd;
constexpr auto OMPD_metadirective = llvm::omp::Directive::OMPD_metadirective;
constexpr auto OMPD_nothing = llvm::omp::Directive::OMPD_nothing;
constexpr auto OMPD_ordered = llvm::omp::Directive::OMPD_ordered;
constexpr auto OMPD_parallel = llvm::omp::Directive::OMPD_parallel;
constexpr auto OMPD_parallel_do = llvm::omp::Directive::OMPD_parallel_do;
constexpr auto OMPD_parallel_do_simd = llvm::omp::Directive::OMPD_parallel_do_simd;
constexpr auto OMPD_parallel_for = llvm::omp::Directive::OMPD_parallel_for;
constexpr auto OMPD_parallel_for_simd = llvm::omp::Directive::OMPD_parallel_for_simd;
constexpr auto OMPD_parallel_masked = llvm::omp::Directive::OMPD_parallel_masked;
constexpr auto OMPD_parallel_masked_taskloop = llvm::omp::Directive::OMPD_parallel_masked_taskloop;
constexpr auto OMPD_parallel_masked_taskloop_simd = llvm::omp::Directive::OMPD_parallel_masked_taskloop_simd;
constexpr auto OMPD_parallel_master = llvm::omp::Directive::OMPD_parallel_master;
constexpr auto OMPD_parallel_master_taskloop = llvm::omp::Directive::OMPD_parallel_master_taskloop;
constexpr auto OMPD_parallel_master_taskloop_simd = llvm::omp::Directive::OMPD_parallel_master_taskloop_simd;
constexpr auto OMPD_parallel_sections = llvm::omp::Directive::OMPD_parallel_sections;
constexpr auto OMPD_parallel_workshare = llvm::omp::Directive::OMPD_parallel_workshare;
constexpr auto OMPD_requires = llvm::omp::Directive::OMPD_requires;
constexpr auto OMPD_reverse = llvm::omp::Directive::OMPD_reverse;
constexpr auto OMPD_scan = llvm::omp::Directive::OMPD_scan;
constexpr auto OMPD_scope = llvm::omp::Directive::OMPD_scope;
constexpr auto OMPD_section = llvm::omp::Directive::OMPD_section;
constexpr auto OMPD_sections = llvm::omp::Directive::OMPD_sections;
constexpr auto OMPD_simd = llvm::omp::Directive::OMPD_simd;
constexpr auto OMPD_single = llvm::omp::Directive::OMPD_single;
constexpr auto OMPD_stripe = llvm::omp::Directive::OMPD_stripe;
constexpr auto OMPD_target = llvm::omp::Directive::OMPD_target;
constexpr auto OMPD_target_data = llvm::omp::Directive::OMPD_target_data;
constexpr auto OMPD_target_enter_data = llvm::omp::Directive::OMPD_target_enter_data;
constexpr auto OMPD_target_exit_data = llvm::omp::Directive::OMPD_target_exit_data;
constexpr auto OMPD_target_parallel = llvm::omp::Directive::OMPD_target_parallel;
constexpr auto OMPD_target_parallel_do = llvm::omp::Directive::OMPD_target_parallel_do;
constexpr auto OMPD_target_parallel_do_simd = llvm::omp::Directive::OMPD_target_parallel_do_simd;
constexpr auto OMPD_target_parallel_for = llvm::omp::Directive::OMPD_target_parallel_for;
constexpr auto OMPD_target_parallel_for_simd = llvm::omp::Directive::OMPD_target_parallel_for_simd;
constexpr auto OMPD_target_simd = llvm::omp::Directive::OMPD_target_simd;
constexpr auto OMPD_target_teams = llvm::omp::Directive::OMPD_target_teams;
constexpr auto OMPD_target_teams_distribute = llvm::omp::Directive::OMPD_target_teams_distribute;
constexpr auto OMPD_target_teams_distribute_parallel_do = llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do;
constexpr auto OMPD_target_teams_distribute_parallel_do_simd = llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd;
constexpr auto OMPD_target_teams_distribute_parallel_for = llvm::omp::Directive::OMPD_target_teams_distribute_parallel_for;
constexpr auto OMPD_target_teams_distribute_parallel_for_simd = llvm::omp::Directive::OMPD_target_teams_distribute_parallel_for_simd;
constexpr auto OMPD_target_teams_distribute_simd = llvm::omp::Directive::OMPD_target_teams_distribute_simd;
constexpr auto OMPD_target_update = llvm::omp::Directive::OMPD_target_update;
constexpr auto OMPD_task = llvm::omp::Directive::OMPD_task;
constexpr auto OMPD_taskgroup = llvm::omp::Directive::OMPD_taskgroup;
constexpr auto OMPD_taskloop = llvm::omp::Directive::OMPD_taskloop;
constexpr auto OMPD_taskloop_simd = llvm::omp::Directive::OMPD_taskloop_simd;
constexpr auto OMPD_taskwait = llvm::omp::Directive::OMPD_taskwait;
constexpr auto OMPD_taskyield = llvm::omp::Directive::OMPD_taskyield;
constexpr auto OMPD_teams = llvm::omp::Directive::OMPD_teams;
constexpr auto OMPD_teams_distribute = llvm::omp::Directive::OMPD_teams_distribute;
constexpr auto OMPD_teams_distribute_parallel_do = llvm::omp::Directive::OMPD_teams_distribute_parallel_do;
constexpr auto OMPD_teams_distribute_parallel_do_simd = llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd;
constexpr auto OMPD_teams_distribute_parallel_for = llvm::omp::Directive::OMPD_teams_distribute_parallel_for;
constexpr auto OMPD_teams_distribute_parallel_for_simd = llvm::omp::Directive::OMPD_teams_distribute_parallel_for_simd;
constexpr auto OMPD_teams_distribute_simd = llvm::omp::Directive::OMPD_teams_distribute_simd;
constexpr auto OMPD_threadprivate = llvm::omp::Directive::OMPD_threadprivate;
constexpr auto OMPD_tile = llvm::omp::Directive::OMPD_tile;
constexpr auto OMPD_unknown = llvm::omp::Directive::OMPD_unknown;
constexpr auto OMPD_unroll = llvm::omp::Directive::OMPD_unroll;
constexpr auto OMPD_workshare = llvm::omp::Directive::OMPD_workshare;
constexpr auto OMPD_dispatch = llvm::omp::Directive::OMPD_dispatch;
constexpr auto OMPD_interop = llvm::omp::Directive::OMPD_interop;
constexpr auto OMPD_loop = llvm::omp::Directive::OMPD_loop;
constexpr auto OMPD_masked = llvm::omp::Directive::OMPD_masked;
constexpr auto OMPD_parallel_loop = llvm::omp::Directive::OMPD_parallel_loop;
constexpr auto OMPD_target_loop = llvm::omp::Directive::OMPD_target_loop;
constexpr auto OMPD_target_parallel_loop = llvm::omp::Directive::OMPD_target_parallel_loop;
constexpr auto OMPD_target_teams_loop = llvm::omp::Directive::OMPD_target_teams_loop;
constexpr auto OMPD_teams_loop = llvm::omp::Directive::OMPD_teams_loop;

enum class Clause {
  OMPC_absent,
  OMPC_acq_rel,
  OMPC_acquire,
  OMPC_adjust_args,
  OMPC_affinity,
  OMPC_align,
  OMPC_aligned,
  OMPC_allocate,
  OMPC_allocator,
  OMPC_append_args,
  OMPC_at,
  OMPC_atomic_default_mem_order,
  OMPC_bind,
  OMPC_cancellation_construct_type,
  OMPC_capture,
  OMPC_collapse,
  OMPC_compare,
  OMPC_contains,
  OMPC_copyprivate,
  OMPC_copyin,
  OMPC_default,
  OMPC_defaultmap,
  OMPC_depend,
  OMPC_depobj,
  OMPC_destroy,
  OMPC_detach,
  OMPC_device,
  OMPC_device_type,
  OMPC_dist_schedule,
  OMPC_doacross,
  OMPC_dynamic_allocators,
  OMPC_enter,
  OMPC_exclusive,
  OMPC_fail,
  OMPC_filter,
  OMPC_final,
  OMPC_firstprivate,
  OMPC_flush,
  OMPC_from,
  OMPC_full,
  OMPC_grainsize,
  OMPC_has_device_addr,
  OMPC_hint,
  OMPC_holds,
  OMPC_if,
  OMPC_in_reduction,
  OMPC_inbranch,
  OMPC_inclusive,
  OMPC_indirect,
  OMPC_init,
  OMPC_initializer,
  OMPC_is_device_ptr,
  OMPC_lastprivate,
  OMPC_linear,
  OMPC_link,
  OMPC_map,
  OMPC_match,
  OMPC_memory_order,
  OMPC_mergeable,
  OMPC_message,
  OMPC_nogroup,
  OMPC_no_openmp,
  OMPC_no_openmp_constructs,
  OMPC_no_openmp_routines,
  OMPC_no_parallelism,
  OMPC_nowait,
  OMPC_nocontext,
  OMPC_nontemporal,
  OMPC_notinbranch,
  OMPC_novariants,
  OMPC_num_tasks,
  OMPC_num_teams,
  OMPC_num_threads,
  OMPC_ompx_attribute,
  OMPC_ompx_bare,
  OMPC_ompx_dyn_cgroup_mem,
  OMPC_order,
  OMPC_ordered,
  OMPC_otherwise,
  OMPC_partial,
  OMPC_permutation,
  OMPC_priority,
  OMPC_private,
  OMPC_proc_bind,
  OMPC_read,
  OMPC_reduction,
  OMPC_relaxed,
  OMPC_release,
  OMPC_reverse_offload,
  OMPC_safelen,
  OMPC_schedule,
  OMPC_self_maps,
  OMPC_seq_cst,
  OMPC_severity,
  OMPC_shared,
  OMPC_simd,
  OMPC_simdlen,
  OMPC_sizes,
  OMPC_task_reduction,
  OMPC_thread_limit,
  OMPC_threadprivate,
  OMPC_threads,
  OMPC_to,
  OMPC_unified_address,
  OMPC_unified_shared_memory,
  OMPC_uniform,
  OMPC_unknown,
  OMPC_untied,
  OMPC_update,
  OMPC_use,
  OMPC_use_device_addr,
  OMPC_use_device_ptr,
  OMPC_uses_allocators,
  OMPC_weak,
  OMPC_when,
  OMPC_write,
};

static constexpr std::size_t Clause_enumSize = 116;

constexpr auto OMPC_absent = llvm::omp::Clause::OMPC_absent;
constexpr auto OMPC_acq_rel = llvm::omp::Clause::OMPC_acq_rel;
constexpr auto OMPC_acquire = llvm::omp::Clause::OMPC_acquire;
constexpr auto OMPC_adjust_args = llvm::omp::Clause::OMPC_adjust_args;
constexpr auto OMPC_affinity = llvm::omp::Clause::OMPC_affinity;
constexpr auto OMPC_align = llvm::omp::Clause::OMPC_align;
constexpr auto OMPC_aligned = llvm::omp::Clause::OMPC_aligned;
constexpr auto OMPC_allocate = llvm::omp::Clause::OMPC_allocate;
constexpr auto OMPC_allocator = llvm::omp::Clause::OMPC_allocator;
constexpr auto OMPC_append_args = llvm::omp::Clause::OMPC_append_args;
constexpr auto OMPC_at = llvm::omp::Clause::OMPC_at;
constexpr auto OMPC_atomic_default_mem_order = llvm::omp::Clause::OMPC_atomic_default_mem_order;
constexpr auto OMPC_bind = llvm::omp::Clause::OMPC_bind;
constexpr auto OMPC_cancellation_construct_type = llvm::omp::Clause::OMPC_cancellation_construct_type;
constexpr auto OMPC_capture = llvm::omp::Clause::OMPC_capture;
constexpr auto OMPC_collapse = llvm::omp::Clause::OMPC_collapse;
constexpr auto OMPC_compare = llvm::omp::Clause::OMPC_compare;
constexpr auto OMPC_contains = llvm::omp::Clause::OMPC_contains;
constexpr auto OMPC_copyprivate = llvm::omp::Clause::OMPC_copyprivate;
constexpr auto OMPC_copyin = llvm::omp::Clause::OMPC_copyin;
constexpr auto OMPC_default = llvm::omp::Clause::OMPC_default;
constexpr auto OMPC_defaultmap = llvm::omp::Clause::OMPC_defaultmap;
constexpr auto OMPC_depend = llvm::omp::Clause::OMPC_depend;
constexpr auto OMPC_depobj = llvm::omp::Clause::OMPC_depobj;
constexpr auto OMPC_destroy = llvm::omp::Clause::OMPC_destroy;
constexpr auto OMPC_detach = llvm::omp::Clause::OMPC_detach;
constexpr auto OMPC_device = llvm::omp::Clause::OMPC_device;
constexpr auto OMPC_device_type = llvm::omp::Clause::OMPC_device_type;
constexpr auto OMPC_dist_schedule = llvm::omp::Clause::OMPC_dist_schedule;
constexpr auto OMPC_doacross = llvm::omp::Clause::OMPC_doacross;
constexpr auto OMPC_dynamic_allocators = llvm::omp::Clause::OMPC_dynamic_allocators;
constexpr auto OMPC_enter = llvm::omp::Clause::OMPC_enter;
constexpr auto OMPC_exclusive = llvm::omp::Clause::OMPC_exclusive;
constexpr auto OMPC_fail = llvm::omp::Clause::OMPC_fail;
constexpr auto OMPC_filter = llvm::omp::Clause::OMPC_filter;
constexpr auto OMPC_final = llvm::omp::Clause::OMPC_final;
constexpr auto OMPC_firstprivate = llvm::omp::Clause::OMPC_firstprivate;
constexpr auto OMPC_flush = llvm::omp::Clause::OMPC_flush;
constexpr auto OMPC_from = llvm::omp::Clause::OMPC_from;
constexpr auto OMPC_full = llvm::omp::Clause::OMPC_full;
constexpr auto OMPC_grainsize = llvm::omp::Clause::OMPC_grainsize;
constexpr auto OMPC_has_device_addr = llvm::omp::Clause::OMPC_has_device_addr;
constexpr auto OMPC_hint = llvm::omp::Clause::OMPC_hint;
constexpr auto OMPC_holds = llvm::omp::Clause::OMPC_holds;
constexpr auto OMPC_if = llvm::omp::Clause::OMPC_if;
constexpr auto OMPC_in_reduction = llvm::omp::Clause::OMPC_in_reduction;
constexpr auto OMPC_inbranch = llvm::omp::Clause::OMPC_inbranch;
constexpr auto OMPC_inclusive = llvm::omp::Clause::OMPC_inclusive;
constexpr auto OMPC_indirect = llvm::omp::Clause::OMPC_indirect;
constexpr auto OMPC_init = llvm::omp::Clause::OMPC_init;
constexpr auto OMPC_initializer = llvm::omp::Clause::OMPC_initializer;
constexpr auto OMPC_is_device_ptr = llvm::omp::Clause::OMPC_is_device_ptr;
constexpr auto OMPC_lastprivate = llvm::omp::Clause::OMPC_lastprivate;
constexpr auto OMPC_linear = llvm::omp::Clause::OMPC_linear;
constexpr auto OMPC_link = llvm::omp::Clause::OMPC_link;
constexpr auto OMPC_map = llvm::omp::Clause::OMPC_map;
constexpr auto OMPC_match = llvm::omp::Clause::OMPC_match;
constexpr auto OMPC_memory_order = llvm::omp::Clause::OMPC_memory_order;
constexpr auto OMPC_mergeable = llvm::omp::Clause::OMPC_mergeable;
constexpr auto OMPC_message = llvm::omp::Clause::OMPC_message;
constexpr auto OMPC_nogroup = llvm::omp::Clause::OMPC_nogroup;
constexpr auto OMPC_no_openmp = llvm::omp::Clause::OMPC_no_openmp;
constexpr auto OMPC_no_openmp_constructs = llvm::omp::Clause::OMPC_no_openmp_constructs;
constexpr auto OMPC_no_openmp_routines = llvm::omp::Clause::OMPC_no_openmp_routines;
constexpr auto OMPC_no_parallelism = llvm::omp::Clause::OMPC_no_parallelism;
constexpr auto OMPC_nowait = llvm::omp::Clause::OMPC_nowait;
constexpr auto OMPC_nocontext = llvm::omp::Clause::OMPC_nocontext;
constexpr auto OMPC_nontemporal = llvm::omp::Clause::OMPC_nontemporal;
constexpr auto OMPC_notinbranch = llvm::omp::Clause::OMPC_notinbranch;
constexpr auto OMPC_novariants = llvm::omp::Clause::OMPC_novariants;
constexpr auto OMPC_num_tasks = llvm::omp::Clause::OMPC_num_tasks;
constexpr auto OMPC_num_teams = llvm::omp::Clause::OMPC_num_teams;
constexpr auto OMPC_num_threads = llvm::omp::Clause::OMPC_num_threads;
constexpr auto OMPC_ompx_attribute = llvm::omp::Clause::OMPC_ompx_attribute;
constexpr auto OMPC_ompx_bare = llvm::omp::Clause::OMPC_ompx_bare;
constexpr auto OMPC_ompx_dyn_cgroup_mem = llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem;
constexpr auto OMPC_order = llvm::omp::Clause::OMPC_order;
constexpr auto OMPC_ordered = llvm::omp::Clause::OMPC_ordered;
constexpr auto OMPC_otherwise = llvm::omp::Clause::OMPC_otherwise;
constexpr auto OMPC_partial = llvm::omp::Clause::OMPC_partial;
constexpr auto OMPC_permutation = llvm::omp::Clause::OMPC_permutation;
constexpr auto OMPC_priority = llvm::omp::Clause::OMPC_priority;
constexpr auto OMPC_private = llvm::omp::Clause::OMPC_private;
constexpr auto OMPC_proc_bind = llvm::omp::Clause::OMPC_proc_bind;
constexpr auto OMPC_read = llvm::omp::Clause::OMPC_read;
constexpr auto OMPC_reduction = llvm::omp::Clause::OMPC_reduction;
constexpr auto OMPC_relaxed = llvm::omp::Clause::OMPC_relaxed;
constexpr auto OMPC_release = llvm::omp::Clause::OMPC_release;
constexpr auto OMPC_reverse_offload = llvm::omp::Clause::OMPC_reverse_offload;
constexpr auto OMPC_safelen = llvm::omp::Clause::OMPC_safelen;
constexpr auto OMPC_schedule = llvm::omp::Clause::OMPC_schedule;
constexpr auto OMPC_self_maps = llvm::omp::Clause::OMPC_self_maps;
constexpr auto OMPC_seq_cst = llvm::omp::Clause::OMPC_seq_cst;
constexpr auto OMPC_severity = llvm::omp::Clause::OMPC_severity;
constexpr auto OMPC_shared = llvm::omp::Clause::OMPC_shared;
constexpr auto OMPC_simd = llvm::omp::Clause::OMPC_simd;
constexpr auto OMPC_simdlen = llvm::omp::Clause::OMPC_simdlen;
constexpr auto OMPC_sizes = llvm::omp::Clause::OMPC_sizes;
constexpr auto OMPC_task_reduction = llvm::omp::Clause::OMPC_task_reduction;
constexpr auto OMPC_thread_limit = llvm::omp::Clause::OMPC_thread_limit;
constexpr auto OMPC_threadprivate = llvm::omp::Clause::OMPC_threadprivate;
constexpr auto OMPC_threads = llvm::omp::Clause::OMPC_threads;
constexpr auto OMPC_to = llvm::omp::Clause::OMPC_to;
constexpr auto OMPC_unified_address = llvm::omp::Clause::OMPC_unified_address;
constexpr auto OMPC_unified_shared_memory = llvm::omp::Clause::OMPC_unified_shared_memory;
constexpr auto OMPC_uniform = llvm::omp::Clause::OMPC_uniform;
constexpr auto OMPC_unknown = llvm::omp::Clause::OMPC_unknown;
constexpr auto OMPC_untied = llvm::omp::Clause::OMPC_untied;
constexpr auto OMPC_update = llvm::omp::Clause::OMPC_update;
constexpr auto OMPC_use = llvm::omp::Clause::OMPC_use;
constexpr auto OMPC_use_device_addr = llvm::omp::Clause::OMPC_use_device_addr;
constexpr auto OMPC_use_device_ptr = llvm::omp::Clause::OMPC_use_device_ptr;
constexpr auto OMPC_uses_allocators = llvm::omp::Clause::OMPC_uses_allocators;
constexpr auto OMPC_weak = llvm::omp::Clause::OMPC_weak;
constexpr auto OMPC_when = llvm::omp::Clause::OMPC_when;
constexpr auto OMPC_write = llvm::omp::Clause::OMPC_write;

enum class BindKind {
  OMP_BIND_parallel=1,
  OMP_BIND_teams=2,
  OMP_BIND_thread=3,
};

constexpr auto OMP_BIND_parallel = llvm::omp::BindKind::OMP_BIND_parallel;
constexpr auto OMP_BIND_teams = llvm::omp::BindKind::OMP_BIND_teams;
constexpr auto OMP_BIND_thread = llvm::omp::BindKind::OMP_BIND_thread;

enum class CancellationConstructType {
  OMP_CANCELLATION_CONSTRUCT_Parallel=1,
  OMP_CANCELLATION_CONSTRUCT_Loop=2,
  OMP_CANCELLATION_CONSTRUCT_Sections=3,
  OMP_CANCELLATION_CONSTRUCT_Taskgroup=4,
  OMP_CANCELLATION_CONSTRUCT_None=5,
};

constexpr auto OMP_CANCELLATION_CONSTRUCT_Parallel = llvm::omp::CancellationConstructType::OMP_CANCELLATION_CONSTRUCT_Parallel;
constexpr auto OMP_CANCELLATION_CONSTRUCT_Loop = llvm::omp::CancellationConstructType::OMP_CANCELLATION_CONSTRUCT_Loop;
constexpr auto OMP_CANCELLATION_CONSTRUCT_Sections = llvm::omp::CancellationConstructType::OMP_CANCELLATION_CONSTRUCT_Sections;
constexpr auto OMP_CANCELLATION_CONSTRUCT_Taskgroup = llvm::omp::CancellationConstructType::OMP_CANCELLATION_CONSTRUCT_Taskgroup;
constexpr auto OMP_CANCELLATION_CONSTRUCT_None = llvm::omp::CancellationConstructType::OMP_CANCELLATION_CONSTRUCT_None;

enum class GrainsizeType {
  OMP_GRAINSIZE_Strict=1,
  OMP_GRAINSIZE_Unknown=2,
};

constexpr auto OMP_GRAINSIZE_Strict = llvm::omp::GrainsizeType::OMP_GRAINSIZE_Strict;
constexpr auto OMP_GRAINSIZE_Unknown = llvm::omp::GrainsizeType::OMP_GRAINSIZE_Unknown;

enum class MemoryOrderKind {
  OMP_MEMORY_ORDER_SeqCst=1,
  OMP_MEMORY_ORDER_AcqRel=2,
  OMP_MEMORY_ORDER_Acquire=3,
  OMP_MEMORY_ORDER_Release=4,
  OMP_MEMORY_ORDER_Relaxed=5,
  OMP_MEMORY_ORDER_Default=6,
};

constexpr auto OMP_MEMORY_ORDER_SeqCst = llvm::omp::MemoryOrderKind::OMP_MEMORY_ORDER_SeqCst;
constexpr auto OMP_MEMORY_ORDER_AcqRel = llvm::omp::MemoryOrderKind::OMP_MEMORY_ORDER_AcqRel;
constexpr auto OMP_MEMORY_ORDER_Acquire = llvm::omp::MemoryOrderKind::OMP_MEMORY_ORDER_Acquire;
constexpr auto OMP_MEMORY_ORDER_Release = llvm::omp::MemoryOrderKind::OMP_MEMORY_ORDER_Release;
constexpr auto OMP_MEMORY_ORDER_Relaxed = llvm::omp::MemoryOrderKind::OMP_MEMORY_ORDER_Relaxed;
constexpr auto OMP_MEMORY_ORDER_Default = llvm::omp::MemoryOrderKind::OMP_MEMORY_ORDER_Default;

enum class NumTasksType {
  OMP_NUMTASKS_Strict=1,
  OMP_NUMTASKS_Unknown=2,
};

constexpr auto OMP_NUMTASKS_Strict = llvm::omp::NumTasksType::OMP_NUMTASKS_Strict;
constexpr auto OMP_NUMTASKS_Unknown = llvm::omp::NumTasksType::OMP_NUMTASKS_Unknown;

enum class OrderKind {
  OMP_ORDER_unknown=2,
  OMP_ORDER_concurrent=1,
};

constexpr auto OMP_ORDER_unknown = llvm::omp::OrderKind::OMP_ORDER_unknown;
constexpr auto OMP_ORDER_concurrent = llvm::omp::OrderKind::OMP_ORDER_concurrent;

enum class ProcBindKind {
  OMP_PROC_BIND_primary=5,
  OMP_PROC_BIND_master=2,
  OMP_PROC_BIND_close=3,
  OMP_PROC_BIND_spread=4,
  OMP_PROC_BIND_default=6,
  OMP_PROC_BIND_unknown=7,
};

constexpr auto OMP_PROC_BIND_primary = llvm::omp::ProcBindKind::OMP_PROC_BIND_primary;
constexpr auto OMP_PROC_BIND_master = llvm::omp::ProcBindKind::OMP_PROC_BIND_master;
constexpr auto OMP_PROC_BIND_close = llvm::omp::ProcBindKind::OMP_PROC_BIND_close;
constexpr auto OMP_PROC_BIND_spread = llvm::omp::ProcBindKind::OMP_PROC_BIND_spread;
constexpr auto OMP_PROC_BIND_default = llvm::omp::ProcBindKind::OMP_PROC_BIND_default;
constexpr auto OMP_PROC_BIND_unknown = llvm::omp::ProcBindKind::OMP_PROC_BIND_unknown;

enum class ScheduleKind {
  OMP_SCHEDULE_Static=2,
  OMP_SCHEDULE_Dynamic=3,
  OMP_SCHEDULE_Guided=4,
  OMP_SCHEDULE_Auto=5,
  OMP_SCHEDULE_Runtime=6,
  OMP_SCHEDULE_Default=7,
};

constexpr auto OMP_SCHEDULE_Static = llvm::omp::ScheduleKind::OMP_SCHEDULE_Static;
constexpr auto OMP_SCHEDULE_Dynamic = llvm::omp::ScheduleKind::OMP_SCHEDULE_Dynamic;
constexpr auto OMP_SCHEDULE_Guided = llvm::omp::ScheduleKind::OMP_SCHEDULE_Guided;
constexpr auto OMP_SCHEDULE_Auto = llvm::omp::ScheduleKind::OMP_SCHEDULE_Auto;
constexpr auto OMP_SCHEDULE_Runtime = llvm::omp::ScheduleKind::OMP_SCHEDULE_Runtime;
constexpr auto OMP_SCHEDULE_Default = llvm::omp::ScheduleKind::OMP_SCHEDULE_Default;

// Enumeration helper functions
LLVM_ABI Directive getOpenMPDirectiveKind(llvm::StringRef Str);

LLVM_ABI llvm::StringRef getOpenMPDirectiveName(Directive D);

LLVM_ABI Clause getOpenMPClauseKind(llvm::StringRef Str);

LLVM_ABI llvm::StringRef getOpenMPClauseName(Clause C);

/// Return true if \p C is a valid clause for \p D in version \p Version.
LLVM_ABI bool isAllowedClauseForDirective(Directive D, Clause C, unsigned Version);

constexpr std::size_t getMaxLeafCount() { return 6; }
LLVM_ABI Association getDirectiveAssociation(Directive D);
LLVM_ABI Category getDirectiveCategory(Directive D);
LLVM_ABI BindKind getBindKind(StringRef);
LLVM_ABI llvm::StringRef getOpenMPBindKindName(BindKind);
LLVM_ABI CancellationConstructType getCancellationConstructType(StringRef);
LLVM_ABI llvm::StringRef getOpenMPCancellationConstructTypeName(CancellationConstructType);
LLVM_ABI GrainsizeType getGrainsizeType(StringRef);
LLVM_ABI llvm::StringRef getOpenMPGrainsizeTypeName(GrainsizeType);
LLVM_ABI MemoryOrderKind getMemoryOrderKind(StringRef);
LLVM_ABI llvm::StringRef getOpenMPMemoryOrderKindName(MemoryOrderKind);
LLVM_ABI NumTasksType getNumTasksType(StringRef);
LLVM_ABI llvm::StringRef getOpenMPNumTasksTypeName(NumTasksType);
LLVM_ABI OrderKind getOrderKind(StringRef);
LLVM_ABI llvm::StringRef getOpenMPOrderKindName(OrderKind);
LLVM_ABI ProcBindKind getProcBindKind(StringRef);
LLVM_ABI llvm::StringRef getOpenMPProcBindKindName(ProcBindKind);
LLVM_ABI ScheduleKind getScheduleKind(StringRef);
LLVM_ABI llvm::StringRef getOpenMPScheduleKindName(ScheduleKind);

} // namespace omp
} // namespace llvm
#endif // LLVM_OpenMP_INC
