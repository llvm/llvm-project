#ifdef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#undef GEN_FLANG_DIRECTIVE_CLAUSE_SETS

namespace llvm {
namespace omp {

  // Sets for allocate

  static OmpClauseSet allowedClauses_OMPD_allocate {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_allocate {
    llvm::omp::Clause::OMPC_align,
    llvm::omp::Clause::OMPC_allocator,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_allocate {
  };

  static OmpClauseSet requiredClauses_OMPD_allocate {
  };

  // Sets for allocators

  static OmpClauseSet allowedClauses_OMPD_allocators {
    llvm::omp::Clause::OMPC_allocate,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_allocators {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_allocators {
  };

  static OmpClauseSet requiredClauses_OMPD_allocators {
  };

  // Sets for assume

  static OmpClauseSet allowedClauses_OMPD_assume {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_assume {
    llvm::omp::Clause::OMPC_absent,
    llvm::omp::Clause::OMPC_contains,
    llvm::omp::Clause::OMPC_holds,
    llvm::omp::Clause::OMPC_no_openmp,
    llvm::omp::Clause::OMPC_no_openmp_routines,
    llvm::omp::Clause::OMPC_no_parallelism,
    llvm::omp::Clause::OMPC_no_openmp_constructs,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_assume {
  };

  static OmpClauseSet requiredClauses_OMPD_assume {
  };

  // Sets for assumes

  static OmpClauseSet allowedClauses_OMPD_assumes {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_assumes {
    llvm::omp::Clause::OMPC_absent,
    llvm::omp::Clause::OMPC_contains,
    llvm::omp::Clause::OMPC_holds,
    llvm::omp::Clause::OMPC_no_openmp,
    llvm::omp::Clause::OMPC_no_openmp_routines,
    llvm::omp::Clause::OMPC_no_parallelism,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_assumes {
  };

  static OmpClauseSet requiredClauses_OMPD_assumes {
  };

  // Sets for atomic

  static OmpClauseSet allowedClauses_OMPD_atomic {
    llvm::omp::Clause::OMPC_capture,
    llvm::omp::Clause::OMPC_compare,
    llvm::omp::Clause::OMPC_read,
    llvm::omp::Clause::OMPC_update,
    llvm::omp::Clause::OMPC_write,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_atomic {
    llvm::omp::Clause::OMPC_acq_rel,
    llvm::omp::Clause::OMPC_acquire,
    llvm::omp::Clause::OMPC_fail,
    llvm::omp::Clause::OMPC_hint,
    llvm::omp::Clause::OMPC_relaxed,
    llvm::omp::Clause::OMPC_release,
    llvm::omp::Clause::OMPC_seq_cst,
    llvm::omp::Clause::OMPC_weak,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_atomic {
  };

  static OmpClauseSet requiredClauses_OMPD_atomic {
  };

  // Sets for barrier

  static OmpClauseSet allowedClauses_OMPD_barrier {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_barrier {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_barrier {
  };

  static OmpClauseSet requiredClauses_OMPD_barrier {
  };

  // Sets for begin assumes

  static OmpClauseSet allowedClauses_OMPD_begin_assumes {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_begin_assumes {
    llvm::omp::Clause::OMPC_absent,
    llvm::omp::Clause::OMPC_contains,
    llvm::omp::Clause::OMPC_holds,
    llvm::omp::Clause::OMPC_no_openmp,
    llvm::omp::Clause::OMPC_no_openmp_routines,
    llvm::omp::Clause::OMPC_no_parallelism,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_begin_assumes {
  };

  static OmpClauseSet requiredClauses_OMPD_begin_assumes {
  };

  // Sets for begin declare target

  static OmpClauseSet allowedClauses_OMPD_begin_declare_target {
    llvm::omp::Clause::OMPC_device_type,
    llvm::omp::Clause::OMPC_indirect,
    llvm::omp::Clause::OMPC_link,
    llvm::omp::Clause::OMPC_to,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_begin_declare_target {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_begin_declare_target {
  };

  static OmpClauseSet requiredClauses_OMPD_begin_declare_target {
  };

  // Sets for begin declare variant

  static OmpClauseSet allowedClauses_OMPD_begin_declare_variant {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_begin_declare_variant {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_begin_declare_variant {
  };

  static OmpClauseSet requiredClauses_OMPD_begin_declare_variant {
  };

  // Sets for cancel

  static OmpClauseSet allowedClauses_OMPD_cancel {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_cancel {
    llvm::omp::Clause::OMPC_cancellation_construct_type,
    llvm::omp::Clause::OMPC_if,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_cancel {
  };

  static OmpClauseSet requiredClauses_OMPD_cancel {
  };

  // Sets for cancellation point

  static OmpClauseSet allowedClauses_OMPD_cancellation_point {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_cancellation_point {
    llvm::omp::Clause::OMPC_cancellation_construct_type,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_cancellation_point {
  };

  static OmpClauseSet requiredClauses_OMPD_cancellation_point {
  };

  // Sets for critical

  static OmpClauseSet allowedClauses_OMPD_critical {
    llvm::omp::Clause::OMPC_hint,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_critical {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_critical {
  };

  static OmpClauseSet requiredClauses_OMPD_critical {
  };

  // Sets for declare mapper

  static OmpClauseSet allowedClauses_OMPD_declare_mapper {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_declare_mapper {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_declare_mapper {
  };

  static OmpClauseSet requiredClauses_OMPD_declare_mapper {
    llvm::omp::Clause::OMPC_map,
  };

  // Sets for declare reduction

  static OmpClauseSet allowedClauses_OMPD_declare_reduction {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_declare_reduction {
    llvm::omp::Clause::OMPC_initializer,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_declare_reduction {
  };

  static OmpClauseSet requiredClauses_OMPD_declare_reduction {
  };

  // Sets for declare simd

  static OmpClauseSet allowedClauses_OMPD_declare_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_uniform,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_declare_simd {
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_declare_simd {
    llvm::omp::Clause::OMPC_inbranch,
    llvm::omp::Clause::OMPC_notinbranch,
  };

  static OmpClauseSet requiredClauses_OMPD_declare_simd {
  };

  // Sets for declare target

  static OmpClauseSet allowedClauses_OMPD_declare_target {
    llvm::omp::Clause::OMPC_enter,
    llvm::omp::Clause::OMPC_indirect,
    llvm::omp::Clause::OMPC_link,
    llvm::omp::Clause::OMPC_to,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_declare_target {
    llvm::omp::Clause::OMPC_device_type,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_declare_target {
  };

  static OmpClauseSet requiredClauses_OMPD_declare_target {
  };

  // Sets for declare variant

  static OmpClauseSet allowedClauses_OMPD_declare_variant {
    llvm::omp::Clause::OMPC_match,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_declare_variant {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_declare_variant {
    llvm::omp::Clause::OMPC_adjust_args,
    llvm::omp::Clause::OMPC_append_args,
  };

  static OmpClauseSet requiredClauses_OMPD_declare_variant {
  };

  // Sets for depobj

  static OmpClauseSet allowedClauses_OMPD_depobj {
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_depobj,
    llvm::omp::Clause::OMPC_destroy,
    llvm::omp::Clause::OMPC_update,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_depobj {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_depobj {
  };

  static OmpClauseSet requiredClauses_OMPD_depobj {
  };

  // Sets for distribute

  static OmpClauseSet allowedClauses_OMPD_distribute {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_private,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_distribute {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_order,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_distribute {
  };

  static OmpClauseSet requiredClauses_OMPD_distribute {
  };

  // Sets for distribute parallel do

  static OmpClauseSet allowedClauses_OMPD_distribute_parallel_do {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_distribute_parallel_do {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_schedule,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_distribute_parallel_do {
  };

  static OmpClauseSet requiredClauses_OMPD_distribute_parallel_do {
  };

  // Sets for distribute parallel do simd

  static OmpClauseSet allowedClauses_OMPD_distribute_parallel_do_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_distribute_parallel_do_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_distribute_parallel_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_distribute_parallel_do_simd {
  };

  // Sets for distribute parallel for

  static OmpClauseSet allowedClauses_OMPD_distribute_parallel_for {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_distribute_parallel_for {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_distribute_parallel_for {
  };

  static OmpClauseSet requiredClauses_OMPD_distribute_parallel_for {
  };

  // Sets for distribute parallel for simd

  static OmpClauseSet allowedClauses_OMPD_distribute_parallel_for_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_distribute_parallel_for_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_distribute_parallel_for_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_distribute_parallel_for_simd {
  };

  // Sets for distribute simd

  static OmpClauseSet allowedClauses_OMPD_distribute_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_distribute_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_distribute_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_distribute_simd {
  };

  // Sets for do

  static OmpClauseSet allowedClauses_OMPD_do {
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_do {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_schedule,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_do {
  };

  static OmpClauseSet requiredClauses_OMPD_do {
  };

  // Sets for do simd

  static OmpClauseSet allowedClauses_OMPD_do_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_do_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_do_simd {
  };

  // Sets for end assumes

  static OmpClauseSet allowedClauses_OMPD_end_assumes {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_assumes {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_assumes {
  };

  static OmpClauseSet requiredClauses_OMPD_end_assumes {
  };

  // Sets for end declare target

  static OmpClauseSet allowedClauses_OMPD_end_declare_target {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_declare_target {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_declare_target {
  };

  static OmpClauseSet requiredClauses_OMPD_end_declare_target {
  };

  // Sets for end declare variant

  static OmpClauseSet allowedClauses_OMPD_end_declare_variant {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_declare_variant {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_declare_variant {
  };

  static OmpClauseSet requiredClauses_OMPD_end_declare_variant {
  };

  // Sets for end do

  static OmpClauseSet allowedClauses_OMPD_end_do {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_do {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_do {
  };

  static OmpClauseSet requiredClauses_OMPD_end_do {
  };

  // Sets for end do simd

  static OmpClauseSet allowedClauses_OMPD_end_do_simd {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_do_simd {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_end_do_simd {
  };

  // Sets for end scope

  static OmpClauseSet allowedClauses_OMPD_end_scope {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_scope {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_scope {
  };

  static OmpClauseSet requiredClauses_OMPD_end_scope {
  };

  // Sets for end sections

  static OmpClauseSet allowedClauses_OMPD_end_sections {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_sections {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_sections {
  };

  static OmpClauseSet requiredClauses_OMPD_end_sections {
  };

  // Sets for end single

  static OmpClauseSet allowedClauses_OMPD_end_single {
    llvm::omp::Clause::OMPC_copyprivate,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_single {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_single {
  };

  static OmpClauseSet requiredClauses_OMPD_end_single {
  };

  // Sets for end workshare

  static OmpClauseSet allowedClauses_OMPD_end_workshare {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_end_workshare {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_end_workshare {
  };

  static OmpClauseSet requiredClauses_OMPD_end_workshare {
  };

  // Sets for error

  static OmpClauseSet allowedClauses_OMPD_error {
    llvm::omp::Clause::OMPC_at,
    llvm::omp::Clause::OMPC_message,
    llvm::omp::Clause::OMPC_severity,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_error {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_error {
  };

  static OmpClauseSet requiredClauses_OMPD_error {
  };

  // Sets for flush

  static OmpClauseSet allowedClauses_OMPD_flush {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_flush {
    llvm::omp::Clause::OMPC_acq_rel,
    llvm::omp::Clause::OMPC_acquire,
    llvm::omp::Clause::OMPC_flush,
    llvm::omp::Clause::OMPC_release,
    llvm::omp::Clause::OMPC_seq_cst,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_flush {
  };

  static OmpClauseSet requiredClauses_OMPD_flush {
  };

  // Sets for for

  static OmpClauseSet allowedClauses_OMPD_for {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_schedule,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_for {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_for {
  };

  static OmpClauseSet requiredClauses_OMPD_for {
  };

  // Sets for for simd

  static OmpClauseSet allowedClauses_OMPD_for_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_for_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_for_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_for_simd {
  };

  // Sets for interchange

  static OmpClauseSet allowedClauses_OMPD_interchange {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_interchange {
    llvm::omp::Clause::OMPC_permutation,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_interchange {
  };

  static OmpClauseSet requiredClauses_OMPD_interchange {
  };

  // Sets for masked taskloop

  static OmpClauseSet allowedClauses_OMPD_masked_taskloop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_filter,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_masked_taskloop {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_masked_taskloop {
  };

  static OmpClauseSet requiredClauses_OMPD_masked_taskloop {
  };

  // Sets for masked taskloop simd

  static OmpClauseSet allowedClauses_OMPD_masked_taskloop_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_filter,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_masked_taskloop_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_masked_taskloop_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_masked_taskloop_simd {
  };

  // Sets for master

  static OmpClauseSet allowedClauses_OMPD_master {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_master {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_master {
  };

  static OmpClauseSet requiredClauses_OMPD_master {
  };

  // Sets for master taskloop

  static OmpClauseSet allowedClauses_OMPD_master_taskloop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_master_taskloop {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_master_taskloop {
  };

  static OmpClauseSet requiredClauses_OMPD_master_taskloop {
  };

  // Sets for master taskloop simd

  static OmpClauseSet allowedClauses_OMPD_master_taskloop_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_master_taskloop_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_master_taskloop_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_master_taskloop_simd {
  };

  // Sets for metadirective

  static OmpClauseSet allowedClauses_OMPD_metadirective {
    llvm::omp::Clause::OMPC_when,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_metadirective {
    llvm::omp::Clause::OMPC_otherwise,
    llvm::omp::Clause::OMPC_default,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_metadirective {
  };

  static OmpClauseSet requiredClauses_OMPD_metadirective {
  };

  // Sets for nothing

  static OmpClauseSet allowedClauses_OMPD_nothing {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_nothing {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_nothing {
  };

  static OmpClauseSet requiredClauses_OMPD_nothing {
  };

  // Sets for ordered

  static OmpClauseSet allowedClauses_OMPD_ordered {
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_doacross,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_ordered {
    llvm::omp::Clause::OMPC_simd,
    llvm::omp::Clause::OMPC_threads,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_ordered {
  };

  static OmpClauseSet requiredClauses_OMPD_ordered {
  };

  // Sets for parallel

  static OmpClauseSet allowedClauses_OMPD_parallel {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel {
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_proc_bind,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel {
  };

  // Sets for parallel do

  static OmpClauseSet allowedClauses_OMPD_parallel_do {
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_do {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_schedule,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_do {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_do {
  };

  // Sets for parallel do simd

  static OmpClauseSet allowedClauses_OMPD_parallel_do_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_do_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_do_simd {
  };

  // Sets for parallel for

  static OmpClauseSet allowedClauses_OMPD_parallel_for {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_for {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_for {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_for {
  };

  // Sets for parallel for simd

  static OmpClauseSet allowedClauses_OMPD_parallel_for_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_for_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_for_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_for_simd {
  };

  // Sets for parallel masked

  static OmpClauseSet allowedClauses_OMPD_parallel_masked {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_filter,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_masked {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_masked {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_masked {
  };

  // Sets for parallel masked taskloop

  static OmpClauseSet allowedClauses_OMPD_parallel_masked_taskloop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_filter,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_masked_taskloop {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_masked_taskloop {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_masked_taskloop {
  };

  // Sets for parallel masked taskloop simd

  static OmpClauseSet allowedClauses_OMPD_parallel_masked_taskloop_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_filter,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_masked_taskloop_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_masked_taskloop_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_masked_taskloop_simd {
  };

  // Sets for parallel master

  static OmpClauseSet allowedClauses_OMPD_parallel_master {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_master {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_master {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_master {
  };

  // Sets for parallel master taskloop

  static OmpClauseSet allowedClauses_OMPD_parallel_master_taskloop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_master_taskloop {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_master_taskloop {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_master_taskloop {
  };

  // Sets for parallel master taskloop simd

  static OmpClauseSet allowedClauses_OMPD_parallel_master_taskloop_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_tasks,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_master_taskloop_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_master_taskloop_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_master_taskloop_simd {
  };

  // Sets for parallel sections

  static OmpClauseSet allowedClauses_OMPD_parallel_sections {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_sections {
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_sections {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_sections {
  };

  // Sets for parallel workshare

  static OmpClauseSet allowedClauses_OMPD_parallel_workshare {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_workshare {
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_proc_bind,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_workshare {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_workshare {
  };

  // Sets for requires

  static OmpClauseSet allowedClauses_OMPD_requires {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_requires {
    llvm::omp::Clause::OMPC_unified_address,
    llvm::omp::Clause::OMPC_unified_shared_memory,
    llvm::omp::Clause::OMPC_atomic_default_mem_order,
    llvm::omp::Clause::OMPC_dynamic_allocators,
    llvm::omp::Clause::OMPC_self_maps,
    llvm::omp::Clause::OMPC_reverse_offload,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_requires {
  };

  static OmpClauseSet requiredClauses_OMPD_requires {
  };

  // Sets for reverse

  static OmpClauseSet allowedClauses_OMPD_reverse {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_reverse {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_reverse {
  };

  static OmpClauseSet requiredClauses_OMPD_reverse {
  };

  // Sets for scan

  static OmpClauseSet allowedClauses_OMPD_scan {
    llvm::omp::Clause::OMPC_exclusive,
    llvm::omp::Clause::OMPC_inclusive,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_scan {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_scan {
  };

  static OmpClauseSet requiredClauses_OMPD_scan {
  };

  // Sets for scope

  static OmpClauseSet allowedClauses_OMPD_scope {
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_allocate,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_scope {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_scope {
  };

  static OmpClauseSet requiredClauses_OMPD_scope {
  };

  // Sets for section

  static OmpClauseSet allowedClauses_OMPD_section {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_section {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_section {
  };

  static OmpClauseSet requiredClauses_OMPD_section {
  };

  // Sets for sections

  static OmpClauseSet allowedClauses_OMPD_sections {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_sections {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_sections {
  };

  static OmpClauseSet requiredClauses_OMPD_sections {
  };

  // Sets for simd

  static OmpClauseSet allowedClauses_OMPD_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_simd {
  };

  // Sets for single

  static OmpClauseSet allowedClauses_OMPD_single {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyprivate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_private,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_single {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_single {
  };

  static OmpClauseSet requiredClauses_OMPD_single {
  };

  // Sets for stripe

  static OmpClauseSet allowedClauses_OMPD_stripe {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_stripe {
    llvm::omp::Clause::OMPC_sizes,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_stripe {
  };

  static OmpClauseSet requiredClauses_OMPD_stripe {
  };

  // Sets for target

  static OmpClauseSet allowedClauses_OMPD_target {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target {
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_ompx_bare,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target {
  };

  static OmpClauseSet requiredClauses_OMPD_target {
  };

  // Sets for target data

  static OmpClauseSet allowedClauses_OMPD_target_data {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_data {
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_if,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_data {
  };

  static OmpClauseSet requiredClauses_OMPD_target_data {
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_use_device_addr,
    llvm::omp::Clause::OMPC_use_device_ptr,
  };

  // Sets for target enter data

  static OmpClauseSet allowedClauses_OMPD_target_enter_data {
    llvm::omp::Clause::OMPC_depend,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_enter_data {
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_enter_data {
  };

  static OmpClauseSet requiredClauses_OMPD_target_enter_data {
    llvm::omp::Clause::OMPC_map,
  };

  // Sets for target exit data

  static OmpClauseSet allowedClauses_OMPD_target_exit_data {
    llvm::omp::Clause::OMPC_depend,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_exit_data {
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_exit_data {
  };

  static OmpClauseSet requiredClauses_OMPD_target_exit_data {
    llvm::omp::Clause::OMPC_map,
  };

  // Sets for target parallel

  static OmpClauseSet allowedClauses_OMPD_target_parallel {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_parallel {
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_parallel {
  };

  static OmpClauseSet requiredClauses_OMPD_target_parallel {
  };

  // Sets for target parallel do

  static OmpClauseSet allowedClauses_OMPD_target_parallel_do {
    llvm::omp::Clause::OMPC_allocator,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_parallel_do {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_schedule,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_parallel_do {
  };

  static OmpClauseSet requiredClauses_OMPD_target_parallel_do {
  };

  // Sets for target parallel do simd

  static OmpClauseSet allowedClauses_OMPD_target_parallel_do_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_parallel_do_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_parallel_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_target_parallel_do_simd {
  };

  // Sets for target parallel for

  static OmpClauseSet allowedClauses_OMPD_target_parallel_for {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_parallel_for {
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_parallel_for {
  };

  static OmpClauseSet requiredClauses_OMPD_target_parallel_for {
  };

  // Sets for target parallel for simd

  static OmpClauseSet allowedClauses_OMPD_target_parallel_for_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_ordered,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_parallel_for_simd {
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_parallel_for_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_target_parallel_for_simd {
  };

  // Sets for target simd

  static OmpClauseSet allowedClauses_OMPD_target_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_target_simd {
  };

  // Sets for target teams

  static OmpClauseSet allowedClauses_OMPD_target_teams {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams {
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_ompx_bare,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams {
  };

  // Sets for target teams distribute

  static OmpClauseSet allowedClauses_OMPD_target_teams_distribute {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_distribute {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_distribute {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_distribute {
  };

  // Sets for target teams distribute parallel do

  static OmpClauseSet allowedClauses_OMPD_target_teams_distribute_parallel_do {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_distribute_parallel_do {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_do {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_distribute_parallel_do {
  };

  // Sets for target teams distribute parallel do simd

  static OmpClauseSet allowedClauses_OMPD_target_teams_distribute_parallel_do_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_distribute_parallel_do_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_distribute_parallel_do_simd {
  };

  // Sets for target teams distribute parallel for

  static OmpClauseSet allowedClauses_OMPD_target_teams_distribute_parallel_for {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_thread_limit,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_distribute_parallel_for {
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_for {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_distribute_parallel_for {
  };

  // Sets for target teams distribute parallel for simd

  static OmpClauseSet allowedClauses_OMPD_target_teams_distribute_parallel_for_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_distribute_parallel_for_simd {
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_for_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_distribute_parallel_for_simd {
  };

  // Sets for target teams distribute simd

  static OmpClauseSet allowedClauses_OMPD_target_teams_distribute_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_distribute_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_distribute_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_distribute_simd {
  };

  // Sets for target update

  static OmpClauseSet allowedClauses_OMPD_target_update {
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_from,
    llvm::omp::Clause::OMPC_to,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_update {
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_update {
  };

  static OmpClauseSet requiredClauses_OMPD_target_update {
  };

  // Sets for task

  static OmpClauseSet allowedClauses_OMPD_task {
    llvm::omp::Clause::OMPC_affinity,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_detach,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_task {
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_priority,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_task {
  };

  static OmpClauseSet requiredClauses_OMPD_task {
  };

  // Sets for taskgroup

  static OmpClauseSet allowedClauses_OMPD_taskgroup {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_task_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_taskgroup {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_taskgroup {
  };

  static OmpClauseSet requiredClauses_OMPD_taskgroup {
  };

  // Sets for taskloop

  static OmpClauseSet allowedClauses_OMPD_taskloop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_taskloop {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_priority,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_taskloop {
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_num_tasks,
  };

  static OmpClauseSet requiredClauses_OMPD_taskloop {
  };

  // Sets for taskloop simd

  static OmpClauseSet allowedClauses_OMPD_taskloop_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_in_reduction,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_mergeable,
    llvm::omp::Clause::OMPC_nogroup,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_untied,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_taskloop_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_final,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_priority,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_simdlen,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_taskloop_simd {
    llvm::omp::Clause::OMPC_grainsize,
    llvm::omp::Clause::OMPC_num_tasks,
  };

  static OmpClauseSet requiredClauses_OMPD_taskloop_simd {
  };

  // Sets for taskwait

  static OmpClauseSet allowedClauses_OMPD_taskwait {
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_taskwait {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_taskwait {
  };

  static OmpClauseSet requiredClauses_OMPD_taskwait {
  };

  // Sets for taskyield

  static OmpClauseSet allowedClauses_OMPD_taskyield {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_taskyield {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_taskyield {
  };

  static OmpClauseSet requiredClauses_OMPD_taskyield {
  };

  // Sets for teams

  static OmpClauseSet allowedClauses_OMPD_teams {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams {
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams {
  };

  static OmpClauseSet requiredClauses_OMPD_teams {
  };

  // Sets for teams distribute

  static OmpClauseSet allowedClauses_OMPD_teams_distribute {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_distribute {
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_order,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_distribute {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_distribute {
  };

  // Sets for teams distribute parallel do

  static OmpClauseSet allowedClauses_OMPD_teams_distribute_parallel_do {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_distribute_parallel_do {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_distribute_parallel_do {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_distribute_parallel_do {
  };

  // Sets for teams distribute parallel do simd

  static OmpClauseSet allowedClauses_OMPD_teams_distribute_parallel_do_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_distribute_parallel_do_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_distribute_parallel_do_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_distribute_parallel_do_simd {
  };

  // Sets for teams distribute parallel for

  static OmpClauseSet allowedClauses_OMPD_teams_distribute_parallel_for {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_distribute_parallel_for {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_distribute_parallel_for {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_distribute_parallel_for {
  };

  // Sets for teams distribute parallel for simd

  static OmpClauseSet allowedClauses_OMPD_teams_distribute_parallel_for_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_schedule,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_distribute_parallel_for_simd {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_distribute_parallel_for_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_distribute_parallel_for_simd {
  };

  // Sets for teams distribute simd

  static OmpClauseSet allowedClauses_OMPD_teams_distribute_simd {
    llvm::omp::Clause::OMPC_aligned,
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_linear,
    llvm::omp::Clause::OMPC_nontemporal,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_distribute_simd {
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_dist_schedule,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_safelen,
    llvm::omp::Clause::OMPC_simdlen,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_distribute_simd {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_distribute_simd {
  };

  // Sets for threadprivate

  static OmpClauseSet allowedClauses_OMPD_threadprivate {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_threadprivate {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_threadprivate {
  };

  static OmpClauseSet requiredClauses_OMPD_threadprivate {
  };

  // Sets for tile

  static OmpClauseSet allowedClauses_OMPD_tile {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_tile {
    llvm::omp::Clause::OMPC_sizes,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_tile {
  };

  static OmpClauseSet requiredClauses_OMPD_tile {
  };

  // Sets for unknown

  static OmpClauseSet allowedClauses_OMPD_unknown {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_unknown {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_unknown {
  };

  static OmpClauseSet requiredClauses_OMPD_unknown {
  };

  // Sets for unroll

  static OmpClauseSet allowedClauses_OMPD_unroll {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_unroll {
    llvm::omp::Clause::OMPC_full,
    llvm::omp::Clause::OMPC_partial,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_unroll {
  };

  static OmpClauseSet requiredClauses_OMPD_unroll {
  };

  // Sets for workshare

  static OmpClauseSet allowedClauses_OMPD_workshare {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_workshare {
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_workshare {
  };

  static OmpClauseSet requiredClauses_OMPD_workshare {
  };

  // Sets for dispatch

  static OmpClauseSet allowedClauses_OMPD_dispatch {
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_nocontext,
    llvm::omp::Clause::OMPC_novariants,
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_dispatch {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_dispatch {
  };

  static OmpClauseSet requiredClauses_OMPD_dispatch {
  };

  // Sets for interop

  static OmpClauseSet allowedClauses_OMPD_interop {
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_destroy,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_init,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_use,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_interop {
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_interop {
  };

  static OmpClauseSet requiredClauses_OMPD_interop {
  };

  // Sets for loop

  static OmpClauseSet allowedClauses_OMPD_loop {
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_loop {
    llvm::omp::Clause::OMPC_bind,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_order,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_loop {
  };

  static OmpClauseSet requiredClauses_OMPD_loop {
  };

  // Sets for masked

  static OmpClauseSet allowedClauses_OMPD_masked {
  };

  static OmpClauseSet allowedOnceClauses_OMPD_masked {
    llvm::omp::Clause::OMPC_filter,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_masked {
  };

  static OmpClauseSet requiredClauses_OMPD_masked {
  };

  // Sets for parallel loop

  static OmpClauseSet allowedClauses_OMPD_parallel_loop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_copyin,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_parallel_loop {
    llvm::omp::Clause::OMPC_bind,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_parallel_loop {
  };

  static OmpClauseSet requiredClauses_OMPD_parallel_loop {
  };

  // Sets for target loop

  static OmpClauseSet allowedClauses_OMPD_target_loop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_uses_allocators,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_in_reduction,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_loop {
    llvm::omp::Clause::OMPC_bind,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_thread_limit,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_nowait,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_loop {
  };

  static OmpClauseSet requiredClauses_OMPD_target_loop {
  };

  // Sets for target parallel loop

  static OmpClauseSet allowedClauses_OMPD_target_parallel_loop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_parallel_loop {
    llvm::omp::Clause::OMPC_bind,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_threads,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_proc_bind,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_parallel_loop {
  };

  static OmpClauseSet requiredClauses_OMPD_target_parallel_loop {
  };

  // Sets for target teams loop

  static OmpClauseSet allowedClauses_OMPD_target_teams_loop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_defaultmap,
    llvm::omp::Clause::OMPC_depend,
    llvm::omp::Clause::OMPC_device,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_has_device_addr,
    llvm::omp::Clause::OMPC_if,
    llvm::omp::Clause::OMPC_is_device_ptr,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_map,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
    llvm::omp::Clause::OMPC_uses_allocators,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_target_teams_loop {
    llvm::omp::Clause::OMPC_bind,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_nowait,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_target_teams_loop {
  };

  static OmpClauseSet requiredClauses_OMPD_target_teams_loop {
  };

  // Sets for teams loop

  static OmpClauseSet allowedClauses_OMPD_teams_loop {
    llvm::omp::Clause::OMPC_allocate,
    llvm::omp::Clause::OMPC_firstprivate,
    llvm::omp::Clause::OMPC_lastprivate,
    llvm::omp::Clause::OMPC_ompx_attribute,
    llvm::omp::Clause::OMPC_private,
    llvm::omp::Clause::OMPC_reduction,
    llvm::omp::Clause::OMPC_shared,
  };

  static OmpClauseSet allowedOnceClauses_OMPD_teams_loop {
    llvm::omp::Clause::OMPC_bind,
    llvm::omp::Clause::OMPC_collapse,
    llvm::omp::Clause::OMPC_default,
    llvm::omp::Clause::OMPC_num_teams,
    llvm::omp::Clause::OMPC_order,
    llvm::omp::Clause::OMPC_thread_limit,
  };

  static OmpClauseSet allowedExclusiveClauses_OMPD_teams_loop {
  };

  static OmpClauseSet requiredClauses_OMPD_teams_loop {
  };
} // namespace omp
} // namespace llvm

#endif // GEN_FLANG_DIRECTIVE_CLAUSE_SETS

#ifdef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#undef GEN_FLANG_DIRECTIVE_CLAUSE_MAP

{
  {llvm::omp::Directive::OMPD_allocate,
    {
      llvm::omp::allowedClauses_OMPD_allocate,
      llvm::omp::allowedOnceClauses_OMPD_allocate,
      llvm::omp::allowedExclusiveClauses_OMPD_allocate,
      llvm::omp::requiredClauses_OMPD_allocate,
    }
  },
  {llvm::omp::Directive::OMPD_allocators,
    {
      llvm::omp::allowedClauses_OMPD_allocators,
      llvm::omp::allowedOnceClauses_OMPD_allocators,
      llvm::omp::allowedExclusiveClauses_OMPD_allocators,
      llvm::omp::requiredClauses_OMPD_allocators,
    }
  },
  {llvm::omp::Directive::OMPD_assume,
    {
      llvm::omp::allowedClauses_OMPD_assume,
      llvm::omp::allowedOnceClauses_OMPD_assume,
      llvm::omp::allowedExclusiveClauses_OMPD_assume,
      llvm::omp::requiredClauses_OMPD_assume,
    }
  },
  {llvm::omp::Directive::OMPD_assumes,
    {
      llvm::omp::allowedClauses_OMPD_assumes,
      llvm::omp::allowedOnceClauses_OMPD_assumes,
      llvm::omp::allowedExclusiveClauses_OMPD_assumes,
      llvm::omp::requiredClauses_OMPD_assumes,
    }
  },
  {llvm::omp::Directive::OMPD_atomic,
    {
      llvm::omp::allowedClauses_OMPD_atomic,
      llvm::omp::allowedOnceClauses_OMPD_atomic,
      llvm::omp::allowedExclusiveClauses_OMPD_atomic,
      llvm::omp::requiredClauses_OMPD_atomic,
    }
  },
  {llvm::omp::Directive::OMPD_barrier,
    {
      llvm::omp::allowedClauses_OMPD_barrier,
      llvm::omp::allowedOnceClauses_OMPD_barrier,
      llvm::omp::allowedExclusiveClauses_OMPD_barrier,
      llvm::omp::requiredClauses_OMPD_barrier,
    }
  },
  {llvm::omp::Directive::OMPD_begin_assumes,
    {
      llvm::omp::allowedClauses_OMPD_begin_assumes,
      llvm::omp::allowedOnceClauses_OMPD_begin_assumes,
      llvm::omp::allowedExclusiveClauses_OMPD_begin_assumes,
      llvm::omp::requiredClauses_OMPD_begin_assumes,
    }
  },
  {llvm::omp::Directive::OMPD_begin_declare_target,
    {
      llvm::omp::allowedClauses_OMPD_begin_declare_target,
      llvm::omp::allowedOnceClauses_OMPD_begin_declare_target,
      llvm::omp::allowedExclusiveClauses_OMPD_begin_declare_target,
      llvm::omp::requiredClauses_OMPD_begin_declare_target,
    }
  },
  {llvm::omp::Directive::OMPD_begin_declare_variant,
    {
      llvm::omp::allowedClauses_OMPD_begin_declare_variant,
      llvm::omp::allowedOnceClauses_OMPD_begin_declare_variant,
      llvm::omp::allowedExclusiveClauses_OMPD_begin_declare_variant,
      llvm::omp::requiredClauses_OMPD_begin_declare_variant,
    }
  },
  {llvm::omp::Directive::OMPD_cancel,
    {
      llvm::omp::allowedClauses_OMPD_cancel,
      llvm::omp::allowedOnceClauses_OMPD_cancel,
      llvm::omp::allowedExclusiveClauses_OMPD_cancel,
      llvm::omp::requiredClauses_OMPD_cancel,
    }
  },
  {llvm::omp::Directive::OMPD_cancellation_point,
    {
      llvm::omp::allowedClauses_OMPD_cancellation_point,
      llvm::omp::allowedOnceClauses_OMPD_cancellation_point,
      llvm::omp::allowedExclusiveClauses_OMPD_cancellation_point,
      llvm::omp::requiredClauses_OMPD_cancellation_point,
    }
  },
  {llvm::omp::Directive::OMPD_critical,
    {
      llvm::omp::allowedClauses_OMPD_critical,
      llvm::omp::allowedOnceClauses_OMPD_critical,
      llvm::omp::allowedExclusiveClauses_OMPD_critical,
      llvm::omp::requiredClauses_OMPD_critical,
    }
  },
  {llvm::omp::Directive::OMPD_declare_mapper,
    {
      llvm::omp::allowedClauses_OMPD_declare_mapper,
      llvm::omp::allowedOnceClauses_OMPD_declare_mapper,
      llvm::omp::allowedExclusiveClauses_OMPD_declare_mapper,
      llvm::omp::requiredClauses_OMPD_declare_mapper,
    }
  },
  {llvm::omp::Directive::OMPD_declare_reduction,
    {
      llvm::omp::allowedClauses_OMPD_declare_reduction,
      llvm::omp::allowedOnceClauses_OMPD_declare_reduction,
      llvm::omp::allowedExclusiveClauses_OMPD_declare_reduction,
      llvm::omp::requiredClauses_OMPD_declare_reduction,
    }
  },
  {llvm::omp::Directive::OMPD_declare_simd,
    {
      llvm::omp::allowedClauses_OMPD_declare_simd,
      llvm::omp::allowedOnceClauses_OMPD_declare_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_declare_simd,
      llvm::omp::requiredClauses_OMPD_declare_simd,
    }
  },
  {llvm::omp::Directive::OMPD_declare_target,
    {
      llvm::omp::allowedClauses_OMPD_declare_target,
      llvm::omp::allowedOnceClauses_OMPD_declare_target,
      llvm::omp::allowedExclusiveClauses_OMPD_declare_target,
      llvm::omp::requiredClauses_OMPD_declare_target,
    }
  },
  {llvm::omp::Directive::OMPD_declare_variant,
    {
      llvm::omp::allowedClauses_OMPD_declare_variant,
      llvm::omp::allowedOnceClauses_OMPD_declare_variant,
      llvm::omp::allowedExclusiveClauses_OMPD_declare_variant,
      llvm::omp::requiredClauses_OMPD_declare_variant,
    }
  },
  {llvm::omp::Directive::OMPD_depobj,
    {
      llvm::omp::allowedClauses_OMPD_depobj,
      llvm::omp::allowedOnceClauses_OMPD_depobj,
      llvm::omp::allowedExclusiveClauses_OMPD_depobj,
      llvm::omp::requiredClauses_OMPD_depobj,
    }
  },
  {llvm::omp::Directive::OMPD_distribute,
    {
      llvm::omp::allowedClauses_OMPD_distribute,
      llvm::omp::allowedOnceClauses_OMPD_distribute,
      llvm::omp::allowedExclusiveClauses_OMPD_distribute,
      llvm::omp::requiredClauses_OMPD_distribute,
    }
  },
  {llvm::omp::Directive::OMPD_distribute_parallel_do,
    {
      llvm::omp::allowedClauses_OMPD_distribute_parallel_do,
      llvm::omp::allowedOnceClauses_OMPD_distribute_parallel_do,
      llvm::omp::allowedExclusiveClauses_OMPD_distribute_parallel_do,
      llvm::omp::requiredClauses_OMPD_distribute_parallel_do,
    }
  },
  {llvm::omp::Directive::OMPD_distribute_parallel_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_distribute_parallel_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_distribute_parallel_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_distribute_parallel_do_simd,
      llvm::omp::requiredClauses_OMPD_distribute_parallel_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_distribute_parallel_for,
    {
      llvm::omp::allowedClauses_OMPD_distribute_parallel_for,
      llvm::omp::allowedOnceClauses_OMPD_distribute_parallel_for,
      llvm::omp::allowedExclusiveClauses_OMPD_distribute_parallel_for,
      llvm::omp::requiredClauses_OMPD_distribute_parallel_for,
    }
  },
  {llvm::omp::Directive::OMPD_distribute_parallel_for_simd,
    {
      llvm::omp::allowedClauses_OMPD_distribute_parallel_for_simd,
      llvm::omp::allowedOnceClauses_OMPD_distribute_parallel_for_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_distribute_parallel_for_simd,
      llvm::omp::requiredClauses_OMPD_distribute_parallel_for_simd,
    }
  },
  {llvm::omp::Directive::OMPD_distribute_simd,
    {
      llvm::omp::allowedClauses_OMPD_distribute_simd,
      llvm::omp::allowedOnceClauses_OMPD_distribute_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_distribute_simd,
      llvm::omp::requiredClauses_OMPD_distribute_simd,
    }
  },
  {llvm::omp::Directive::OMPD_do,
    {
      llvm::omp::allowedClauses_OMPD_do,
      llvm::omp::allowedOnceClauses_OMPD_do,
      llvm::omp::allowedExclusiveClauses_OMPD_do,
      llvm::omp::requiredClauses_OMPD_do,
    }
  },
  {llvm::omp::Directive::OMPD_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_do_simd,
      llvm::omp::requiredClauses_OMPD_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_end_assumes,
    {
      llvm::omp::allowedClauses_OMPD_end_assumes,
      llvm::omp::allowedOnceClauses_OMPD_end_assumes,
      llvm::omp::allowedExclusiveClauses_OMPD_end_assumes,
      llvm::omp::requiredClauses_OMPD_end_assumes,
    }
  },
  {llvm::omp::Directive::OMPD_end_declare_target,
    {
      llvm::omp::allowedClauses_OMPD_end_declare_target,
      llvm::omp::allowedOnceClauses_OMPD_end_declare_target,
      llvm::omp::allowedExclusiveClauses_OMPD_end_declare_target,
      llvm::omp::requiredClauses_OMPD_end_declare_target,
    }
  },
  {llvm::omp::Directive::OMPD_end_declare_variant,
    {
      llvm::omp::allowedClauses_OMPD_end_declare_variant,
      llvm::omp::allowedOnceClauses_OMPD_end_declare_variant,
      llvm::omp::allowedExclusiveClauses_OMPD_end_declare_variant,
      llvm::omp::requiredClauses_OMPD_end_declare_variant,
    }
  },
  {llvm::omp::Directive::OMPD_end_do,
    {
      llvm::omp::allowedClauses_OMPD_end_do,
      llvm::omp::allowedOnceClauses_OMPD_end_do,
      llvm::omp::allowedExclusiveClauses_OMPD_end_do,
      llvm::omp::requiredClauses_OMPD_end_do,
    }
  },
  {llvm::omp::Directive::OMPD_end_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_end_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_end_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_end_do_simd,
      llvm::omp::requiredClauses_OMPD_end_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_end_scope,
    {
      llvm::omp::allowedClauses_OMPD_end_scope,
      llvm::omp::allowedOnceClauses_OMPD_end_scope,
      llvm::omp::allowedExclusiveClauses_OMPD_end_scope,
      llvm::omp::requiredClauses_OMPD_end_scope,
    }
  },
  {llvm::omp::Directive::OMPD_end_sections,
    {
      llvm::omp::allowedClauses_OMPD_end_sections,
      llvm::omp::allowedOnceClauses_OMPD_end_sections,
      llvm::omp::allowedExclusiveClauses_OMPD_end_sections,
      llvm::omp::requiredClauses_OMPD_end_sections,
    }
  },
  {llvm::omp::Directive::OMPD_end_single,
    {
      llvm::omp::allowedClauses_OMPD_end_single,
      llvm::omp::allowedOnceClauses_OMPD_end_single,
      llvm::omp::allowedExclusiveClauses_OMPD_end_single,
      llvm::omp::requiredClauses_OMPD_end_single,
    }
  },
  {llvm::omp::Directive::OMPD_end_workshare,
    {
      llvm::omp::allowedClauses_OMPD_end_workshare,
      llvm::omp::allowedOnceClauses_OMPD_end_workshare,
      llvm::omp::allowedExclusiveClauses_OMPD_end_workshare,
      llvm::omp::requiredClauses_OMPD_end_workshare,
    }
  },
  {llvm::omp::Directive::OMPD_error,
    {
      llvm::omp::allowedClauses_OMPD_error,
      llvm::omp::allowedOnceClauses_OMPD_error,
      llvm::omp::allowedExclusiveClauses_OMPD_error,
      llvm::omp::requiredClauses_OMPD_error,
    }
  },
  {llvm::omp::Directive::OMPD_flush,
    {
      llvm::omp::allowedClauses_OMPD_flush,
      llvm::omp::allowedOnceClauses_OMPD_flush,
      llvm::omp::allowedExclusiveClauses_OMPD_flush,
      llvm::omp::requiredClauses_OMPD_flush,
    }
  },
  {llvm::omp::Directive::OMPD_for,
    {
      llvm::omp::allowedClauses_OMPD_for,
      llvm::omp::allowedOnceClauses_OMPD_for,
      llvm::omp::allowedExclusiveClauses_OMPD_for,
      llvm::omp::requiredClauses_OMPD_for,
    }
  },
  {llvm::omp::Directive::OMPD_for_simd,
    {
      llvm::omp::allowedClauses_OMPD_for_simd,
      llvm::omp::allowedOnceClauses_OMPD_for_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_for_simd,
      llvm::omp::requiredClauses_OMPD_for_simd,
    }
  },
  {llvm::omp::Directive::OMPD_interchange,
    {
      llvm::omp::allowedClauses_OMPD_interchange,
      llvm::omp::allowedOnceClauses_OMPD_interchange,
      llvm::omp::allowedExclusiveClauses_OMPD_interchange,
      llvm::omp::requiredClauses_OMPD_interchange,
    }
  },
  {llvm::omp::Directive::OMPD_masked_taskloop,
    {
      llvm::omp::allowedClauses_OMPD_masked_taskloop,
      llvm::omp::allowedOnceClauses_OMPD_masked_taskloop,
      llvm::omp::allowedExclusiveClauses_OMPD_masked_taskloop,
      llvm::omp::requiredClauses_OMPD_masked_taskloop,
    }
  },
  {llvm::omp::Directive::OMPD_masked_taskloop_simd,
    {
      llvm::omp::allowedClauses_OMPD_masked_taskloop_simd,
      llvm::omp::allowedOnceClauses_OMPD_masked_taskloop_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_masked_taskloop_simd,
      llvm::omp::requiredClauses_OMPD_masked_taskloop_simd,
    }
  },
  {llvm::omp::Directive::OMPD_master,
    {
      llvm::omp::allowedClauses_OMPD_master,
      llvm::omp::allowedOnceClauses_OMPD_master,
      llvm::omp::allowedExclusiveClauses_OMPD_master,
      llvm::omp::requiredClauses_OMPD_master,
    }
  },
  {llvm::omp::Directive::OMPD_master_taskloop,
    {
      llvm::omp::allowedClauses_OMPD_master_taskloop,
      llvm::omp::allowedOnceClauses_OMPD_master_taskloop,
      llvm::omp::allowedExclusiveClauses_OMPD_master_taskloop,
      llvm::omp::requiredClauses_OMPD_master_taskloop,
    }
  },
  {llvm::omp::Directive::OMPD_master_taskloop_simd,
    {
      llvm::omp::allowedClauses_OMPD_master_taskloop_simd,
      llvm::omp::allowedOnceClauses_OMPD_master_taskloop_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_master_taskloop_simd,
      llvm::omp::requiredClauses_OMPD_master_taskloop_simd,
    }
  },
  {llvm::omp::Directive::OMPD_metadirective,
    {
      llvm::omp::allowedClauses_OMPD_metadirective,
      llvm::omp::allowedOnceClauses_OMPD_metadirective,
      llvm::omp::allowedExclusiveClauses_OMPD_metadirective,
      llvm::omp::requiredClauses_OMPD_metadirective,
    }
  },
  {llvm::omp::Directive::OMPD_nothing,
    {
      llvm::omp::allowedClauses_OMPD_nothing,
      llvm::omp::allowedOnceClauses_OMPD_nothing,
      llvm::omp::allowedExclusiveClauses_OMPD_nothing,
      llvm::omp::requiredClauses_OMPD_nothing,
    }
  },
  {llvm::omp::Directive::OMPD_ordered,
    {
      llvm::omp::allowedClauses_OMPD_ordered,
      llvm::omp::allowedOnceClauses_OMPD_ordered,
      llvm::omp::allowedExclusiveClauses_OMPD_ordered,
      llvm::omp::requiredClauses_OMPD_ordered,
    }
  },
  {llvm::omp::Directive::OMPD_parallel,
    {
      llvm::omp::allowedClauses_OMPD_parallel,
      llvm::omp::allowedOnceClauses_OMPD_parallel,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel,
      llvm::omp::requiredClauses_OMPD_parallel,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_do,
    {
      llvm::omp::allowedClauses_OMPD_parallel_do,
      llvm::omp::allowedOnceClauses_OMPD_parallel_do,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_do,
      llvm::omp::requiredClauses_OMPD_parallel_do,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_parallel_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_parallel_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_do_simd,
      llvm::omp::requiredClauses_OMPD_parallel_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_for,
    {
      llvm::omp::allowedClauses_OMPD_parallel_for,
      llvm::omp::allowedOnceClauses_OMPD_parallel_for,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_for,
      llvm::omp::requiredClauses_OMPD_parallel_for,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_for_simd,
    {
      llvm::omp::allowedClauses_OMPD_parallel_for_simd,
      llvm::omp::allowedOnceClauses_OMPD_parallel_for_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_for_simd,
      llvm::omp::requiredClauses_OMPD_parallel_for_simd,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_masked,
    {
      llvm::omp::allowedClauses_OMPD_parallel_masked,
      llvm::omp::allowedOnceClauses_OMPD_parallel_masked,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_masked,
      llvm::omp::requiredClauses_OMPD_parallel_masked,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_masked_taskloop,
    {
      llvm::omp::allowedClauses_OMPD_parallel_masked_taskloop,
      llvm::omp::allowedOnceClauses_OMPD_parallel_masked_taskloop,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_masked_taskloop,
      llvm::omp::requiredClauses_OMPD_parallel_masked_taskloop,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_masked_taskloop_simd,
    {
      llvm::omp::allowedClauses_OMPD_parallel_masked_taskloop_simd,
      llvm::omp::allowedOnceClauses_OMPD_parallel_masked_taskloop_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_masked_taskloop_simd,
      llvm::omp::requiredClauses_OMPD_parallel_masked_taskloop_simd,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_master,
    {
      llvm::omp::allowedClauses_OMPD_parallel_master,
      llvm::omp::allowedOnceClauses_OMPD_parallel_master,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_master,
      llvm::omp::requiredClauses_OMPD_parallel_master,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_master_taskloop,
    {
      llvm::omp::allowedClauses_OMPD_parallel_master_taskloop,
      llvm::omp::allowedOnceClauses_OMPD_parallel_master_taskloop,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_master_taskloop,
      llvm::omp::requiredClauses_OMPD_parallel_master_taskloop,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_master_taskloop_simd,
    {
      llvm::omp::allowedClauses_OMPD_parallel_master_taskloop_simd,
      llvm::omp::allowedOnceClauses_OMPD_parallel_master_taskloop_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_master_taskloop_simd,
      llvm::omp::requiredClauses_OMPD_parallel_master_taskloop_simd,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_sections,
    {
      llvm::omp::allowedClauses_OMPD_parallel_sections,
      llvm::omp::allowedOnceClauses_OMPD_parallel_sections,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_sections,
      llvm::omp::requiredClauses_OMPD_parallel_sections,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_workshare,
    {
      llvm::omp::allowedClauses_OMPD_parallel_workshare,
      llvm::omp::allowedOnceClauses_OMPD_parallel_workshare,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_workshare,
      llvm::omp::requiredClauses_OMPD_parallel_workshare,
    }
  },
  {llvm::omp::Directive::OMPD_requires,
    {
      llvm::omp::allowedClauses_OMPD_requires,
      llvm::omp::allowedOnceClauses_OMPD_requires,
      llvm::omp::allowedExclusiveClauses_OMPD_requires,
      llvm::omp::requiredClauses_OMPD_requires,
    }
  },
  {llvm::omp::Directive::OMPD_reverse,
    {
      llvm::omp::allowedClauses_OMPD_reverse,
      llvm::omp::allowedOnceClauses_OMPD_reverse,
      llvm::omp::allowedExclusiveClauses_OMPD_reverse,
      llvm::omp::requiredClauses_OMPD_reverse,
    }
  },
  {llvm::omp::Directive::OMPD_scan,
    {
      llvm::omp::allowedClauses_OMPD_scan,
      llvm::omp::allowedOnceClauses_OMPD_scan,
      llvm::omp::allowedExclusiveClauses_OMPD_scan,
      llvm::omp::requiredClauses_OMPD_scan,
    }
  },
  {llvm::omp::Directive::OMPD_scope,
    {
      llvm::omp::allowedClauses_OMPD_scope,
      llvm::omp::allowedOnceClauses_OMPD_scope,
      llvm::omp::allowedExclusiveClauses_OMPD_scope,
      llvm::omp::requiredClauses_OMPD_scope,
    }
  },
  {llvm::omp::Directive::OMPD_section,
    {
      llvm::omp::allowedClauses_OMPD_section,
      llvm::omp::allowedOnceClauses_OMPD_section,
      llvm::omp::allowedExclusiveClauses_OMPD_section,
      llvm::omp::requiredClauses_OMPD_section,
    }
  },
  {llvm::omp::Directive::OMPD_sections,
    {
      llvm::omp::allowedClauses_OMPD_sections,
      llvm::omp::allowedOnceClauses_OMPD_sections,
      llvm::omp::allowedExclusiveClauses_OMPD_sections,
      llvm::omp::requiredClauses_OMPD_sections,
    }
  },
  {llvm::omp::Directive::OMPD_simd,
    {
      llvm::omp::allowedClauses_OMPD_simd,
      llvm::omp::allowedOnceClauses_OMPD_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_simd,
      llvm::omp::requiredClauses_OMPD_simd,
    }
  },
  {llvm::omp::Directive::OMPD_single,
    {
      llvm::omp::allowedClauses_OMPD_single,
      llvm::omp::allowedOnceClauses_OMPD_single,
      llvm::omp::allowedExclusiveClauses_OMPD_single,
      llvm::omp::requiredClauses_OMPD_single,
    }
  },
  {llvm::omp::Directive::OMPD_stripe,
    {
      llvm::omp::allowedClauses_OMPD_stripe,
      llvm::omp::allowedOnceClauses_OMPD_stripe,
      llvm::omp::allowedExclusiveClauses_OMPD_stripe,
      llvm::omp::requiredClauses_OMPD_stripe,
    }
  },
  {llvm::omp::Directive::OMPD_target,
    {
      llvm::omp::allowedClauses_OMPD_target,
      llvm::omp::allowedOnceClauses_OMPD_target,
      llvm::omp::allowedExclusiveClauses_OMPD_target,
      llvm::omp::requiredClauses_OMPD_target,
    }
  },
  {llvm::omp::Directive::OMPD_target_data,
    {
      llvm::omp::allowedClauses_OMPD_target_data,
      llvm::omp::allowedOnceClauses_OMPD_target_data,
      llvm::omp::allowedExclusiveClauses_OMPD_target_data,
      llvm::omp::requiredClauses_OMPD_target_data,
    }
  },
  {llvm::omp::Directive::OMPD_target_enter_data,
    {
      llvm::omp::allowedClauses_OMPD_target_enter_data,
      llvm::omp::allowedOnceClauses_OMPD_target_enter_data,
      llvm::omp::allowedExclusiveClauses_OMPD_target_enter_data,
      llvm::omp::requiredClauses_OMPD_target_enter_data,
    }
  },
  {llvm::omp::Directive::OMPD_target_exit_data,
    {
      llvm::omp::allowedClauses_OMPD_target_exit_data,
      llvm::omp::allowedOnceClauses_OMPD_target_exit_data,
      llvm::omp::allowedExclusiveClauses_OMPD_target_exit_data,
      llvm::omp::requiredClauses_OMPD_target_exit_data,
    }
  },
  {llvm::omp::Directive::OMPD_target_parallel,
    {
      llvm::omp::allowedClauses_OMPD_target_parallel,
      llvm::omp::allowedOnceClauses_OMPD_target_parallel,
      llvm::omp::allowedExclusiveClauses_OMPD_target_parallel,
      llvm::omp::requiredClauses_OMPD_target_parallel,
    }
  },
  {llvm::omp::Directive::OMPD_target_parallel_do,
    {
      llvm::omp::allowedClauses_OMPD_target_parallel_do,
      llvm::omp::allowedOnceClauses_OMPD_target_parallel_do,
      llvm::omp::allowedExclusiveClauses_OMPD_target_parallel_do,
      llvm::omp::requiredClauses_OMPD_target_parallel_do,
    }
  },
  {llvm::omp::Directive::OMPD_target_parallel_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_target_parallel_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_target_parallel_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_target_parallel_do_simd,
      llvm::omp::requiredClauses_OMPD_target_parallel_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_target_parallel_for,
    {
      llvm::omp::allowedClauses_OMPD_target_parallel_for,
      llvm::omp::allowedOnceClauses_OMPD_target_parallel_for,
      llvm::omp::allowedExclusiveClauses_OMPD_target_parallel_for,
      llvm::omp::requiredClauses_OMPD_target_parallel_for,
    }
  },
  {llvm::omp::Directive::OMPD_target_parallel_for_simd,
    {
      llvm::omp::allowedClauses_OMPD_target_parallel_for_simd,
      llvm::omp::allowedOnceClauses_OMPD_target_parallel_for_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_target_parallel_for_simd,
      llvm::omp::requiredClauses_OMPD_target_parallel_for_simd,
    }
  },
  {llvm::omp::Directive::OMPD_target_simd,
    {
      llvm::omp::allowedClauses_OMPD_target_simd,
      llvm::omp::allowedOnceClauses_OMPD_target_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_target_simd,
      llvm::omp::requiredClauses_OMPD_target_simd,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams,
    {
      llvm::omp::allowedClauses_OMPD_target_teams,
      llvm::omp::allowedOnceClauses_OMPD_target_teams,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams,
      llvm::omp::requiredClauses_OMPD_target_teams,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_distribute,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_distribute,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_distribute,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_distribute,
      llvm::omp::requiredClauses_OMPD_target_teams_distribute,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_distribute_parallel_do,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_distribute_parallel_do,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_do,
      llvm::omp::requiredClauses_OMPD_target_teams_distribute_parallel_do,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_distribute_parallel_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_distribute_parallel_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_do_simd,
      llvm::omp::requiredClauses_OMPD_target_teams_distribute_parallel_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_distribute_parallel_for,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_distribute_parallel_for,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_distribute_parallel_for,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_for,
      llvm::omp::requiredClauses_OMPD_target_teams_distribute_parallel_for,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_distribute_parallel_for_simd,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_distribute_parallel_for_simd,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_distribute_parallel_for_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_distribute_parallel_for_simd,
      llvm::omp::requiredClauses_OMPD_target_teams_distribute_parallel_for_simd,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_distribute_simd,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_distribute_simd,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_distribute_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_distribute_simd,
      llvm::omp::requiredClauses_OMPD_target_teams_distribute_simd,
    }
  },
  {llvm::omp::Directive::OMPD_target_update,
    {
      llvm::omp::allowedClauses_OMPD_target_update,
      llvm::omp::allowedOnceClauses_OMPD_target_update,
      llvm::omp::allowedExclusiveClauses_OMPD_target_update,
      llvm::omp::requiredClauses_OMPD_target_update,
    }
  },
  {llvm::omp::Directive::OMPD_task,
    {
      llvm::omp::allowedClauses_OMPD_task,
      llvm::omp::allowedOnceClauses_OMPD_task,
      llvm::omp::allowedExclusiveClauses_OMPD_task,
      llvm::omp::requiredClauses_OMPD_task,
    }
  },
  {llvm::omp::Directive::OMPD_taskgroup,
    {
      llvm::omp::allowedClauses_OMPD_taskgroup,
      llvm::omp::allowedOnceClauses_OMPD_taskgroup,
      llvm::omp::allowedExclusiveClauses_OMPD_taskgroup,
      llvm::omp::requiredClauses_OMPD_taskgroup,
    }
  },
  {llvm::omp::Directive::OMPD_taskloop,
    {
      llvm::omp::allowedClauses_OMPD_taskloop,
      llvm::omp::allowedOnceClauses_OMPD_taskloop,
      llvm::omp::allowedExclusiveClauses_OMPD_taskloop,
      llvm::omp::requiredClauses_OMPD_taskloop,
    }
  },
  {llvm::omp::Directive::OMPD_taskloop_simd,
    {
      llvm::omp::allowedClauses_OMPD_taskloop_simd,
      llvm::omp::allowedOnceClauses_OMPD_taskloop_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_taskloop_simd,
      llvm::omp::requiredClauses_OMPD_taskloop_simd,
    }
  },
  {llvm::omp::Directive::OMPD_taskwait,
    {
      llvm::omp::allowedClauses_OMPD_taskwait,
      llvm::omp::allowedOnceClauses_OMPD_taskwait,
      llvm::omp::allowedExclusiveClauses_OMPD_taskwait,
      llvm::omp::requiredClauses_OMPD_taskwait,
    }
  },
  {llvm::omp::Directive::OMPD_taskyield,
    {
      llvm::omp::allowedClauses_OMPD_taskyield,
      llvm::omp::allowedOnceClauses_OMPD_taskyield,
      llvm::omp::allowedExclusiveClauses_OMPD_taskyield,
      llvm::omp::requiredClauses_OMPD_taskyield,
    }
  },
  {llvm::omp::Directive::OMPD_teams,
    {
      llvm::omp::allowedClauses_OMPD_teams,
      llvm::omp::allowedOnceClauses_OMPD_teams,
      llvm::omp::allowedExclusiveClauses_OMPD_teams,
      llvm::omp::requiredClauses_OMPD_teams,
    }
  },
  {llvm::omp::Directive::OMPD_teams_distribute,
    {
      llvm::omp::allowedClauses_OMPD_teams_distribute,
      llvm::omp::allowedOnceClauses_OMPD_teams_distribute,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_distribute,
      llvm::omp::requiredClauses_OMPD_teams_distribute,
    }
  },
  {llvm::omp::Directive::OMPD_teams_distribute_parallel_do,
    {
      llvm::omp::allowedClauses_OMPD_teams_distribute_parallel_do,
      llvm::omp::allowedOnceClauses_OMPD_teams_distribute_parallel_do,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_distribute_parallel_do,
      llvm::omp::requiredClauses_OMPD_teams_distribute_parallel_do,
    }
  },
  {llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd,
    {
      llvm::omp::allowedClauses_OMPD_teams_distribute_parallel_do_simd,
      llvm::omp::allowedOnceClauses_OMPD_teams_distribute_parallel_do_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_distribute_parallel_do_simd,
      llvm::omp::requiredClauses_OMPD_teams_distribute_parallel_do_simd,
    }
  },
  {llvm::omp::Directive::OMPD_teams_distribute_parallel_for,
    {
      llvm::omp::allowedClauses_OMPD_teams_distribute_parallel_for,
      llvm::omp::allowedOnceClauses_OMPD_teams_distribute_parallel_for,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_distribute_parallel_for,
      llvm::omp::requiredClauses_OMPD_teams_distribute_parallel_for,
    }
  },
  {llvm::omp::Directive::OMPD_teams_distribute_parallel_for_simd,
    {
      llvm::omp::allowedClauses_OMPD_teams_distribute_parallel_for_simd,
      llvm::omp::allowedOnceClauses_OMPD_teams_distribute_parallel_for_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_distribute_parallel_for_simd,
      llvm::omp::requiredClauses_OMPD_teams_distribute_parallel_for_simd,
    }
  },
  {llvm::omp::Directive::OMPD_teams_distribute_simd,
    {
      llvm::omp::allowedClauses_OMPD_teams_distribute_simd,
      llvm::omp::allowedOnceClauses_OMPD_teams_distribute_simd,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_distribute_simd,
      llvm::omp::requiredClauses_OMPD_teams_distribute_simd,
    }
  },
  {llvm::omp::Directive::OMPD_threadprivate,
    {
      llvm::omp::allowedClauses_OMPD_threadprivate,
      llvm::omp::allowedOnceClauses_OMPD_threadprivate,
      llvm::omp::allowedExclusiveClauses_OMPD_threadprivate,
      llvm::omp::requiredClauses_OMPD_threadprivate,
    }
  },
  {llvm::omp::Directive::OMPD_tile,
    {
      llvm::omp::allowedClauses_OMPD_tile,
      llvm::omp::allowedOnceClauses_OMPD_tile,
      llvm::omp::allowedExclusiveClauses_OMPD_tile,
      llvm::omp::requiredClauses_OMPD_tile,
    }
  },
  {llvm::omp::Directive::OMPD_unknown,
    {
      llvm::omp::allowedClauses_OMPD_unknown,
      llvm::omp::allowedOnceClauses_OMPD_unknown,
      llvm::omp::allowedExclusiveClauses_OMPD_unknown,
      llvm::omp::requiredClauses_OMPD_unknown,
    }
  },
  {llvm::omp::Directive::OMPD_unroll,
    {
      llvm::omp::allowedClauses_OMPD_unroll,
      llvm::omp::allowedOnceClauses_OMPD_unroll,
      llvm::omp::allowedExclusiveClauses_OMPD_unroll,
      llvm::omp::requiredClauses_OMPD_unroll,
    }
  },
  {llvm::omp::Directive::OMPD_workshare,
    {
      llvm::omp::allowedClauses_OMPD_workshare,
      llvm::omp::allowedOnceClauses_OMPD_workshare,
      llvm::omp::allowedExclusiveClauses_OMPD_workshare,
      llvm::omp::requiredClauses_OMPD_workshare,
    }
  },
  {llvm::omp::Directive::OMPD_dispatch,
    {
      llvm::omp::allowedClauses_OMPD_dispatch,
      llvm::omp::allowedOnceClauses_OMPD_dispatch,
      llvm::omp::allowedExclusiveClauses_OMPD_dispatch,
      llvm::omp::requiredClauses_OMPD_dispatch,
    }
  },
  {llvm::omp::Directive::OMPD_interop,
    {
      llvm::omp::allowedClauses_OMPD_interop,
      llvm::omp::allowedOnceClauses_OMPD_interop,
      llvm::omp::allowedExclusiveClauses_OMPD_interop,
      llvm::omp::requiredClauses_OMPD_interop,
    }
  },
  {llvm::omp::Directive::OMPD_loop,
    {
      llvm::omp::allowedClauses_OMPD_loop,
      llvm::omp::allowedOnceClauses_OMPD_loop,
      llvm::omp::allowedExclusiveClauses_OMPD_loop,
      llvm::omp::requiredClauses_OMPD_loop,
    }
  },
  {llvm::omp::Directive::OMPD_masked,
    {
      llvm::omp::allowedClauses_OMPD_masked,
      llvm::omp::allowedOnceClauses_OMPD_masked,
      llvm::omp::allowedExclusiveClauses_OMPD_masked,
      llvm::omp::requiredClauses_OMPD_masked,
    }
  },
  {llvm::omp::Directive::OMPD_parallel_loop,
    {
      llvm::omp::allowedClauses_OMPD_parallel_loop,
      llvm::omp::allowedOnceClauses_OMPD_parallel_loop,
      llvm::omp::allowedExclusiveClauses_OMPD_parallel_loop,
      llvm::omp::requiredClauses_OMPD_parallel_loop,
    }
  },
  {llvm::omp::Directive::OMPD_target_loop,
    {
      llvm::omp::allowedClauses_OMPD_target_loop,
      llvm::omp::allowedOnceClauses_OMPD_target_loop,
      llvm::omp::allowedExclusiveClauses_OMPD_target_loop,
      llvm::omp::requiredClauses_OMPD_target_loop,
    }
  },
  {llvm::omp::Directive::OMPD_target_parallel_loop,
    {
      llvm::omp::allowedClauses_OMPD_target_parallel_loop,
      llvm::omp::allowedOnceClauses_OMPD_target_parallel_loop,
      llvm::omp::allowedExclusiveClauses_OMPD_target_parallel_loop,
      llvm::omp::requiredClauses_OMPD_target_parallel_loop,
    }
  },
  {llvm::omp::Directive::OMPD_target_teams_loop,
    {
      llvm::omp::allowedClauses_OMPD_target_teams_loop,
      llvm::omp::allowedOnceClauses_OMPD_target_teams_loop,
      llvm::omp::allowedExclusiveClauses_OMPD_target_teams_loop,
      llvm::omp::requiredClauses_OMPD_target_teams_loop,
    }
  },
  {llvm::omp::Directive::OMPD_teams_loop,
    {
      llvm::omp::allowedClauses_OMPD_teams_loop,
      llvm::omp::allowedOnceClauses_OMPD_teams_loop,
      llvm::omp::allowedExclusiveClauses_OMPD_teams_loop,
      llvm::omp::requiredClauses_OMPD_teams_loop,
    }
  },
}

#endif // GEN_FLANG_DIRECTIVE_CLAUSE_MAP

#ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES
#undef GEN_FLANG_CLAUSE_PARSER_CLASSES

WRAPPER_CLASS(Absent, OmpAbsentClause);
EMPTY_CLASS(AcqRel);
EMPTY_CLASS(Acquire);
EMPTY_CLASS(AdjustArgs);
WRAPPER_CLASS(Affinity, OmpAffinityClause);
WRAPPER_CLASS(Align, OmpAlignClause);
WRAPPER_CLASS(Aligned, OmpAlignedClause);
WRAPPER_CLASS(Allocate, OmpAllocateClause);
WRAPPER_CLASS(Allocator, ScalarIntExpr);
EMPTY_CLASS(AppendArgs);
WRAPPER_CLASS(At, OmpAtClause);
WRAPPER_CLASS(AtomicDefaultMemOrder, OmpAtomicDefaultMemOrderClause);
WRAPPER_CLASS(Bind, OmpBindClause);
WRAPPER_CLASS(CancellationConstructType, OmpCancellationConstructTypeClause);
EMPTY_CLASS(Capture);
WRAPPER_CLASS(Collapse, ScalarIntConstantExpr);
EMPTY_CLASS(Compare);
WRAPPER_CLASS(Contains, OmpContainsClause);
WRAPPER_CLASS(Copyprivate, OmpObjectList);
WRAPPER_CLASS(Copyin, OmpObjectList);
WRAPPER_CLASS(Default, OmpDefaultClause);
WRAPPER_CLASS(Defaultmap, OmpDefaultmapClause);
WRAPPER_CLASS(Depend, OmpDependClause);
EMPTY_CLASS(Depobj);
WRAPPER_CLASS(Destroy, std::optional<OmpDestroyClause>);
WRAPPER_CLASS(Detach, OmpDetachClause);
WRAPPER_CLASS(Device, OmpDeviceClause);
WRAPPER_CLASS(DeviceType, OmpDeviceTypeClause);
WRAPPER_CLASS(DistSchedule, std::optional<ScalarIntExpr>);
WRAPPER_CLASS(Doacross, OmpDoacrossClause);
EMPTY_CLASS(DynamicAllocators);
WRAPPER_CLASS(Enter, OmpObjectList);
WRAPPER_CLASS(Exclusive, OmpObjectList);
WRAPPER_CLASS(Fail, OmpFailClause);
WRAPPER_CLASS(Filter, ScalarIntExpr);
WRAPPER_CLASS(Final, ScalarLogicalExpr);
WRAPPER_CLASS(Firstprivate, OmpObjectList);
EMPTY_CLASS(Flush);
WRAPPER_CLASS(From, OmpFromClause);
EMPTY_CLASS(Full);
WRAPPER_CLASS(Grainsize, OmpGrainsizeClause);
WRAPPER_CLASS(HasDeviceAddr, OmpObjectList);
WRAPPER_CLASS(Hint, OmpHintClause);
WRAPPER_CLASS(Holds, OmpHoldsClause);
WRAPPER_CLASS(If, OmpIfClause);
WRAPPER_CLASS(InReduction, OmpInReductionClause);
EMPTY_CLASS(Inbranch);
WRAPPER_CLASS(Inclusive, OmpObjectList);
EMPTY_CLASS(Indirect);
WRAPPER_CLASS(Init, OmpInitClause);
WRAPPER_CLASS(Initializer, OmpInitializerClause);
WRAPPER_CLASS(IsDevicePtr, OmpObjectList);
WRAPPER_CLASS(Lastprivate, OmpLastprivateClause);
WRAPPER_CLASS(Linear, OmpLinearClause);
WRAPPER_CLASS(Link, OmpObjectList);
WRAPPER_CLASS(Map, OmpMapClause);
WRAPPER_CLASS(Match, OmpMatchClause);
EMPTY_CLASS(MemoryOrder);
EMPTY_CLASS(Mergeable);
WRAPPER_CLASS(Message, OmpMessageClause);
EMPTY_CLASS(Nogroup);
EMPTY_CLASS(NoOpenmp);
EMPTY_CLASS(NoOpenmpConstructs);
EMPTY_CLASS(NoOpenmpRoutines);
EMPTY_CLASS(NoParallelism);
EMPTY_CLASS(Nowait);
WRAPPER_CLASS(Nocontext, ScalarLogicalExpr);
WRAPPER_CLASS(Nontemporal, std::list<Name>);
EMPTY_CLASS(Notinbranch);
WRAPPER_CLASS(Novariants, ScalarLogicalExpr);
WRAPPER_CLASS(NumTasks, OmpNumTasksClause);
WRAPPER_CLASS(NumTeams, ScalarIntExpr);
WRAPPER_CLASS(NumThreads, ScalarIntExpr);
EMPTY_CLASS(OmpxAttribute);
EMPTY_CLASS(OmpxBare);
WRAPPER_CLASS(OmpxDynCgroupMem, ScalarIntExpr);
WRAPPER_CLASS(Order, OmpOrderClause);
WRAPPER_CLASS(Ordered, std::optional<ScalarIntConstantExpr>);
WRAPPER_CLASS(Otherwise, std::optional<OmpOtherwiseClause>);
WRAPPER_CLASS(Partial, std::optional<ScalarIntConstantExpr>);
WRAPPER_CLASS(Permutation, std::list<ScalarIntExpr>);
WRAPPER_CLASS(Priority, ScalarIntExpr);
WRAPPER_CLASS(Private, OmpObjectList);
WRAPPER_CLASS(ProcBind, OmpProcBindClause);
EMPTY_CLASS(Read);
WRAPPER_CLASS(Reduction, OmpReductionClause);
EMPTY_CLASS(Relaxed);
EMPTY_CLASS(Release);
EMPTY_CLASS(ReverseOffload);
WRAPPER_CLASS(Safelen, ScalarIntConstantExpr);
WRAPPER_CLASS(Schedule, OmpScheduleClause);
EMPTY_CLASS(SelfMaps);
EMPTY_CLASS(SeqCst);
WRAPPER_CLASS(Severity, OmpSeverityClause);
WRAPPER_CLASS(Shared, OmpObjectList);
EMPTY_CLASS(Simd);
WRAPPER_CLASS(Simdlen, ScalarIntConstantExpr);
WRAPPER_CLASS(Sizes, std::list<ScalarIntExpr>);
WRAPPER_CLASS(TaskReduction, OmpTaskReductionClause);
WRAPPER_CLASS(ThreadLimit, ScalarIntExpr);
EMPTY_CLASS(Threadprivate);
EMPTY_CLASS(Threads);
WRAPPER_CLASS(To, OmpToClause);
EMPTY_CLASS(UnifiedAddress);
EMPTY_CLASS(UnifiedSharedMemory);
WRAPPER_CLASS(Uniform, std::list<Name>);
EMPTY_CLASS(Unknown);
EMPTY_CLASS(Untied);
WRAPPER_CLASS(Update, OmpUpdateClause);
WRAPPER_CLASS(Use, OmpUseClause);
WRAPPER_CLASS(UseDeviceAddr, OmpObjectList);
WRAPPER_CLASS(UseDevicePtr, OmpObjectList);
EMPTY_CLASS(UsesAllocators);
EMPTY_CLASS(Weak);
WRAPPER_CLASS(When, OmpWhenClause);
EMPTY_CLASS(Write);

#endif // GEN_FLANG_CLAUSE_PARSER_CLASSES

#ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
#undef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST

Absent
, AcqRel
, Acquire
, AdjustArgs
, Affinity
, Align
, Aligned
, Allocate
, Allocator
, AppendArgs
, At
, AtomicDefaultMemOrder
, Bind
, CancellationConstructType
, Capture
, Collapse
, Compare
, Contains
, Copyprivate
, Copyin
, Default
, Defaultmap
, Depend
, Depobj
, Destroy
, Detach
, Device
, DeviceType
, DistSchedule
, Doacross
, DynamicAllocators
, Enter
, Exclusive
, Fail
, Filter
, Final
, Firstprivate
, Flush
, From
, Full
, Grainsize
, HasDeviceAddr
, Hint
, Holds
, If
, InReduction
, Inbranch
, Inclusive
, Indirect
, Init
, Initializer
, IsDevicePtr
, Lastprivate
, Linear
, Link
, Map
, Match
, MemoryOrder
, Mergeable
, Message
, Nogroup
, NoOpenmp
, NoOpenmpConstructs
, NoOpenmpRoutines
, NoParallelism
, Nowait
, Nocontext
, Nontemporal
, Notinbranch
, Novariants
, NumTasks
, NumTeams
, NumThreads
, OmpxAttribute
, OmpxBare
, OmpxDynCgroupMem
, Order
, Ordered
, Otherwise
, Partial
, Permutation
, Priority
, Private
, ProcBind
, Read
, Reduction
, Relaxed
, Release
, ReverseOffload
, Safelen
, Schedule
, SelfMaps
, SeqCst
, Severity
, Shared
, Simd
, Simdlen
, Sizes
, TaskReduction
, ThreadLimit
, Threadprivate
, Threads
, To
, UnifiedAddress
, UnifiedSharedMemory
, Uniform
, Unknown
, Untied
, Update
, Use
, UseDeviceAddr
, UseDevicePtr
, UsesAllocators
, Weak
, When
, Write

#endif // GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST

#ifdef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
#undef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES

NODE(OmpClause, Absent)
NODE(OmpClause, AcqRel)
NODE(OmpClause, Acquire)
NODE(OmpClause, AdjustArgs)
NODE(OmpClause, Affinity)
NODE(OmpClause, Align)
NODE(OmpClause, Aligned)
NODE(OmpClause, Allocate)
NODE(OmpClause, Allocator)
NODE(OmpClause, AppendArgs)
NODE(OmpClause, At)
NODE(OmpClause, AtomicDefaultMemOrder)
NODE(OmpClause, Bind)
NODE(OmpClause, CancellationConstructType)
NODE(OmpClause, Capture)
NODE(OmpClause, Collapse)
NODE(OmpClause, Compare)
NODE(OmpClause, Contains)
NODE(OmpClause, Copyprivate)
NODE(OmpClause, Copyin)
NODE(OmpClause, Default)
NODE(OmpClause, Defaultmap)
NODE(OmpClause, Depend)
NODE(OmpClause, Depobj)
NODE(OmpClause, Destroy)
NODE(OmpClause, Detach)
NODE(OmpClause, Device)
NODE(OmpClause, DeviceType)
NODE(OmpClause, DistSchedule)
NODE(OmpClause, Doacross)
NODE(OmpClause, DynamicAllocators)
NODE(OmpClause, Enter)
NODE(OmpClause, Exclusive)
NODE(OmpClause, Fail)
NODE(OmpClause, Filter)
NODE(OmpClause, Final)
NODE(OmpClause, Firstprivate)
NODE(OmpClause, Flush)
NODE(OmpClause, From)
NODE(OmpClause, Full)
NODE(OmpClause, Grainsize)
NODE(OmpClause, HasDeviceAddr)
NODE(OmpClause, Hint)
NODE(OmpClause, Holds)
NODE(OmpClause, If)
NODE(OmpClause, InReduction)
NODE(OmpClause, Inbranch)
NODE(OmpClause, Inclusive)
NODE(OmpClause, Indirect)
NODE(OmpClause, Init)
NODE(OmpClause, Initializer)
NODE(OmpClause, IsDevicePtr)
NODE(OmpClause, Lastprivate)
NODE(OmpClause, Linear)
NODE(OmpClause, Link)
NODE(OmpClause, Map)
NODE(OmpClause, Match)
NODE(OmpClause, MemoryOrder)
NODE(OmpClause, Mergeable)
NODE(OmpClause, Message)
NODE(OmpClause, Nogroup)
NODE(OmpClause, NoOpenmp)
NODE(OmpClause, NoOpenmpConstructs)
NODE(OmpClause, NoOpenmpRoutines)
NODE(OmpClause, NoParallelism)
NODE(OmpClause, Nowait)
NODE(OmpClause, Nocontext)
NODE(OmpClause, Nontemporal)
NODE(OmpClause, Notinbranch)
NODE(OmpClause, Novariants)
NODE(OmpClause, NumTasks)
NODE(OmpClause, NumTeams)
NODE(OmpClause, NumThreads)
NODE(OmpClause, OmpxAttribute)
NODE(OmpClause, OmpxBare)
NODE(OmpClause, OmpxDynCgroupMem)
NODE(OmpClause, Order)
NODE(OmpClause, Ordered)
NODE(OmpClause, Otherwise)
NODE(OmpClause, Partial)
NODE(OmpClause, Permutation)
NODE(OmpClause, Priority)
NODE(OmpClause, Private)
NODE(OmpClause, ProcBind)
NODE(OmpClause, Read)
NODE(OmpClause, Reduction)
NODE(OmpClause, Relaxed)
NODE(OmpClause, Release)
NODE(OmpClause, ReverseOffload)
NODE(OmpClause, Safelen)
NODE(OmpClause, Schedule)
NODE(OmpClause, SelfMaps)
NODE(OmpClause, SeqCst)
NODE(OmpClause, Severity)
NODE(OmpClause, Shared)
NODE(OmpClause, Simd)
NODE(OmpClause, Simdlen)
NODE(OmpClause, Sizes)
NODE(OmpClause, TaskReduction)
NODE(OmpClause, ThreadLimit)
NODE(OmpClause, Threadprivate)
NODE(OmpClause, Threads)
NODE(OmpClause, To)
NODE(OmpClause, UnifiedAddress)
NODE(OmpClause, UnifiedSharedMemory)
NODE(OmpClause, Uniform)
NODE(OmpClause, Unknown)
NODE(OmpClause, Untied)
NODE(OmpClause, Update)
NODE(OmpClause, Use)
NODE(OmpClause, UseDeviceAddr)
NODE(OmpClause, UseDevicePtr)
NODE(OmpClause, UsesAllocators)
NODE(OmpClause, Weak)
NODE(OmpClause, When)
NODE(OmpClause, Write)

#endif // GEN_FLANG_DUMP_PARSE_TREE_CLAUSES

#ifdef GEN_FLANG_CLAUSE_UNPARSE
#undef GEN_FLANG_CLAUSE_UNPARSE

void Unparse(const OmpClause::Absent &x) {
  Word("ABSENT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::AcqRel &) { Word("ACQ_REL"); }
void Before(const OmpClause::Acquire &) { Word("ACQUIRE"); }
void Before(const OmpClause::AdjustArgs &) { Word("ADJUST_ARGS"); }
void Unparse(const OmpClause::Affinity &x) {
  Word("AFFINITY");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Align &x) {
  Word("ALIGN");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Aligned &x) {
  Word("ALIGNED");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Allocate &x) {
  Word("ALLOCATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Allocator &x) {
  Word("ALLOCATOR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::AppendArgs &) { Word("APPEND_ARGS"); }
void Unparse(const OmpClause::At &x) {
  Word("AT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::AtomicDefaultMemOrder &x) {
  Word("ATOMIC_DEFAULT_MEM_ORDER");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Bind &x) {
  Word("BIND");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Capture &) { Word("CAPTURE"); }
void Unparse(const OmpClause::Collapse &x) {
  Word("COLLAPSE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Compare &) { Word("COMPARE"); }
void Unparse(const OmpClause::Contains &x) {
  Word("CONTAINS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Copyprivate &x) {
  Word("COPYPRIVATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Copyin &x) {
  Word("COPYIN");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Default &x) {
  Word("DEFAULT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Defaultmap &x) {
  Word("DEFAULTMAP");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Depend &x) {
  Word("DEPEND");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Depobj &) { Word("DEPOBJ"); }
void Unparse(const OmpClause::Destroy &x) {
  Word("DESTROY");
  Walk("(", x.v, ")");
}
void Unparse(const OmpClause::Detach &x) {
  Word("DETACH");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Device &x) {
  Word("DEVICE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::DeviceType &x) {
  Word("DEVICE_TYPE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::DistSchedule &x) {
  Word("DIST_SCHEDULE");
  Walk("(", x.v, ")");
}
void Unparse(const OmpClause::Doacross &x) {
  Word("DOACROSS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::DynamicAllocators &) { Word("DYNAMIC_ALLOCATORS"); }
void Unparse(const OmpClause::Enter &x) {
  Word("ENTER");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Exclusive &x) {
  Word("EXCLUSIVE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Fail &x) {
  Word("FAIL");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Filter &x) {
  Word("FILTER");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Final &x) {
  Word("FINAL");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Firstprivate &x) {
  Word("FIRSTPRIVATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Flush &) { Word("FLUSH"); }
void Unparse(const OmpClause::From &x) {
  Word("FROM");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Full &) { Word("FULL"); }
void Unparse(const OmpClause::Grainsize &x) {
  Word("GRAINSIZE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::HasDeviceAddr &x) {
  Word("HAS_DEVICE_ADDR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Hint &x) {
  Word("HINT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Holds &x) {
  Word("HOLDS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::If &x) {
  Word("IF");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::InReduction &x) {
  Word("IN_REDUCTION");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Inbranch &) { Word("INBRANCH"); }
void Unparse(const OmpClause::Inclusive &x) {
  Word("INCLUSIVE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Indirect &) { Word("INDIRECT"); }
void Unparse(const OmpClause::Init &x) {
  Word("INIT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Initializer &x) {
  Word("INITIALIZER");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::IsDevicePtr &x) {
  Word("IS_DEVICE_PTR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Lastprivate &x) {
  Word("LASTPRIVATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Linear &x) {
  Word("LINEAR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Link &x) {
  Word("LINK");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Map &x) {
  Word("MAP");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Match &x) {
  Word("MATCH");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::MemoryOrder &) { Word("MEMORY_ORDER"); }
void Before(const OmpClause::Mergeable &) { Word("MERGEABLE"); }
void Unparse(const OmpClause::Message &x) {
  Word("MESSAGE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Nogroup &) { Word("NOGROUP"); }
void Before(const OmpClause::NoOpenmp &) { Word("NO_OPENMP"); }
void Before(const OmpClause::NoOpenmpConstructs &) { Word("NO_OPENMP_CONSTRUCTS"); }
void Before(const OmpClause::NoOpenmpRoutines &) { Word("NO_OPENMP_ROUTINES"); }
void Before(const OmpClause::NoParallelism &) { Word("NO_PARALLELISM"); }
void Before(const OmpClause::Nowait &) { Word("NOWAIT"); }
void Unparse(const OmpClause::Nocontext &x) {
  Word("NOCONTEXT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Nontemporal &x) {
  Word("NONTEMPORAL");
  Put("(");
  Walk(x.v, ",");
  Put(")");
}
void Before(const OmpClause::Notinbranch &) { Word("NOTINBRANCH"); }
void Unparse(const OmpClause::Novariants &x) {
  Word("NOVARIANTS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::NumTasks &x) {
  Word("NUM_TASKS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::NumTeams &x) {
  Word("NUM_TEAMS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::NumThreads &x) {
  Word("NUM_THREADS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::OmpxAttribute &) { Word("OMPX_ATTRIBUTE"); }
void Before(const OmpClause::OmpxBare &) { Word("OMPX_BARE"); }
void Unparse(const OmpClause::OmpxDynCgroupMem &x) {
  Word("OMPX_DYN_CGROUP_MEM");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Order &x) {
  Word("ORDER");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Ordered &x) {
  Word("ORDERED");
  Walk("(", x.v, ")");
}
void Unparse(const OmpClause::Otherwise &x) {
  Word("OTHERWISE");
  Walk("(", x.v, ")");
}
void Unparse(const OmpClause::Partial &x) {
  Word("PARTIAL");
  Walk("(", x.v, ")");
}
void Unparse(const OmpClause::Permutation &x) {
  Word("PERMUTATION");
  Put("(");
  Walk(x.v, ",");
  Put(")");
}
void Unparse(const OmpClause::Priority &x) {
  Word("PRIORITY");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Private &x) {
  Word("PRIVATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::ProcBind &x) {
  Word("PROC_BIND");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Read &) { Word("READ"); }
void Unparse(const OmpClause::Reduction &x) {
  Word("REDUCTION");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Relaxed &) { Word("RELAXED"); }
void Before(const OmpClause::Release &) { Word("RELEASE"); }
void Before(const OmpClause::ReverseOffload &) { Word("REVERSE_OFFLOAD"); }
void Unparse(const OmpClause::Safelen &x) {
  Word("SAFELEN");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Schedule &x) {
  Word("SCHEDULE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::SelfMaps &) { Word("SELF_MAPS"); }
void Before(const OmpClause::SeqCst &) { Word("SEQ_CST"); }
void Unparse(const OmpClause::Severity &x) {
  Word("SEVERITY");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Shared &x) {
  Word("SHARED");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Simd &) { Word("SIMD"); }
void Unparse(const OmpClause::Simdlen &x) {
  Word("SIMDLEN");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Sizes &x) {
  Word("SIZES");
  Put("(");
  Walk(x.v, ",");
  Put(")");
}
void Unparse(const OmpClause::TaskReduction &x) {
  Word("TASK_REDUCTION");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::ThreadLimit &x) {
  Word("THREAD_LIMIT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Threadprivate &) { Word("THREADPRIVATE"); }
void Before(const OmpClause::Threads &) { Word("THREADS"); }
void Unparse(const OmpClause::To &x) {
  Word("TO");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::UnifiedAddress &) { Word("UNIFIED_ADDRESS"); }
void Before(const OmpClause::UnifiedSharedMemory &) { Word("UNIFIED_SHARED_MEMORY"); }
void Unparse(const OmpClause::Uniform &x) {
  Word("UNIFORM");
  Put("(");
  Walk(x.v, ",");
  Put(")");
}
void Before(const OmpClause::Unknown &) { Word("UNKNOWN"); }
void Before(const OmpClause::Untied &) { Word("UNTIED"); }
void Unparse(const OmpClause::Update &x) {
  Word("UPDATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::Use &x) {
  Word("USE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::UseDeviceAddr &x) {
  Word("USE_DEVICE_ADDR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const OmpClause::UseDevicePtr &x) {
  Word("USE_DEVICE_PTR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::UsesAllocators &) { Word("USES_ALLOCATORS"); }
void Before(const OmpClause::Weak &) { Word("WEAK"); }
void Unparse(const OmpClause::When &x) {
  Word("WHEN");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const OmpClause::Write &) { Word("WRITE"); }

#endif // GEN_FLANG_CLAUSE_UNPARSE

#ifdef GEN_FLANG_CLAUSE_CHECK_ENTER
#undef GEN_FLANG_CLAUSE_CHECK_ENTER

void Enter(const parser::OmpClause::Absent &);
void Enter(const parser::OmpClause::AcqRel &);
void Enter(const parser::OmpClause::Acquire &);
void Enter(const parser::OmpClause::AdjustArgs &);
void Enter(const parser::OmpClause::Affinity &);
void Enter(const parser::OmpClause::Align &);
void Enter(const parser::OmpClause::Aligned &);
void Enter(const parser::OmpClause::Allocate &);
void Enter(const parser::OmpClause::Allocator &);
void Enter(const parser::OmpClause::AppendArgs &);
void Enter(const parser::OmpClause::At &);
void Enter(const parser::OmpClause::AtomicDefaultMemOrder &);
void Enter(const parser::OmpClause::Bind &);
void Enter(const parser::OmpClause::CancellationConstructType &);
void Enter(const parser::OmpClause::Capture &);
void Enter(const parser::OmpClause::Collapse &);
void Enter(const parser::OmpClause::Compare &);
void Enter(const parser::OmpClause::Contains &);
void Enter(const parser::OmpClause::Copyprivate &);
void Enter(const parser::OmpClause::Copyin &);
void Enter(const parser::OmpClause::Default &);
void Enter(const parser::OmpClause::Defaultmap &);
void Enter(const parser::OmpClause::Depend &);
void Enter(const parser::OmpClause::Depobj &);
void Enter(const parser::OmpClause::Destroy &);
void Enter(const parser::OmpClause::Detach &);
void Enter(const parser::OmpClause::Device &);
void Enter(const parser::OmpClause::DeviceType &);
void Enter(const parser::OmpClause::DistSchedule &);
void Enter(const parser::OmpClause::Doacross &);
void Enter(const parser::OmpClause::DynamicAllocators &);
void Enter(const parser::OmpClause::Enter &);
void Enter(const parser::OmpClause::Exclusive &);
void Enter(const parser::OmpClause::Fail &);
void Enter(const parser::OmpClause::Filter &);
void Enter(const parser::OmpClause::Final &);
void Enter(const parser::OmpClause::Firstprivate &);
void Enter(const parser::OmpClause::Flush &);
void Enter(const parser::OmpClause::From &);
void Enter(const parser::OmpClause::Full &);
void Enter(const parser::OmpClause::Grainsize &);
void Enter(const parser::OmpClause::HasDeviceAddr &);
void Enter(const parser::OmpClause::Hint &);
void Enter(const parser::OmpClause::Holds &);
void Enter(const parser::OmpClause::If &);
void Enter(const parser::OmpClause::InReduction &);
void Enter(const parser::OmpClause::Inbranch &);
void Enter(const parser::OmpClause::Inclusive &);
void Enter(const parser::OmpClause::Indirect &);
void Enter(const parser::OmpClause::Init &);
void Enter(const parser::OmpClause::Initializer &);
void Enter(const parser::OmpClause::IsDevicePtr &);
void Enter(const parser::OmpClause::Lastprivate &);
void Enter(const parser::OmpClause::Linear &);
void Enter(const parser::OmpClause::Link &);
void Enter(const parser::OmpClause::Map &);
void Enter(const parser::OmpClause::Match &);
void Enter(const parser::OmpClause::MemoryOrder &);
void Enter(const parser::OmpClause::Mergeable &);
void Enter(const parser::OmpClause::Message &);
void Enter(const parser::OmpClause::Nogroup &);
void Enter(const parser::OmpClause::NoOpenmp &);
void Enter(const parser::OmpClause::NoOpenmpConstructs &);
void Enter(const parser::OmpClause::NoOpenmpRoutines &);
void Enter(const parser::OmpClause::NoParallelism &);
void Enter(const parser::OmpClause::Nowait &);
void Enter(const parser::OmpClause::Nocontext &);
void Enter(const parser::OmpClause::Nontemporal &);
void Enter(const parser::OmpClause::Notinbranch &);
void Enter(const parser::OmpClause::Novariants &);
void Enter(const parser::OmpClause::NumTasks &);
void Enter(const parser::OmpClause::NumTeams &);
void Enter(const parser::OmpClause::NumThreads &);
void Enter(const parser::OmpClause::OmpxAttribute &);
void Enter(const parser::OmpClause::OmpxBare &);
void Enter(const parser::OmpClause::OmpxDynCgroupMem &);
void Enter(const parser::OmpClause::Order &);
void Enter(const parser::OmpClause::Ordered &);
void Enter(const parser::OmpClause::Otherwise &);
void Enter(const parser::OmpClause::Partial &);
void Enter(const parser::OmpClause::Permutation &);
void Enter(const parser::OmpClause::Priority &);
void Enter(const parser::OmpClause::Private &);
void Enter(const parser::OmpClause::ProcBind &);
void Enter(const parser::OmpClause::Read &);
void Enter(const parser::OmpClause::Reduction &);
void Enter(const parser::OmpClause::Relaxed &);
void Enter(const parser::OmpClause::Release &);
void Enter(const parser::OmpClause::ReverseOffload &);
void Enter(const parser::OmpClause::Safelen &);
void Enter(const parser::OmpClause::Schedule &);
void Enter(const parser::OmpClause::SelfMaps &);
void Enter(const parser::OmpClause::SeqCst &);
void Enter(const parser::OmpClause::Severity &);
void Enter(const parser::OmpClause::Shared &);
void Enter(const parser::OmpClause::Simd &);
void Enter(const parser::OmpClause::Simdlen &);
void Enter(const parser::OmpClause::Sizes &);
void Enter(const parser::OmpClause::TaskReduction &);
void Enter(const parser::OmpClause::ThreadLimit &);
void Enter(const parser::OmpClause::Threadprivate &);
void Enter(const parser::OmpClause::Threads &);
void Enter(const parser::OmpClause::To &);
void Enter(const parser::OmpClause::UnifiedAddress &);
void Enter(const parser::OmpClause::UnifiedSharedMemory &);
void Enter(const parser::OmpClause::Uniform &);
void Enter(const parser::OmpClause::Unknown &);
void Enter(const parser::OmpClause::Untied &);
void Enter(const parser::OmpClause::Update &);
void Enter(const parser::OmpClause::Use &);
void Enter(const parser::OmpClause::UseDeviceAddr &);
void Enter(const parser::OmpClause::UseDevicePtr &);
void Enter(const parser::OmpClause::UsesAllocators &);
void Enter(const parser::OmpClause::Weak &);
void Enter(const parser::OmpClause::When &);
void Enter(const parser::OmpClause::Write &);

#endif // GEN_FLANG_CLAUSE_CHECK_ENTER

#ifdef GEN_FLANG_CLAUSE_PARSER_KIND_MAP
#undef GEN_FLANG_CLAUSE_PARSER_KIND_MAP

if constexpr (std::is_same_v<A, parser::OmpClause::Absent>)
  return llvm::omp::Clause::OMPC_absent;
if constexpr (std::is_same_v<A, parser::OmpClause::AcqRel>)
  return llvm::omp::Clause::OMPC_acq_rel;
if constexpr (std::is_same_v<A, parser::OmpClause::Acquire>)
  return llvm::omp::Clause::OMPC_acquire;
if constexpr (std::is_same_v<A, parser::OmpClause::AdjustArgs>)
  return llvm::omp::Clause::OMPC_adjust_args;
if constexpr (std::is_same_v<A, parser::OmpClause::Affinity>)
  return llvm::omp::Clause::OMPC_affinity;
if constexpr (std::is_same_v<A, parser::OmpClause::Align>)
  return llvm::omp::Clause::OMPC_align;
if constexpr (std::is_same_v<A, parser::OmpClause::Aligned>)
  return llvm::omp::Clause::OMPC_aligned;
if constexpr (std::is_same_v<A, parser::OmpClause::Allocate>)
  return llvm::omp::Clause::OMPC_allocate;
if constexpr (std::is_same_v<A, parser::OmpClause::Allocator>)
  return llvm::omp::Clause::OMPC_allocator;
if constexpr (std::is_same_v<A, parser::OmpClause::AppendArgs>)
  return llvm::omp::Clause::OMPC_append_args;
if constexpr (std::is_same_v<A, parser::OmpClause::At>)
  return llvm::omp::Clause::OMPC_at;
if constexpr (std::is_same_v<A, parser::OmpClause::AtomicDefaultMemOrder>)
  return llvm::omp::Clause::OMPC_atomic_default_mem_order;
if constexpr (std::is_same_v<A, parser::OmpClause::Bind>)
  return llvm::omp::Clause::OMPC_bind;
if constexpr (std::is_same_v<A, parser::OmpClause::CancellationConstructType>)
  return llvm::omp::Clause::OMPC_cancellation_construct_type;
if constexpr (std::is_same_v<A, parser::OmpClause::Capture>)
  return llvm::omp::Clause::OMPC_capture;
if constexpr (std::is_same_v<A, parser::OmpClause::Collapse>)
  return llvm::omp::Clause::OMPC_collapse;
if constexpr (std::is_same_v<A, parser::OmpClause::Compare>)
  return llvm::omp::Clause::OMPC_compare;
if constexpr (std::is_same_v<A, parser::OmpClause::Contains>)
  return llvm::omp::Clause::OMPC_contains;
if constexpr (std::is_same_v<A, parser::OmpClause::Copyprivate>)
  return llvm::omp::Clause::OMPC_copyprivate;
if constexpr (std::is_same_v<A, parser::OmpClause::Copyin>)
  return llvm::omp::Clause::OMPC_copyin;
if constexpr (std::is_same_v<A, parser::OmpClause::Default>)
  return llvm::omp::Clause::OMPC_default;
if constexpr (std::is_same_v<A, parser::OmpClause::Defaultmap>)
  return llvm::omp::Clause::OMPC_defaultmap;
if constexpr (std::is_same_v<A, parser::OmpClause::Depend>)
  return llvm::omp::Clause::OMPC_depend;
if constexpr (std::is_same_v<A, parser::OmpClause::Depobj>)
  return llvm::omp::Clause::OMPC_depobj;
if constexpr (std::is_same_v<A, parser::OmpClause::Destroy>)
  return llvm::omp::Clause::OMPC_destroy;
if constexpr (std::is_same_v<A, parser::OmpClause::Detach>)
  return llvm::omp::Clause::OMPC_detach;
if constexpr (std::is_same_v<A, parser::OmpClause::Device>)
  return llvm::omp::Clause::OMPC_device;
if constexpr (std::is_same_v<A, parser::OmpClause::DeviceType>)
  return llvm::omp::Clause::OMPC_device_type;
if constexpr (std::is_same_v<A, parser::OmpClause::DistSchedule>)
  return llvm::omp::Clause::OMPC_dist_schedule;
if constexpr (std::is_same_v<A, parser::OmpClause::Doacross>)
  return llvm::omp::Clause::OMPC_doacross;
if constexpr (std::is_same_v<A, parser::OmpClause::DynamicAllocators>)
  return llvm::omp::Clause::OMPC_dynamic_allocators;
if constexpr (std::is_same_v<A, parser::OmpClause::Enter>)
  return llvm::omp::Clause::OMPC_enter;
if constexpr (std::is_same_v<A, parser::OmpClause::Exclusive>)
  return llvm::omp::Clause::OMPC_exclusive;
if constexpr (std::is_same_v<A, parser::OmpClause::Fail>)
  return llvm::omp::Clause::OMPC_fail;
if constexpr (std::is_same_v<A, parser::OmpClause::Filter>)
  return llvm::omp::Clause::OMPC_filter;
if constexpr (std::is_same_v<A, parser::OmpClause::Final>)
  return llvm::omp::Clause::OMPC_final;
if constexpr (std::is_same_v<A, parser::OmpClause::Firstprivate>)
  return llvm::omp::Clause::OMPC_firstprivate;
if constexpr (std::is_same_v<A, parser::OmpClause::Flush>)
  return llvm::omp::Clause::OMPC_flush;
if constexpr (std::is_same_v<A, parser::OmpClause::From>)
  return llvm::omp::Clause::OMPC_from;
if constexpr (std::is_same_v<A, parser::OmpClause::Full>)
  return llvm::omp::Clause::OMPC_full;
if constexpr (std::is_same_v<A, parser::OmpClause::Grainsize>)
  return llvm::omp::Clause::OMPC_grainsize;
if constexpr (std::is_same_v<A, parser::OmpClause::HasDeviceAddr>)
  return llvm::omp::Clause::OMPC_has_device_addr;
if constexpr (std::is_same_v<A, parser::OmpClause::Hint>)
  return llvm::omp::Clause::OMPC_hint;
if constexpr (std::is_same_v<A, parser::OmpClause::Holds>)
  return llvm::omp::Clause::OMPC_holds;
if constexpr (std::is_same_v<A, parser::OmpClause::If>)
  return llvm::omp::Clause::OMPC_if;
if constexpr (std::is_same_v<A, parser::OmpClause::InReduction>)
  return llvm::omp::Clause::OMPC_in_reduction;
if constexpr (std::is_same_v<A, parser::OmpClause::Inbranch>)
  return llvm::omp::Clause::OMPC_inbranch;
if constexpr (std::is_same_v<A, parser::OmpClause::Inclusive>)
  return llvm::omp::Clause::OMPC_inclusive;
if constexpr (std::is_same_v<A, parser::OmpClause::Indirect>)
  return llvm::omp::Clause::OMPC_indirect;
if constexpr (std::is_same_v<A, parser::OmpClause::Init>)
  return llvm::omp::Clause::OMPC_init;
if constexpr (std::is_same_v<A, parser::OmpClause::Initializer>)
  return llvm::omp::Clause::OMPC_initializer;
if constexpr (std::is_same_v<A, parser::OmpClause::IsDevicePtr>)
  return llvm::omp::Clause::OMPC_is_device_ptr;
if constexpr (std::is_same_v<A, parser::OmpClause::Lastprivate>)
  return llvm::omp::Clause::OMPC_lastprivate;
if constexpr (std::is_same_v<A, parser::OmpClause::Linear>)
  return llvm::omp::Clause::OMPC_linear;
if constexpr (std::is_same_v<A, parser::OmpClause::Link>)
  return llvm::omp::Clause::OMPC_link;
if constexpr (std::is_same_v<A, parser::OmpClause::Map>)
  return llvm::omp::Clause::OMPC_map;
if constexpr (std::is_same_v<A, parser::OmpClause::Match>)
  return llvm::omp::Clause::OMPC_match;
if constexpr (std::is_same_v<A, parser::OmpClause::MemoryOrder>)
  return llvm::omp::Clause::OMPC_memory_order;
if constexpr (std::is_same_v<A, parser::OmpClause::Mergeable>)
  return llvm::omp::Clause::OMPC_mergeable;
if constexpr (std::is_same_v<A, parser::OmpClause::Message>)
  return llvm::omp::Clause::OMPC_message;
if constexpr (std::is_same_v<A, parser::OmpClause::Nogroup>)
  return llvm::omp::Clause::OMPC_nogroup;
if constexpr (std::is_same_v<A, parser::OmpClause::NoOpenmp>)
  return llvm::omp::Clause::OMPC_no_openmp;
if constexpr (std::is_same_v<A, parser::OmpClause::NoOpenmpConstructs>)
  return llvm::omp::Clause::OMPC_no_openmp_constructs;
if constexpr (std::is_same_v<A, parser::OmpClause::NoOpenmpRoutines>)
  return llvm::omp::Clause::OMPC_no_openmp_routines;
if constexpr (std::is_same_v<A, parser::OmpClause::NoParallelism>)
  return llvm::omp::Clause::OMPC_no_parallelism;
if constexpr (std::is_same_v<A, parser::OmpClause::Nowait>)
  return llvm::omp::Clause::OMPC_nowait;
if constexpr (std::is_same_v<A, parser::OmpClause::Nocontext>)
  return llvm::omp::Clause::OMPC_nocontext;
if constexpr (std::is_same_v<A, parser::OmpClause::Nontemporal>)
  return llvm::omp::Clause::OMPC_nontemporal;
if constexpr (std::is_same_v<A, parser::OmpClause::Notinbranch>)
  return llvm::omp::Clause::OMPC_notinbranch;
if constexpr (std::is_same_v<A, parser::OmpClause::Novariants>)
  return llvm::omp::Clause::OMPC_novariants;
if constexpr (std::is_same_v<A, parser::OmpClause::NumTasks>)
  return llvm::omp::Clause::OMPC_num_tasks;
if constexpr (std::is_same_v<A, parser::OmpClause::NumTeams>)
  return llvm::omp::Clause::OMPC_num_teams;
if constexpr (std::is_same_v<A, parser::OmpClause::NumThreads>)
  return llvm::omp::Clause::OMPC_num_threads;
if constexpr (std::is_same_v<A, parser::OmpClause::OmpxAttribute>)
  return llvm::omp::Clause::OMPC_ompx_attribute;
if constexpr (std::is_same_v<A, parser::OmpClause::OmpxBare>)
  return llvm::omp::Clause::OMPC_ompx_bare;
if constexpr (std::is_same_v<A, parser::OmpClause::OmpxDynCgroupMem>)
  return llvm::omp::Clause::OMPC_ompx_dyn_cgroup_mem;
if constexpr (std::is_same_v<A, parser::OmpClause::Order>)
  return llvm::omp::Clause::OMPC_order;
if constexpr (std::is_same_v<A, parser::OmpClause::Ordered>)
  return llvm::omp::Clause::OMPC_ordered;
if constexpr (std::is_same_v<A, parser::OmpClause::Otherwise>)
  return llvm::omp::Clause::OMPC_otherwise;
if constexpr (std::is_same_v<A, parser::OmpClause::Partial>)
  return llvm::omp::Clause::OMPC_partial;
if constexpr (std::is_same_v<A, parser::OmpClause::Permutation>)
  return llvm::omp::Clause::OMPC_permutation;
if constexpr (std::is_same_v<A, parser::OmpClause::Priority>)
  return llvm::omp::Clause::OMPC_priority;
if constexpr (std::is_same_v<A, parser::OmpClause::Private>)
  return llvm::omp::Clause::OMPC_private;
if constexpr (std::is_same_v<A, parser::OmpClause::ProcBind>)
  return llvm::omp::Clause::OMPC_proc_bind;
if constexpr (std::is_same_v<A, parser::OmpClause::Read>)
  return llvm::omp::Clause::OMPC_read;
if constexpr (std::is_same_v<A, parser::OmpClause::Reduction>)
  return llvm::omp::Clause::OMPC_reduction;
if constexpr (std::is_same_v<A, parser::OmpClause::Relaxed>)
  return llvm::omp::Clause::OMPC_relaxed;
if constexpr (std::is_same_v<A, parser::OmpClause::Release>)
  return llvm::omp::Clause::OMPC_release;
if constexpr (std::is_same_v<A, parser::OmpClause::ReverseOffload>)
  return llvm::omp::Clause::OMPC_reverse_offload;
if constexpr (std::is_same_v<A, parser::OmpClause::Safelen>)
  return llvm::omp::Clause::OMPC_safelen;
if constexpr (std::is_same_v<A, parser::OmpClause::Schedule>)
  return llvm::omp::Clause::OMPC_schedule;
if constexpr (std::is_same_v<A, parser::OmpClause::SelfMaps>)
  return llvm::omp::Clause::OMPC_self_maps;
if constexpr (std::is_same_v<A, parser::OmpClause::SeqCst>)
  return llvm::omp::Clause::OMPC_seq_cst;
if constexpr (std::is_same_v<A, parser::OmpClause::Severity>)
  return llvm::omp::Clause::OMPC_severity;
if constexpr (std::is_same_v<A, parser::OmpClause::Shared>)
  return llvm::omp::Clause::OMPC_shared;
if constexpr (std::is_same_v<A, parser::OmpClause::Simd>)
  return llvm::omp::Clause::OMPC_simd;
if constexpr (std::is_same_v<A, parser::OmpClause::Simdlen>)
  return llvm::omp::Clause::OMPC_simdlen;
if constexpr (std::is_same_v<A, parser::OmpClause::Sizes>)
  return llvm::omp::Clause::OMPC_sizes;
if constexpr (std::is_same_v<A, parser::OmpClause::TaskReduction>)
  return llvm::omp::Clause::OMPC_task_reduction;
if constexpr (std::is_same_v<A, parser::OmpClause::ThreadLimit>)
  return llvm::omp::Clause::OMPC_thread_limit;
if constexpr (std::is_same_v<A, parser::OmpClause::Threadprivate>)
  return llvm::omp::Clause::OMPC_threadprivate;
if constexpr (std::is_same_v<A, parser::OmpClause::Threads>)
  return llvm::omp::Clause::OMPC_threads;
if constexpr (std::is_same_v<A, parser::OmpClause::To>)
  return llvm::omp::Clause::OMPC_to;
if constexpr (std::is_same_v<A, parser::OmpClause::UnifiedAddress>)
  return llvm::omp::Clause::OMPC_unified_address;
if constexpr (std::is_same_v<A, parser::OmpClause::UnifiedSharedMemory>)
  return llvm::omp::Clause::OMPC_unified_shared_memory;
if constexpr (std::is_same_v<A, parser::OmpClause::Uniform>)
  return llvm::omp::Clause::OMPC_uniform;
if constexpr (std::is_same_v<A, parser::OmpClause::Unknown>)
  return llvm::omp::Clause::OMPC_unknown;
if constexpr (std::is_same_v<A, parser::OmpClause::Untied>)
  return llvm::omp::Clause::OMPC_untied;
if constexpr (std::is_same_v<A, parser::OmpClause::Update>)
  return llvm::omp::Clause::OMPC_update;
if constexpr (std::is_same_v<A, parser::OmpClause::Use>)
  return llvm::omp::Clause::OMPC_use;
if constexpr (std::is_same_v<A, parser::OmpClause::UseDeviceAddr>)
  return llvm::omp::Clause::OMPC_use_device_addr;
if constexpr (std::is_same_v<A, parser::OmpClause::UseDevicePtr>)
  return llvm::omp::Clause::OMPC_use_device_ptr;
if constexpr (std::is_same_v<A, parser::OmpClause::UsesAllocators>)
  return llvm::omp::Clause::OMPC_uses_allocators;
if constexpr (std::is_same_v<A, parser::OmpClause::Weak>)
  return llvm::omp::Clause::OMPC_weak;
if constexpr (std::is_same_v<A, parser::OmpClause::When>)
  return llvm::omp::Clause::OMPC_when;
if constexpr (std::is_same_v<A, parser::OmpClause::Write>)
  return llvm::omp::Clause::OMPC_write;
llvm_unreachable("Invalid OpenMP Parser clause");

#endif // GEN_FLANG_CLAUSE_PARSER_KIND_MAP

#ifdef GEN_FLANG_CLAUSES_PARSER
#undef GEN_FLANG_CLAUSES_PARSER

TYPE_PARSER(
  "write" >> construct<OmpClause>(construct<OmpClause::Write>()) ||
  "when" >> construct<OmpClause>(construct<OmpClause::When>(parenthesized(Parser<OmpWhenClause>{}))) ||
  "weak" >> construct<OmpClause>(construct<OmpClause::Weak>()) ||
  "uses_allocators" >> construct<OmpClause>(construct<OmpClause::UsesAllocators>()) ||
  "use_device_ptr" >> construct<OmpClause>(construct<OmpClause::UseDevicePtr>(parenthesized(Parser<OmpObjectList>{}))) ||
  "use_device_addr" >> construct<OmpClause>(construct<OmpClause::UseDeviceAddr>(parenthesized(Parser<OmpObjectList>{}))) ||
  "use" >> construct<OmpClause>(construct<OmpClause::Use>(parenthesized(Parser<OmpUseClause>{}))) ||
  "update" >> construct<OmpClause>(construct<OmpClause::Update>(parenthesized(Parser<OmpUpdateClause>{}))) ||
  "untied" >> construct<OmpClause>(construct<OmpClause::Untied>()) ||
  "unknown" >> construct<OmpClause>(construct<OmpClause::Unknown>()) ||
  "uniform" >> construct<OmpClause>(construct<OmpClause::Uniform>(parenthesized(nonemptyList(name)))) ||
  "unified_shared_memory" >> construct<OmpClause>(construct<OmpClause::UnifiedSharedMemory>()) ||
  "unified_address" >> construct<OmpClause>(construct<OmpClause::UnifiedAddress>()) ||
  "to" >> construct<OmpClause>(construct<OmpClause::To>(parenthesized(Parser<OmpToClause>{}))) ||
  "threads" >> construct<OmpClause>(construct<OmpClause::Threads>()) ||
  "threadprivate" >> construct<OmpClause>(construct<OmpClause::Threadprivate>()) ||
  "thread_limit" >> construct<OmpClause>(construct<OmpClause::ThreadLimit>(parenthesized(scalarIntExpr))) ||
  "task_reduction" >> construct<OmpClause>(construct<OmpClause::TaskReduction>(parenthesized(Parser<OmpTaskReductionClause>{}))) ||
  "sizes" >> construct<OmpClause>(construct<OmpClause::Sizes>(parenthesized(nonemptyList(scalarIntExpr)))) ||
  "simdlen" >> construct<OmpClause>(construct<OmpClause::Simdlen>(parenthesized(scalarIntConstantExpr))) ||
  "simd" >> construct<OmpClause>(construct<OmpClause::Simd>()) ||
  "shared" >> construct<OmpClause>(construct<OmpClause::Shared>(parenthesized(Parser<OmpObjectList>{}))) ||
  "severity" >> construct<OmpClause>(construct<OmpClause::Severity>(parenthesized(Parser<OmpSeverityClause>{}))) ||
  "seq_cst" >> construct<OmpClause>(construct<OmpClause::SeqCst>()) ||
  "self_maps" >> construct<OmpClause>(construct<OmpClause::SelfMaps>()) ||
  "schedule" >> construct<OmpClause>(construct<OmpClause::Schedule>(parenthesized(Parser<OmpScheduleClause>{}))) ||
  "safelen" >> construct<OmpClause>(construct<OmpClause::Safelen>(parenthesized(scalarIntConstantExpr))) ||
  "reverse_offload" >> construct<OmpClause>(construct<OmpClause::ReverseOffload>()) ||
  "release" >> construct<OmpClause>(construct<OmpClause::Release>()) ||
  "relaxed" >> construct<OmpClause>(construct<OmpClause::Relaxed>()) ||
  "reduction" >> construct<OmpClause>(construct<OmpClause::Reduction>(parenthesized(Parser<OmpReductionClause>{}))) ||
  "read" >> construct<OmpClause>(construct<OmpClause::Read>()) ||
  "proc_bind" >> construct<OmpClause>(construct<OmpClause::ProcBind>(parenthesized(Parser<OmpProcBindClause>{}))) ||
  "private" >> construct<OmpClause>(construct<OmpClause::Private>(parenthesized(Parser<OmpObjectList>{}))) ||
  "priority" >> construct<OmpClause>(construct<OmpClause::Priority>(parenthesized(scalarIntExpr))) ||
  "permutation" >> construct<OmpClause>(construct<OmpClause::Permutation>(parenthesized(nonemptyList(scalarIntExpr)))) ||
  "partial" >> construct<OmpClause>(construct<OmpClause::Partial>(maybe(parenthesized(scalarIntConstantExpr)))) ||
  "otherwise" >> construct<OmpClause>(construct<OmpClause::Otherwise>(maybe(parenthesized(Parser<OmpOtherwiseClause>{})))) ||
  "ordered" >> construct<OmpClause>(construct<OmpClause::Ordered>(maybe(parenthesized(scalarIntConstantExpr)))) ||
  "order" >> construct<OmpClause>(construct<OmpClause::Order>(parenthesized(Parser<OmpOrderClause>{}))) ||
  "ompx_dyn_cgroup_mem" >> construct<OmpClause>(construct<OmpClause::OmpxDynCgroupMem>(parenthesized(scalarIntExpr))) ||
  "ompx_bare" >> construct<OmpClause>(construct<OmpClause::OmpxBare>()) ||
  "ompx_attribute" >> construct<OmpClause>(construct<OmpClause::OmpxAttribute>()) ||
  "num_threads" >> construct<OmpClause>(construct<OmpClause::NumThreads>(parenthesized(scalarIntExpr))) ||
  "num_teams" >> construct<OmpClause>(construct<OmpClause::NumTeams>(parenthesized(scalarIntExpr))) ||
  "num_tasks" >> construct<OmpClause>(construct<OmpClause::NumTasks>(parenthesized(Parser<OmpNumTasksClause>{}))) ||
  "nowait" >> construct<OmpClause>(construct<OmpClause::Nowait>()) ||
  "novariants" >> construct<OmpClause>(construct<OmpClause::Novariants>(parenthesized(scalarLogicalExpr))) ||
  "notinbranch" >> construct<OmpClause>(construct<OmpClause::Notinbranch>()) ||
  "nontemporal" >> construct<OmpClause>(construct<OmpClause::Nontemporal>(parenthesized(nonemptyList(name)))) ||
  "nogroup" >> construct<OmpClause>(construct<OmpClause::Nogroup>()) ||
  "nocontext" >> construct<OmpClause>(construct<OmpClause::Nocontext>(parenthesized(scalarLogicalExpr))) ||
  "no_parallelism" >> construct<OmpClause>(construct<OmpClause::NoParallelism>()) ||
  "no_openmp_routines" >> construct<OmpClause>(construct<OmpClause::NoOpenmpRoutines>()) ||
  "no_openmp_constructs" >> construct<OmpClause>(construct<OmpClause::NoOpenmpConstructs>()) ||
  "no_openmp" >> construct<OmpClause>(construct<OmpClause::NoOpenmp>()) ||
  "message" >> construct<OmpClause>(construct<OmpClause::Message>(parenthesized(Parser<OmpMessageClause>{}))) ||
  "mergeable" >> construct<OmpClause>(construct<OmpClause::Mergeable>()) ||
  "memory_order" >> construct<OmpClause>(construct<OmpClause::MemoryOrder>()) ||
  "match" >> construct<OmpClause>(construct<OmpClause::Match>(parenthesized(Parser<OmpMatchClause>{}))) ||
  "map" >> construct<OmpClause>(construct<OmpClause::Map>(parenthesized(Parser<OmpMapClause>{}))) ||
  "link" >> construct<OmpClause>(construct<OmpClause::Link>(parenthesized(Parser<OmpObjectList>{}))) ||
  "linear" >> construct<OmpClause>(construct<OmpClause::Linear>(parenthesized(Parser<OmpLinearClause>{}))) ||
  "lastprivate" >> construct<OmpClause>(construct<OmpClause::Lastprivate>(parenthesized(Parser<OmpLastprivateClause>{}))) ||
  "is_device_ptr" >> construct<OmpClause>(construct<OmpClause::IsDevicePtr>(parenthesized(Parser<OmpObjectList>{}))) ||
  "initializer" >> construct<OmpClause>(construct<OmpClause::Initializer>(parenthesized(Parser<OmpInitializerClause>{}))) ||
  "init" >> construct<OmpClause>(construct<OmpClause::Init>(parenthesized(Parser<OmpInitClause>{}))) ||
  "indirect" >> construct<OmpClause>(construct<OmpClause::Indirect>()) ||
  "inclusive" >> construct<OmpClause>(construct<OmpClause::Inclusive>(parenthesized(Parser<OmpObjectList>{}))) ||
  "inbranch" >> construct<OmpClause>(construct<OmpClause::Inbranch>()) ||
  "in_reduction" >> construct<OmpClause>(construct<OmpClause::InReduction>(parenthesized(Parser<OmpInReductionClause>{}))) ||
  "if" >> construct<OmpClause>(construct<OmpClause::If>(parenthesized(Parser<OmpIfClause>{}))) ||
  "holds" >> construct<OmpClause>(construct<OmpClause::Holds>(parenthesized(Parser<OmpHoldsClause>{}))) ||
  "hint" >> construct<OmpClause>(construct<OmpClause::Hint>(parenthesized(Parser<OmpHintClause>{}))) ||
  "has_device_addr" >> construct<OmpClause>(construct<OmpClause::HasDeviceAddr>(parenthesized(Parser<OmpObjectList>{}))) ||
  "grainsize" >> construct<OmpClause>(construct<OmpClause::Grainsize>(parenthesized(Parser<OmpGrainsizeClause>{}))) ||
  "full" >> construct<OmpClause>(construct<OmpClause::Full>()) ||
  "from" >> construct<OmpClause>(construct<OmpClause::From>(parenthesized(Parser<OmpFromClause>{}))) ||
  "flush" >> construct<OmpClause>(construct<OmpClause::Flush>()) ||
  "firstprivate" >> construct<OmpClause>(construct<OmpClause::Firstprivate>(parenthesized(Parser<OmpObjectList>{}))) ||
  "final" >> construct<OmpClause>(construct<OmpClause::Final>(parenthesized(scalarLogicalExpr))) ||
  "filter" >> construct<OmpClause>(construct<OmpClause::Filter>(parenthesized(scalarIntExpr))) ||
  "fail" >> construct<OmpClause>(construct<OmpClause::Fail>(parenthesized(Parser<OmpFailClause>{}))) ||
  "exclusive" >> construct<OmpClause>(construct<OmpClause::Exclusive>(parenthesized(Parser<OmpObjectList>{}))) ||
  "enter" >> construct<OmpClause>(construct<OmpClause::Enter>(parenthesized(Parser<OmpObjectList>{}))) ||
  "dynamic_allocators" >> construct<OmpClause>(construct<OmpClause::DynamicAllocators>()) ||
  "doacross" >> construct<OmpClause>(construct<OmpClause::Doacross>(parenthesized(Parser<OmpDoacrossClause>{}))) ||
  "dist_schedule" >> construct<OmpClause>(construct<OmpClause::DistSchedule>(maybe(parenthesized(scalarIntExpr)))) ||
  "device_type" >> construct<OmpClause>(construct<OmpClause::DeviceType>(parenthesized(Parser<OmpDeviceTypeClause>{}))) ||
  "device" >> construct<OmpClause>(construct<OmpClause::Device>(parenthesized(Parser<OmpDeviceClause>{}))) ||
  "detach" >> construct<OmpClause>(construct<OmpClause::Detach>(parenthesized(Parser<OmpDetachClause>{}))) ||
  "destroy" >> construct<OmpClause>(construct<OmpClause::Destroy>(maybe(parenthesized(Parser<OmpDestroyClause>{})))) ||
  "depobj" >> construct<OmpClause>(construct<OmpClause::Depobj>()) ||
  "depend" >> construct<OmpClause>(construct<OmpClause::Depend>(parenthesized(Parser<OmpDependClause>{}))) ||
  "defaultmap" >> construct<OmpClause>(construct<OmpClause::Defaultmap>(parenthesized(Parser<OmpDefaultmapClause>{}))) ||
  "default" >> construct<OmpClause>(construct<OmpClause::Default>(parenthesized(Parser<OmpDefaultClause>{}))) ||
  "copyprivate" >> construct<OmpClause>(construct<OmpClause::Copyprivate>(parenthesized(Parser<OmpObjectList>{}))) ||
  "copyin" >> construct<OmpClause>(construct<OmpClause::Copyin>(parenthesized(Parser<OmpObjectList>{}))) ||
  "contains" >> construct<OmpClause>(construct<OmpClause::Contains>(parenthesized(Parser<OmpContainsClause>{}))) ||
  "compare" >> construct<OmpClause>(construct<OmpClause::Compare>()) ||
  "collapse" >> construct<OmpClause>(construct<OmpClause::Collapse>(parenthesized(scalarIntConstantExpr))) ||
  "capture" >> construct<OmpClause>(construct<OmpClause::Capture>()) ||
  "cancellation_construct_type" >> construct<OmpClause>(construct<OmpClause::CancellationConstructType>(parenthesized(Parser<OmpCancellationConstructTypeClause>{}))) ||
  "bind" >> construct<OmpClause>(construct<OmpClause::Bind>(parenthesized(Parser<OmpBindClause>{}))) ||
  "atomic_default_mem_order" >> construct<OmpClause>(construct<OmpClause::AtomicDefaultMemOrder>(parenthesized(Parser<OmpAtomicDefaultMemOrderClause>{}))) ||
  "at" >> construct<OmpClause>(construct<OmpClause::At>(parenthesized(Parser<OmpAtClause>{}))) ||
  "append_args" >> construct<OmpClause>(construct<OmpClause::AppendArgs>()) ||
  "allocator" >> construct<OmpClause>(construct<OmpClause::Allocator>(parenthesized(scalarIntExpr))) ||
  "allocate" >> construct<OmpClause>(construct<OmpClause::Allocate>(parenthesized(Parser<OmpAllocateClause>{}))) ||
  "aligned" >> construct<OmpClause>(construct<OmpClause::Aligned>(parenthesized(Parser<OmpAlignedClause>{}))) ||
  "align" >> construct<OmpClause>(construct<OmpClause::Align>(parenthesized(Parser<OmpAlignClause>{}))) ||
  "affinity" >> construct<OmpClause>(construct<OmpClause::Affinity>(parenthesized(Parser<OmpAffinityClause>{}))) ||
  "adjust_args" >> construct<OmpClause>(construct<OmpClause::AdjustArgs>()) ||
  "acquire" >> construct<OmpClause>(construct<OmpClause::Acquire>()) ||
  "acq_rel" >> construct<OmpClause>(construct<OmpClause::AcqRel>()) ||
  "absent" >> construct<OmpClause>(construct<OmpClause::Absent>(parenthesized(Parser<OmpAbsentClause>{})))
)

#endif // GEN_FLANG_CLAUSES_PARSER

#ifdef GEN_CLANG_CLAUSE_CLASS
#undef GEN_CLANG_CLAUSE_CLASS

#ifndef CLAUSE
#define CLAUSE(Enum, Str, Implicit)
#endif
#ifndef CLAUSE_CLASS
#define CLAUSE_CLASS(Enum, Str, Class)
#endif
#ifndef CLAUSE_NO_CLASS
#define CLAUSE_NO_CLASS(Enum, Str)
#endif

#define __CLAUSE(Name, Class)                      \
  CLAUSE(OMPC_##Name, #Name, /* Implicit */ false) \
  CLAUSE_CLASS(OMPC_##Name, #Name, Class)
#define __CLAUSE_NO_CLASS(Name)                    \
  CLAUSE(OMPC_##Name, #Name, /* Implicit */ false) \
  CLAUSE_NO_CLASS(OMPC_##Name, #Name)
#define __IMPLICIT_CLAUSE_CLASS(Name, Str, Class)  \
  CLAUSE(OMPC_##Name, Str, /* Implicit */ true)    \
  CLAUSE_CLASS(OMPC_##Name, Str, Class)
#define __IMPLICIT_CLAUSE_NO_CLASS(Name, Str)      \
  CLAUSE(OMPC_##Name, Str, /* Implicit */ true)    \
  CLAUSE_NO_CLASS(OMPC_##Name, Str)

__CLAUSE(absent, OMPAbsentClause)
__CLAUSE(acq_rel, OMPAcqRelClause)
__CLAUSE(acquire, OMPAcquireClause)
__CLAUSE_NO_CLASS(adjust_args)
__CLAUSE(affinity, OMPAffinityClause)
__CLAUSE(align, OMPAlignClause)
__CLAUSE(aligned, OMPAlignedClause)
__CLAUSE(allocate, OMPAllocateClause)
__CLAUSE(allocator, OMPAllocatorClause)
__CLAUSE_NO_CLASS(append_args)
__CLAUSE(at, OMPAtClause)
__CLAUSE(atomic_default_mem_order, OMPAtomicDefaultMemOrderClause)
__CLAUSE(bind, OMPBindClause)
__CLAUSE_NO_CLASS(cancellation_construct_type)
__CLAUSE(capture, OMPCaptureClause)
__CLAUSE(collapse, OMPCollapseClause)
__CLAUSE(compare, OMPCompareClause)
__CLAUSE(contains, OMPContainsClause)
__CLAUSE(copyprivate, OMPCopyprivateClause)
__CLAUSE(copyin, OMPCopyinClause)
__CLAUSE(default, OMPDefaultClause)
__CLAUSE(defaultmap, OMPDefaultmapClause)
__CLAUSE(depend, OMPDependClause)
__IMPLICIT_CLAUSE_CLASS(depobj, "depobj", OMPDepobjClause)
__CLAUSE(destroy, OMPDestroyClause)
__CLAUSE(detach, OMPDetachClause)
__CLAUSE(device, OMPDeviceClause)
__CLAUSE_NO_CLASS(device_type)
__CLAUSE(dist_schedule, OMPDistScheduleClause)
__CLAUSE(doacross, OMPDoacrossClause)
__CLAUSE(dynamic_allocators, OMPDynamicAllocatorsClause)
__CLAUSE_NO_CLASS(enter)
__CLAUSE(exclusive, OMPExclusiveClause)
__CLAUSE(fail, OMPFailClause)
__CLAUSE(filter, OMPFilterClause)
__CLAUSE(final, OMPFinalClause)
__CLAUSE(firstprivate, OMPFirstprivateClause)
__IMPLICIT_CLAUSE_CLASS(flush, "flush", OMPFlushClause)
__CLAUSE(from, OMPFromClause)
__CLAUSE(full, OMPFullClause)
__CLAUSE(grainsize, OMPGrainsizeClause)
__CLAUSE(has_device_addr, OMPHasDeviceAddrClause)
__CLAUSE(hint, OMPHintClause)
__CLAUSE(holds, OMPHoldsClause)
__CLAUSE(if, OMPIfClause)
__CLAUSE(in_reduction, OMPInReductionClause)
__CLAUSE_NO_CLASS(inbranch)
__CLAUSE(inclusive, OMPInclusiveClause)
__CLAUSE_NO_CLASS(indirect)
__CLAUSE(init, OMPInitClause)
__CLAUSE_NO_CLASS(initializer)
__CLAUSE(is_device_ptr, OMPIsDevicePtrClause)
__CLAUSE(lastprivate, OMPLastprivateClause)
__CLAUSE(linear, OMPLinearClause)
__CLAUSE_NO_CLASS(link)
__CLAUSE(map, OMPMapClause)
__CLAUSE_NO_CLASS(match)
__CLAUSE_NO_CLASS(memory_order)
__CLAUSE(mergeable, OMPMergeableClause)
__CLAUSE(message, OMPMessageClause)
__CLAUSE(nogroup, OMPNogroupClause)
__CLAUSE(no_openmp, OMPNoOpenMPClause)
__CLAUSE(no_openmp_constructs, OMPNoOpenMPConstructsClause)
__CLAUSE(no_openmp_routines, OMPNoOpenMPRoutinesClause)
__CLAUSE(no_parallelism, OMPNoParallelismClause)
__CLAUSE(nowait, OMPNowaitClause)
__CLAUSE(nocontext, OMPNocontextClause)
__CLAUSE(nontemporal, OMPNontemporalClause)
__CLAUSE_NO_CLASS(notinbranch)
__CLAUSE(novariants, OMPNovariantsClause)
__CLAUSE(num_tasks, OMPNumTasksClause)
__CLAUSE(num_teams, OMPNumTeamsClause)
__CLAUSE(num_threads, OMPNumThreadsClause)
__CLAUSE(ompx_attribute, OMPXAttributeClause)
__CLAUSE(ompx_bare, OMPXBareClause)
__CLAUSE(ompx_dyn_cgroup_mem, OMPXDynCGroupMemClause)
__CLAUSE(order, OMPOrderClause)
__CLAUSE(ordered, OMPOrderedClause)
__CLAUSE_NO_CLASS(otherwise)
__CLAUSE(partial, OMPPartialClause)
__CLAUSE(permutation, OMPPermutationClause)
__CLAUSE(priority, OMPPriorityClause)
__CLAUSE(private, OMPPrivateClause)
__CLAUSE(proc_bind, OMPProcBindClause)
__CLAUSE(read, OMPReadClause)
__CLAUSE(reduction, OMPReductionClause)
__CLAUSE(relaxed, OMPRelaxedClause)
__CLAUSE(release, OMPReleaseClause)
__CLAUSE(reverse_offload, OMPReverseOffloadClause)
__CLAUSE(safelen, OMPSafelenClause)
__CLAUSE(schedule, OMPScheduleClause)
__CLAUSE(self_maps, OMPSelfMapsClause)
__CLAUSE(seq_cst, OMPSeqCstClause)
__CLAUSE(severity, OMPSeverityClause)
__CLAUSE(shared, OMPSharedClause)
__CLAUSE(simd, OMPSIMDClause)
__CLAUSE(simdlen, OMPSimdlenClause)
__CLAUSE(sizes, OMPSizesClause)
__CLAUSE(task_reduction, OMPTaskReductionClause)
__CLAUSE(thread_limit, OMPThreadLimitClause)
__IMPLICIT_CLAUSE_NO_CLASS(threadprivate, "threadprivate")
__CLAUSE(threads, OMPThreadsClause)
__CLAUSE(to, OMPToClause)
__CLAUSE(unified_address, OMPUnifiedAddressClause)
__CLAUSE(unified_shared_memory, OMPUnifiedSharedMemoryClause)
__CLAUSE_NO_CLASS(uniform)
__IMPLICIT_CLAUSE_NO_CLASS(unknown, "unknown")
__CLAUSE(untied, OMPUntiedClause)
__CLAUSE(update, OMPUpdateClause)
__CLAUSE(use, OMPUseClause)
__CLAUSE(use_device_addr, OMPUseDeviceAddrClause)
__CLAUSE(use_device_ptr, OMPUseDevicePtrClause)
__CLAUSE(uses_allocators, OMPUsesAllocatorsClause)
__CLAUSE(weak, OMPWeakClause)
__CLAUSE_NO_CLASS(when)
__CLAUSE(write, OMPWriteClause)

#undef __IMPLICIT_CLAUSE_NO_CLASS
#undef __IMPLICIT_CLAUSE_CLASS
#undef __CLAUSE_NO_CLASS
#undef __CLAUSE
#undef CLAUSE_NO_CLASS
#undef CLAUSE_CLASS
#undef CLAUSE

#endif // GEN_CLANG_CLAUSE_CLASS

#ifdef GEN_DIRECTIVES_IMPL
#undef GEN_DIRECTIVES_IMPL

#include "llvm/Support/ErrorHandling.h"

Directive llvm::omp::getOpenMPDirectiveKind(llvm::StringRef Str) {
  return llvm::StringSwitch<Directive>(Str)
    .Case("allocate",OMPD_allocate)
    .Case("allocators",OMPD_allocators)
    .Case("assume",OMPD_assume)
    .Case("assumes",OMPD_assumes)
    .Case("atomic",OMPD_atomic)
    .Case("barrier",OMPD_barrier)
    .Case("begin assumes",OMPD_begin_assumes)
    .Case("begin declare target",OMPD_begin_declare_target)
    .Case("begin declare variant",OMPD_begin_declare_variant)
    .Case("cancel",OMPD_cancel)
    .Case("cancellation point",OMPD_cancellation_point)
    .Case("critical",OMPD_critical)
    .Case("declare mapper",OMPD_declare_mapper)
    .Case("declare reduction",OMPD_declare_reduction)
    .Case("declare simd",OMPD_declare_simd)
    .Case("declare target",OMPD_declare_target)
    .Case("declare variant",OMPD_declare_variant)
    .Case("depobj",OMPD_depobj)
    .Case("distribute",OMPD_distribute)
    .Case("distribute parallel do",OMPD_distribute_parallel_do)
    .Case("distribute parallel do simd",OMPD_distribute_parallel_do_simd)
    .Case("distribute parallel for",OMPD_distribute_parallel_for)
    .Case("distribute parallel for simd",OMPD_distribute_parallel_for_simd)
    .Case("distribute simd",OMPD_distribute_simd)
    .Case("do",OMPD_do)
    .Case("do simd",OMPD_do_simd)
    .Case("end assumes",OMPD_end_assumes)
    .Case("end declare target",OMPD_end_declare_target)
    .Case("end declare variant",OMPD_end_declare_variant)
    .Case("end do",OMPD_end_do)
    .Case("end do simd",OMPD_end_do_simd)
    .Case("end scope",OMPD_end_scope)
    .Case("end sections",OMPD_end_sections)
    .Case("end single",OMPD_end_single)
    .Case("end workshare",OMPD_end_workshare)
    .Case("error",OMPD_error)
    .Case("flush",OMPD_flush)
    .Case("for",OMPD_for)
    .Case("for simd",OMPD_for_simd)
    .Case("interchange",OMPD_interchange)
    .Case("masked taskloop",OMPD_masked_taskloop)
    .Case("masked taskloop simd",OMPD_masked_taskloop_simd)
    .Case("master",OMPD_master)
    .Case("master taskloop",OMPD_master_taskloop)
    .Case("master taskloop simd",OMPD_master_taskloop_simd)
    .Case("metadirective",OMPD_metadirective)
    .Case("nothing",OMPD_nothing)
    .Case("ordered",OMPD_ordered)
    .Case("parallel",OMPD_parallel)
    .Case("parallel do",OMPD_parallel_do)
    .Case("parallel do simd",OMPD_parallel_do_simd)
    .Case("parallel for",OMPD_parallel_for)
    .Case("parallel for simd",OMPD_parallel_for_simd)
    .Case("parallel masked",OMPD_parallel_masked)
    .Case("parallel masked taskloop",OMPD_parallel_masked_taskloop)
    .Case("parallel masked taskloop simd",OMPD_parallel_masked_taskloop_simd)
    .Case("parallel master",OMPD_parallel_master)
    .Case("parallel master taskloop",OMPD_parallel_master_taskloop)
    .Case("parallel master taskloop simd",OMPD_parallel_master_taskloop_simd)
    .Case("parallel sections",OMPD_parallel_sections)
    .Case("parallel workshare",OMPD_parallel_workshare)
    .Case("requires",OMPD_requires)
    .Case("reverse",OMPD_reverse)
    .Case("scan",OMPD_scan)
    .Case("scope",OMPD_scope)
    .Case("section",OMPD_section)
    .Case("sections",OMPD_sections)
    .Case("simd",OMPD_simd)
    .Case("single",OMPD_single)
    .Case("stripe",OMPD_stripe)
    .Case("target",OMPD_target)
    .Case("target data",OMPD_target_data)
    .Case("target enter data",OMPD_target_enter_data)
    .Case("target exit data",OMPD_target_exit_data)
    .Case("target parallel",OMPD_target_parallel)
    .Case("target parallel do",OMPD_target_parallel_do)
    .Case("target parallel do simd",OMPD_target_parallel_do_simd)
    .Case("target parallel for",OMPD_target_parallel_for)
    .Case("target parallel for simd",OMPD_target_parallel_for_simd)
    .Case("target simd",OMPD_target_simd)
    .Case("target teams",OMPD_target_teams)
    .Case("target teams distribute",OMPD_target_teams_distribute)
    .Case("target teams distribute parallel do",OMPD_target_teams_distribute_parallel_do)
    .Case("target teams distribute parallel do simd",OMPD_target_teams_distribute_parallel_do_simd)
    .Case("target teams distribute parallel for",OMPD_target_teams_distribute_parallel_for)
    .Case("target teams distribute parallel for simd",OMPD_target_teams_distribute_parallel_for_simd)
    .Case("target teams distribute simd",OMPD_target_teams_distribute_simd)
    .Case("target update",OMPD_target_update)
    .Case("task",OMPD_task)
    .Case("taskgroup",OMPD_taskgroup)
    .Case("taskloop",OMPD_taskloop)
    .Case("taskloop simd",OMPD_taskloop_simd)
    .Case("taskwait",OMPD_taskwait)
    .Case("taskyield",OMPD_taskyield)
    .Case("teams",OMPD_teams)
    .Case("teams distribute",OMPD_teams_distribute)
    .Case("teams distribute parallel do",OMPD_teams_distribute_parallel_do)
    .Case("teams distribute parallel do simd",OMPD_teams_distribute_parallel_do_simd)
    .Case("teams distribute parallel for",OMPD_teams_distribute_parallel_for)
    .Case("teams distribute parallel for simd",OMPD_teams_distribute_parallel_for_simd)
    .Case("teams distribute simd",OMPD_teams_distribute_simd)
    .Case("threadprivate",OMPD_threadprivate)
    .Case("tile",OMPD_tile)
    .Case("unknown",OMPD_unknown)
    .Case("unroll",OMPD_unroll)
    .Case("workshare",OMPD_workshare)
    .Case("dispatch",OMPD_dispatch)
    .Case("interop",OMPD_interop)
    .Case("loop",OMPD_loop)
    .Case("masked",OMPD_masked)
    .Case("parallel loop",OMPD_parallel_loop)
    .Case("target loop",OMPD_target_loop)
    .Case("target parallel loop",OMPD_target_parallel_loop)
    .Case("target teams loop",OMPD_target_teams_loop)
    .Case("teams loop",OMPD_teams_loop)
    .Default(OMPD_unknown);
}

llvm::StringRef llvm::omp::getOpenMPDirectiveName(Directive Kind) {
  switch (Kind) {
    case OMPD_allocate:
      return "allocate";
    case OMPD_allocators:
      return "allocators";
    case OMPD_assume:
      return "assume";
    case OMPD_assumes:
      return "assumes";
    case OMPD_atomic:
      return "atomic";
    case OMPD_barrier:
      return "barrier";
    case OMPD_begin_assumes:
      return "begin assumes";
    case OMPD_begin_declare_target:
      return "begin declare target";
    case OMPD_begin_declare_variant:
      return "begin declare variant";
    case OMPD_cancel:
      return "cancel";
    case OMPD_cancellation_point:
      return "cancellation point";
    case OMPD_critical:
      return "critical";
    case OMPD_declare_mapper:
      return "declare mapper";
    case OMPD_declare_reduction:
      return "declare reduction";
    case OMPD_declare_simd:
      return "declare simd";
    case OMPD_declare_target:
      return "declare target";
    case OMPD_declare_variant:
      return "declare variant";
    case OMPD_depobj:
      return "depobj";
    case OMPD_distribute:
      return "distribute";
    case OMPD_distribute_parallel_do:
      return "distribute parallel do";
    case OMPD_distribute_parallel_do_simd:
      return "distribute parallel do simd";
    case OMPD_distribute_parallel_for:
      return "distribute parallel for";
    case OMPD_distribute_parallel_for_simd:
      return "distribute parallel for simd";
    case OMPD_distribute_simd:
      return "distribute simd";
    case OMPD_do:
      return "do";
    case OMPD_do_simd:
      return "do simd";
    case OMPD_end_assumes:
      return "end assumes";
    case OMPD_end_declare_target:
      return "end declare target";
    case OMPD_end_declare_variant:
      return "end declare variant";
    case OMPD_end_do:
      return "end do";
    case OMPD_end_do_simd:
      return "end do simd";
    case OMPD_end_scope:
      return "end scope";
    case OMPD_end_sections:
      return "end sections";
    case OMPD_end_single:
      return "end single";
    case OMPD_end_workshare:
      return "end workshare";
    case OMPD_error:
      return "error";
    case OMPD_flush:
      return "flush";
    case OMPD_for:
      return "for";
    case OMPD_for_simd:
      return "for simd";
    case OMPD_interchange:
      return "interchange";
    case OMPD_masked_taskloop:
      return "masked taskloop";
    case OMPD_masked_taskloop_simd:
      return "masked taskloop simd";
    case OMPD_master:
      return "master";
    case OMPD_master_taskloop:
      return "master taskloop";
    case OMPD_master_taskloop_simd:
      return "master taskloop simd";
    case OMPD_metadirective:
      return "metadirective";
    case OMPD_nothing:
      return "nothing";
    case OMPD_ordered:
      return "ordered";
    case OMPD_parallel:
      return "parallel";
    case OMPD_parallel_do:
      return "parallel do";
    case OMPD_parallel_do_simd:
      return "parallel do simd";
    case OMPD_parallel_for:
      return "parallel for";
    case OMPD_parallel_for_simd:
      return "parallel for simd";
    case OMPD_parallel_masked:
      return "parallel masked";
    case OMPD_parallel_masked_taskloop:
      return "parallel masked taskloop";
    case OMPD_parallel_masked_taskloop_simd:
      return "parallel masked taskloop simd";
    case OMPD_parallel_master:
      return "parallel master";
    case OMPD_parallel_master_taskloop:
      return "parallel master taskloop";
    case OMPD_parallel_master_taskloop_simd:
      return "parallel master taskloop simd";
    case OMPD_parallel_sections:
      return "parallel sections";
    case OMPD_parallel_workshare:
      return "parallel workshare";
    case OMPD_requires:
      return "requires";
    case OMPD_reverse:
      return "reverse";
    case OMPD_scan:
      return "scan";
    case OMPD_scope:
      return "scope";
    case OMPD_section:
      return "section";
    case OMPD_sections:
      return "sections";
    case OMPD_simd:
      return "simd";
    case OMPD_single:
      return "single";
    case OMPD_stripe:
      return "stripe";
    case OMPD_target:
      return "target";
    case OMPD_target_data:
      return "target data";
    case OMPD_target_enter_data:
      return "target enter data";
    case OMPD_target_exit_data:
      return "target exit data";
    case OMPD_target_parallel:
      return "target parallel";
    case OMPD_target_parallel_do:
      return "target parallel do";
    case OMPD_target_parallel_do_simd:
      return "target parallel do simd";
    case OMPD_target_parallel_for:
      return "target parallel for";
    case OMPD_target_parallel_for_simd:
      return "target parallel for simd";
    case OMPD_target_simd:
      return "target simd";
    case OMPD_target_teams:
      return "target teams";
    case OMPD_target_teams_distribute:
      return "target teams distribute";
    case OMPD_target_teams_distribute_parallel_do:
      return "target teams distribute parallel do";
    case OMPD_target_teams_distribute_parallel_do_simd:
      return "target teams distribute parallel do simd";
    case OMPD_target_teams_distribute_parallel_for:
      return "target teams distribute parallel for";
    case OMPD_target_teams_distribute_parallel_for_simd:
      return "target teams distribute parallel for simd";
    case OMPD_target_teams_distribute_simd:
      return "target teams distribute simd";
    case OMPD_target_update:
      return "target update";
    case OMPD_task:
      return "task";
    case OMPD_taskgroup:
      return "taskgroup";
    case OMPD_taskloop:
      return "taskloop";
    case OMPD_taskloop_simd:
      return "taskloop simd";
    case OMPD_taskwait:
      return "taskwait";
    case OMPD_taskyield:
      return "taskyield";
    case OMPD_teams:
      return "teams";
    case OMPD_teams_distribute:
      return "teams distribute";
    case OMPD_teams_distribute_parallel_do:
      return "teams distribute parallel do";
    case OMPD_teams_distribute_parallel_do_simd:
      return "teams distribute parallel do simd";
    case OMPD_teams_distribute_parallel_for:
      return "teams distribute parallel for";
    case OMPD_teams_distribute_parallel_for_simd:
      return "teams distribute parallel for simd";
    case OMPD_teams_distribute_simd:
      return "teams distribute simd";
    case OMPD_threadprivate:
      return "threadprivate";
    case OMPD_tile:
      return "tile";
    case OMPD_unknown:
      return "unknown";
    case OMPD_unroll:
      return "unroll";
    case OMPD_workshare:
      return "workshare";
    case OMPD_dispatch:
      return "dispatch";
    case OMPD_interop:
      return "interop";
    case OMPD_loop:
      return "loop";
    case OMPD_masked:
      return "masked";
    case OMPD_parallel_loop:
      return "parallel loop";
    case OMPD_target_loop:
      return "target loop";
    case OMPD_target_parallel_loop:
      return "target parallel loop";
    case OMPD_target_teams_loop:
      return "target teams loop";
    case OMPD_teams_loop:
      return "teams loop";
  }
  llvm_unreachable("Invalid OpenMP Directive kind");
}

Clause llvm::omp::getOpenMPClauseKind(llvm::StringRef Str) {
  return llvm::StringSwitch<Clause>(Str)
    .Case("absent",OMPC_absent)
    .Case("acq_rel",OMPC_acq_rel)
    .Case("acquire",OMPC_acquire)
    .Case("adjust_args",OMPC_adjust_args)
    .Case("affinity",OMPC_affinity)
    .Case("align",OMPC_align)
    .Case("aligned",OMPC_aligned)
    .Case("allocate",OMPC_allocate)
    .Case("allocator",OMPC_allocator)
    .Case("append_args",OMPC_append_args)
    .Case("at",OMPC_at)
    .Case("atomic_default_mem_order",OMPC_atomic_default_mem_order)
    .Case("bind",OMPC_bind)
    .Case("cancellation_construct_type",OMPC_cancellation_construct_type)
    .Case("capture",OMPC_capture)
    .Case("collapse",OMPC_collapse)
    .Case("compare",OMPC_compare)
    .Case("contains",OMPC_contains)
    .Case("copyprivate",OMPC_copyprivate)
    .Case("copyin",OMPC_copyin)
    .Case("default",OMPC_default)
    .Case("defaultmap",OMPC_defaultmap)
    .Case("depend",OMPC_depend)
    .Case("depobj",OMPC_unknown)
    .Case("destroy",OMPC_destroy)
    .Case("detach",OMPC_detach)
    .Case("device",OMPC_device)
    .Case("device_type",OMPC_device_type)
    .Case("dist_schedule",OMPC_dist_schedule)
    .Case("doacross",OMPC_doacross)
    .Case("dynamic_allocators",OMPC_dynamic_allocators)
    .Case("enter",OMPC_enter)
    .Case("exclusive",OMPC_exclusive)
    .Case("fail",OMPC_fail)
    .Case("filter",OMPC_filter)
    .Case("final",OMPC_final)
    .Case("firstprivate",OMPC_firstprivate)
    .Case("flush",OMPC_unknown)
    .Case("from",OMPC_from)
    .Case("full",OMPC_full)
    .Case("grainsize",OMPC_grainsize)
    .Case("has_device_addr",OMPC_has_device_addr)
    .Case("hint",OMPC_hint)
    .Case("holds",OMPC_holds)
    .Case("if",OMPC_if)
    .Case("in_reduction",OMPC_in_reduction)
    .Case("inbranch",OMPC_inbranch)
    .Case("inclusive",OMPC_inclusive)
    .Case("indirect",OMPC_indirect)
    .Case("init",OMPC_init)
    .Case("initializer",OMPC_initializer)
    .Case("is_device_ptr",OMPC_is_device_ptr)
    .Case("lastprivate",OMPC_lastprivate)
    .Case("linear",OMPC_linear)
    .Case("link",OMPC_link)
    .Case("map",OMPC_map)
    .Case("match",OMPC_match)
    .Case("memory_order",OMPC_memory_order)
    .Case("mergeable",OMPC_mergeable)
    .Case("message",OMPC_message)
    .Case("nogroup",OMPC_nogroup)
    .Case("no_openmp",OMPC_no_openmp)
    .Case("no_openmp_constructs",OMPC_no_openmp_constructs)
    .Case("no_openmp_routines",OMPC_no_openmp_routines)
    .Case("no_parallelism",OMPC_no_parallelism)
    .Case("nowait",OMPC_nowait)
    .Case("nocontext",OMPC_nocontext)
    .Case("nontemporal",OMPC_nontemporal)
    .Case("notinbranch",OMPC_notinbranch)
    .Case("novariants",OMPC_novariants)
    .Case("num_tasks",OMPC_num_tasks)
    .Case("num_teams",OMPC_num_teams)
    .Case("num_threads",OMPC_num_threads)
    .Case("ompx_attribute",OMPC_ompx_attribute)
    .Case("ompx_bare",OMPC_ompx_bare)
    .Case("ompx_dyn_cgroup_mem",OMPC_ompx_dyn_cgroup_mem)
    .Case("order",OMPC_order)
    .Case("ordered",OMPC_ordered)
    .Case("otherwise",OMPC_otherwise)
    .Case("partial",OMPC_partial)
    .Case("permutation",OMPC_permutation)
    .Case("priority",OMPC_priority)
    .Case("private",OMPC_private)
    .Case("proc_bind",OMPC_proc_bind)
    .Case("read",OMPC_read)
    .Case("reduction",OMPC_reduction)
    .Case("relaxed",OMPC_relaxed)
    .Case("release",OMPC_release)
    .Case("reverse_offload",OMPC_reverse_offload)
    .Case("safelen",OMPC_safelen)
    .Case("schedule",OMPC_schedule)
    .Case("self_maps",OMPC_self_maps)
    .Case("seq_cst",OMPC_seq_cst)
    .Case("severity",OMPC_severity)
    .Case("shared",OMPC_shared)
    .Case("simd",OMPC_simd)
    .Case("simdlen",OMPC_simdlen)
    .Case("sizes",OMPC_sizes)
    .Case("task_reduction",OMPC_task_reduction)
    .Case("thread_limit",OMPC_thread_limit)
    .Case("threadprivate",OMPC_unknown)
    .Case("threads",OMPC_threads)
    .Case("to",OMPC_to)
    .Case("unified_address",OMPC_unified_address)
    .Case("unified_shared_memory",OMPC_unified_shared_memory)
    .Case("uniform",OMPC_uniform)
    .Case("unknown",OMPC_unknown)
    .Case("untied",OMPC_untied)
    .Case("update",OMPC_update)
    .Case("use",OMPC_use)
    .Case("use_device_addr",OMPC_use_device_addr)
    .Case("use_device_ptr",OMPC_use_device_ptr)
    .Case("uses_allocators",OMPC_uses_allocators)
    .Case("weak",OMPC_weak)
    .Case("when",OMPC_when)
    .Case("write",OMPC_write)
    .Default(OMPC_unknown);
}

llvm::StringRef llvm::omp::getOpenMPClauseName(Clause Kind) {
  switch (Kind) {
    case OMPC_absent:
      return "absent";
    case OMPC_acq_rel:
      return "acq_rel";
    case OMPC_acquire:
      return "acquire";
    case OMPC_adjust_args:
      return "adjust_args";
    case OMPC_affinity:
      return "affinity";
    case OMPC_align:
      return "align";
    case OMPC_aligned:
      return "aligned";
    case OMPC_allocate:
      return "allocate";
    case OMPC_allocator:
      return "allocator";
    case OMPC_append_args:
      return "append_args";
    case OMPC_at:
      return "at";
    case OMPC_atomic_default_mem_order:
      return "atomic_default_mem_order";
    case OMPC_bind:
      return "bind";
    case OMPC_cancellation_construct_type:
      return "cancellation_construct_type";
    case OMPC_capture:
      return "capture";
    case OMPC_collapse:
      return "collapse";
    case OMPC_compare:
      return "compare";
    case OMPC_contains:
      return "contains";
    case OMPC_copyprivate:
      return "copyprivate";
    case OMPC_copyin:
      return "copyin";
    case OMPC_default:
      return "default";
    case OMPC_defaultmap:
      return "defaultmap";
    case OMPC_depend:
      return "depend";
    case OMPC_depobj:
      return "depobj";
    case OMPC_destroy:
      return "destroy";
    case OMPC_detach:
      return "detach";
    case OMPC_device:
      return "device";
    case OMPC_device_type:
      return "device_type";
    case OMPC_dist_schedule:
      return "dist_schedule";
    case OMPC_doacross:
      return "doacross";
    case OMPC_dynamic_allocators:
      return "dynamic_allocators";
    case OMPC_enter:
      return "enter";
    case OMPC_exclusive:
      return "exclusive";
    case OMPC_fail:
      return "fail";
    case OMPC_filter:
      return "filter";
    case OMPC_final:
      return "final";
    case OMPC_firstprivate:
      return "firstprivate";
    case OMPC_flush:
      return "flush";
    case OMPC_from:
      return "from";
    case OMPC_full:
      return "full";
    case OMPC_grainsize:
      return "grainsize";
    case OMPC_has_device_addr:
      return "has_device_addr";
    case OMPC_hint:
      return "hint";
    case OMPC_holds:
      return "holds";
    case OMPC_if:
      return "if";
    case OMPC_in_reduction:
      return "in_reduction";
    case OMPC_inbranch:
      return "inbranch";
    case OMPC_inclusive:
      return "inclusive";
    case OMPC_indirect:
      return "indirect";
    case OMPC_init:
      return "init";
    case OMPC_initializer:
      return "initializer";
    case OMPC_is_device_ptr:
      return "is_device_ptr";
    case OMPC_lastprivate:
      return "lastprivate";
    case OMPC_linear:
      return "linear";
    case OMPC_link:
      return "link";
    case OMPC_map:
      return "map";
    case OMPC_match:
      return "match";
    case OMPC_memory_order:
      return "memory_order";
    case OMPC_mergeable:
      return "mergeable";
    case OMPC_message:
      return "message";
    case OMPC_nogroup:
      return "nogroup";
    case OMPC_no_openmp:
      return "no_openmp";
    case OMPC_no_openmp_constructs:
      return "no_openmp_constructs";
    case OMPC_no_openmp_routines:
      return "no_openmp_routines";
    case OMPC_no_parallelism:
      return "no_parallelism";
    case OMPC_nowait:
      return "nowait";
    case OMPC_nocontext:
      return "nocontext";
    case OMPC_nontemporal:
      return "nontemporal";
    case OMPC_notinbranch:
      return "notinbranch";
    case OMPC_novariants:
      return "novariants";
    case OMPC_num_tasks:
      return "num_tasks";
    case OMPC_num_teams:
      return "num_teams";
    case OMPC_num_threads:
      return "num_threads";
    case OMPC_ompx_attribute:
      return "ompx_attribute";
    case OMPC_ompx_bare:
      return "ompx_bare";
    case OMPC_ompx_dyn_cgroup_mem:
      return "ompx_dyn_cgroup_mem";
    case OMPC_order:
      return "order";
    case OMPC_ordered:
      return "ordered";
    case OMPC_otherwise:
      return "otherwise";
    case OMPC_partial:
      return "partial";
    case OMPC_permutation:
      return "permutation";
    case OMPC_priority:
      return "priority";
    case OMPC_private:
      return "private";
    case OMPC_proc_bind:
      return "proc_bind";
    case OMPC_read:
      return "read";
    case OMPC_reduction:
      return "reduction";
    case OMPC_relaxed:
      return "relaxed";
    case OMPC_release:
      return "release";
    case OMPC_reverse_offload:
      return "reverse_offload";
    case OMPC_safelen:
      return "safelen";
    case OMPC_schedule:
      return "schedule";
    case OMPC_self_maps:
      return "self_maps";
    case OMPC_seq_cst:
      return "seq_cst";
    case OMPC_severity:
      return "severity";
    case OMPC_shared:
      return "shared";
    case OMPC_simd:
      return "simd";
    case OMPC_simdlen:
      return "simdlen";
    case OMPC_sizes:
      return "sizes";
    case OMPC_task_reduction:
      return "task_reduction";
    case OMPC_thread_limit:
      return "thread_limit";
    case OMPC_threadprivate:
      return "threadprivate or thread local";
    case OMPC_threads:
      return "threads";
    case OMPC_to:
      return "to";
    case OMPC_unified_address:
      return "unified_address";
    case OMPC_unified_shared_memory:
      return "unified_shared_memory";
    case OMPC_uniform:
      return "uniform";
    case OMPC_unknown:
      return "unknown";
    case OMPC_untied:
      return "untied";
    case OMPC_update:
      return "update";
    case OMPC_use:
      return "use";
    case OMPC_use_device_addr:
      return "use_device_addr";
    case OMPC_use_device_ptr:
      return "use_device_ptr";
    case OMPC_uses_allocators:
      return "uses_allocators";
    case OMPC_weak:
      return "weak";
    case OMPC_when:
      return "when";
    case OMPC_write:
      return "write";
  }
  llvm_unreachable("Invalid OpenMP Clause kind");
}

BindKind llvm::omp::getBindKind(llvm::StringRef Str) {
  return llvm::StringSwitch<BindKind>(Str)
    .Case("parallel",OMP_BIND_parallel)
    .Case("teams",OMP_BIND_teams)
    .Case("thread",OMP_BIND_thread)
    .Default(OMP_BIND_thread);
}

llvm::StringRef llvm::omp::getOpenMPBindKindName(llvm::omp::BindKind x) {
  switch (x) {
    case OMP_BIND_parallel:
      return "parallel";
    case OMP_BIND_teams:
      return "teams";
    case OMP_BIND_thread:
      return "thread";
  }
  llvm_unreachable("Invalid OpenMP BindKind kind");
}

CancellationConstructType llvm::omp::getCancellationConstructType(llvm::StringRef Str) {
  return llvm::StringSwitch<CancellationConstructType>(Str)
    .Case("parallel",OMP_CANCELLATION_CONSTRUCT_Parallel)
    .Case("loop",OMP_CANCELLATION_CONSTRUCT_Loop)
    .Case("sections",OMP_CANCELLATION_CONSTRUCT_Sections)
    .Case("taskgroup",OMP_CANCELLATION_CONSTRUCT_Taskgroup)
    .Case("none",OMP_CANCELLATION_CONSTRUCT_None)
    .Default(OMP_CANCELLATION_CONSTRUCT_None);
}

llvm::StringRef llvm::omp::getOpenMPCancellationConstructTypeName(llvm::omp::CancellationConstructType x) {
  switch (x) {
    case OMP_CANCELLATION_CONSTRUCT_Parallel:
      return "parallel";
    case OMP_CANCELLATION_CONSTRUCT_Loop:
      return "loop";
    case OMP_CANCELLATION_CONSTRUCT_Sections:
      return "sections";
    case OMP_CANCELLATION_CONSTRUCT_Taskgroup:
      return "taskgroup";
    case OMP_CANCELLATION_CONSTRUCT_None:
      return "none";
  }
  llvm_unreachable("Invalid OpenMP CancellationConstructType kind");
}

GrainsizeType llvm::omp::getGrainsizeType(llvm::StringRef Str) {
  return llvm::StringSwitch<GrainsizeType>(Str)
    .Case("strict",OMP_GRAINSIZE_Strict)
    .Case("unknown",OMP_GRAINSIZE_Unknown)
    .Default(OMP_GRAINSIZE_Unknown);
}

llvm::StringRef llvm::omp::getOpenMPGrainsizeTypeName(llvm::omp::GrainsizeType x) {
  switch (x) {
    case OMP_GRAINSIZE_Strict:
      return "strict";
    case OMP_GRAINSIZE_Unknown:
      return "unknown";
  }
  llvm_unreachable("Invalid OpenMP GrainsizeType kind");
}

MemoryOrderKind llvm::omp::getMemoryOrderKind(llvm::StringRef Str) {
  return llvm::StringSwitch<MemoryOrderKind>(Str)
    .Case("seq_cst",OMP_MEMORY_ORDER_SeqCst)
    .Case("acq_rel",OMP_MEMORY_ORDER_AcqRel)
    .Case("acquire",OMP_MEMORY_ORDER_Acquire)
    .Case("release",OMP_MEMORY_ORDER_Release)
    .Case("relaxed",OMP_MEMORY_ORDER_Relaxed)
    .Case("default",OMP_MEMORY_ORDER_Default)
    .Default(OMP_MEMORY_ORDER_Default);
}

llvm::StringRef llvm::omp::getOpenMPMemoryOrderKindName(llvm::omp::MemoryOrderKind x) {
  switch (x) {
    case OMP_MEMORY_ORDER_SeqCst:
      return "seq_cst";
    case OMP_MEMORY_ORDER_AcqRel:
      return "acq_rel";
    case OMP_MEMORY_ORDER_Acquire:
      return "acquire";
    case OMP_MEMORY_ORDER_Release:
      return "release";
    case OMP_MEMORY_ORDER_Relaxed:
      return "relaxed";
    case OMP_MEMORY_ORDER_Default:
      return "default";
  }
  llvm_unreachable("Invalid OpenMP MemoryOrderKind kind");
}

NumTasksType llvm::omp::getNumTasksType(llvm::StringRef Str) {
  return llvm::StringSwitch<NumTasksType>(Str)
    .Case("strict",OMP_NUMTASKS_Strict)
    .Case("unknown",OMP_NUMTASKS_Unknown)
    .Default(OMP_NUMTASKS_Unknown);
}

llvm::StringRef llvm::omp::getOpenMPNumTasksTypeName(llvm::omp::NumTasksType x) {
  switch (x) {
    case OMP_NUMTASKS_Strict:
      return "strict";
    case OMP_NUMTASKS_Unknown:
      return "unknown";
  }
  llvm_unreachable("Invalid OpenMP NumTasksType kind");
}

OrderKind llvm::omp::getOrderKind(llvm::StringRef Str) {
  return llvm::StringSwitch<OrderKind>(Str)
    .Case("unknown",OMP_ORDER_unknown)
    .Case("concurrent",OMP_ORDER_concurrent)
    .Default(OMP_ORDER_unknown);
}

llvm::StringRef llvm::omp::getOpenMPOrderKindName(llvm::omp::OrderKind x) {
  switch (x) {
    case OMP_ORDER_unknown:
      return "unknown";
    case OMP_ORDER_concurrent:
      return "concurrent";
  }
  llvm_unreachable("Invalid OpenMP OrderKind kind");
}

ProcBindKind llvm::omp::getProcBindKind(llvm::StringRef Str) {
  return llvm::StringSwitch<ProcBindKind>(Str)
    .Case("primary",OMP_PROC_BIND_primary)
    .Case("master",OMP_PROC_BIND_master)
    .Case("close",OMP_PROC_BIND_close)
    .Case("spread",OMP_PROC_BIND_spread)
    .Case("default",OMP_PROC_BIND_default)
    .Case("unknown",OMP_PROC_BIND_unknown)
    .Default(OMP_PROC_BIND_unknown);
}

llvm::StringRef llvm::omp::getOpenMPProcBindKindName(llvm::omp::ProcBindKind x) {
  switch (x) {
    case OMP_PROC_BIND_primary:
      return "primary";
    case OMP_PROC_BIND_master:
      return "master";
    case OMP_PROC_BIND_close:
      return "close";
    case OMP_PROC_BIND_spread:
      return "spread";
    case OMP_PROC_BIND_default:
      return "default";
    case OMP_PROC_BIND_unknown:
      return "unknown";
  }
  llvm_unreachable("Invalid OpenMP ProcBindKind kind");
}

ScheduleKind llvm::omp::getScheduleKind(llvm::StringRef Str) {
  return llvm::StringSwitch<ScheduleKind>(Str)
    .Case("static",OMP_SCHEDULE_Static)
    .Case("dynamic",OMP_SCHEDULE_Dynamic)
    .Case("guided",OMP_SCHEDULE_Guided)
    .Case("auto",OMP_SCHEDULE_Auto)
    .Case("runtime",OMP_SCHEDULE_Runtime)
    .Case("default",OMP_SCHEDULE_Default)
    .Default(OMP_SCHEDULE_Default);
}

llvm::StringRef llvm::omp::getOpenMPScheduleKindName(llvm::omp::ScheduleKind x) {
  switch (x) {
    case OMP_SCHEDULE_Static:
      return "static";
    case OMP_SCHEDULE_Dynamic:
      return "dynamic";
    case OMP_SCHEDULE_Guided:
      return "guided";
    case OMP_SCHEDULE_Auto:
      return "auto";
    case OMP_SCHEDULE_Runtime:
      return "runtime";
    case OMP_SCHEDULE_Default:
      return "default";
  }
  llvm_unreachable("Invalid OpenMP ScheduleKind kind");
}

bool llvm::omp::isAllowedClauseForDirective(Directive D, Clause C, unsigned Version) {
  assert(unsigned(D) <= llvm::omp::Directive_enumSize);
  assert(unsigned(C) <= llvm::omp::Clause_enumSize);
  switch (D) {
    case OMPD_allocate:
      switch (C) {
        case OMPC_align:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_allocator:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_allocators:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_assume:
      switch (C) {
        case OMPC_absent:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_contains:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_holds:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp_routines:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_parallelism:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp_constructs:
          return 60 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_assumes:
      switch (C) {
        case OMPC_absent:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_contains:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_holds:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp_routines:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_parallelism:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_atomic:
      switch (C) {
        case OMPC_capture:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_compare:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_read:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_update:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_write:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_acq_rel:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_acquire:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_fail:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_hint:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_relaxed:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_release:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_seq_cst:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_weak:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_barrier:
      return false;
      break;
    case OMPD_begin_assumes:
      switch (C) {
        case OMPC_absent:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_contains:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_holds:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_openmp_routines:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_no_parallelism:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_begin_declare_target:
      switch (C) {
        case OMPC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_indirect:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_link:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_to:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_begin_declare_variant:
      return false;
      break;
    case OMPD_cancel:
      switch (C) {
        case OMPC_cancellation_construct_type:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_cancellation_point:
      switch (C) {
        case OMPC_cancellation_construct_type:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_critical:
      switch (C) {
        case OMPC_hint:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_declare_mapper:
      switch (C) {
        case OMPC_map:
          return 45 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_declare_reduction:
      switch (C) {
        case OMPC_initializer:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_declare_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uniform:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_inbranch:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_notinbranch:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_declare_target:
      switch (C) {
        case OMPC_enter:
          return 52 <= Version && 2147483647 >= Version;
        case OMPC_indirect:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_link:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_to:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device_type:
          return 50 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_declare_variant:
      switch (C) {
        case OMPC_match:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_adjust_args:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_append_args:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_depobj:
      switch (C) {
        case OMPC_depend:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_depobj:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_destroy:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_update:
          return 50 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_distribute:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_distribute_parallel_do:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_distribute_parallel_do_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_distribute_parallel_for:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_distribute_parallel_for_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_distribute_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_do:
      switch (C) {
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_do_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_end_assumes:
      return false;
      break;
    case OMPD_end_declare_target:
      return false;
      break;
    case OMPD_end_declare_variant:
      return false;
      break;
    case OMPD_end_do:
      switch (C) {
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_end_do_simd:
      switch (C) {
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_end_scope:
      switch (C) {
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_end_sections:
      switch (C) {
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_end_single:
      switch (C) {
        case OMPC_copyprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_end_workshare:
      switch (C) {
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_error:
      switch (C) {
        case OMPC_at:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_message:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_severity:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_flush:
      switch (C) {
        case OMPC_acq_rel:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_acquire:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_flush:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_release:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_seq_cst:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_for:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_for_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_interchange:
      switch (C) {
        case OMPC_permutation:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_masked_taskloop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_filter:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_masked_taskloop_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_filter:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_master:
      return false;
      break;
    case OMPD_master_taskloop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_master_taskloop_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_metadirective:
      switch (C) {
        case OMPC_when:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_otherwise:
          return 52 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 50 <= Version && 51 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_nothing:
      return false;
      break;
    case OMPD_ordered:
      switch (C) {
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_doacross:
          return 52 <= Version && 2147483647 >= Version;
        case OMPC_simd:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_threads:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_do:
      switch (C) {
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_do_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_for:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_for_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_masked:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_filter:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_masked_taskloop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_filter:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_masked_taskloop_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_filter:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_master:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_master_taskloop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_master_taskloop_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_sections:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_workshare:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_requires:
      switch (C) {
        case OMPC_unified_address:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_unified_shared_memory:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_atomic_default_mem_order:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dynamic_allocators:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_self_maps:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reverse_offload:
          return 99 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_reverse:
      return false;
      break;
    case OMPD_scan:
      switch (C) {
        case OMPC_exclusive:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_inclusive:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_scope:
      switch (C) {
        case OMPC_private:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 52 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 52 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_section:
      return false;
      break;
    case OMPD_sections:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_single:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_stripe:
      switch (C) {
        case OMPC_sizes:
          return 60 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_bare:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_data:
      switch (C) {
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_use_device_addr:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_use_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_enter_data:
      switch (C) {
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_exit_data:
      switch (C) {
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_parallel:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_parallel_do:
      switch (C) {
        case OMPC_allocator:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_parallel_do_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_parallel_for:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_parallel_for_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ordered:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_bare:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_distribute:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_distribute_parallel_do:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_distribute_parallel_do_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_distribute_parallel_for:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_distribute_parallel_for_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_distribute_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_update:
      switch (C) {
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_from:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_to:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_task:
      switch (C) {
        case OMPC_affinity:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_detach:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_taskgroup:
      switch (C) {
        case OMPC_allocate:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_task_reduction:
          return 50 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_taskloop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_taskloop_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_mergeable:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nogroup:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_untied:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_final:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_priority:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_grainsize:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_tasks:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_taskwait:
      switch (C) {
        case OMPC_depend:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_taskyield:
      return false;
      break;
    case OMPD_teams:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 52 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_distribute:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_distribute_parallel_do:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_distribute_parallel_do_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_distribute_parallel_for:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_distribute_parallel_for_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_distribute_simd:
      switch (C) {
        case OMPC_aligned:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_linear:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nontemporal:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_dist_schedule:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_safelen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_simdlen:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_threadprivate:
      return false;
      break;
    case OMPD_tile:
      switch (C) {
        case OMPC_sizes:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_unknown:
      return false;
      break;
    case OMPD_unroll:
      switch (C) {
        case OMPC_full:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_partial:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_workshare:
      switch (C) {
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_dispatch:
      switch (C) {
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nocontext:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_novariants:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_interop:
      switch (C) {
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_destroy:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_init:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_use:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_loop:
      switch (C) {
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_bind:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 50 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_masked:
      switch (C) {
        case OMPC_filter:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_parallel_loop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_bind:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_loop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_in_reduction:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_bind:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_parallel_loop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_bind:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_threads:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_proc_bind:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 51 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_target_teams_loop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_defaultmap:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_depend:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_device:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_has_device_addr:
          return 51 <= Version && 2147483647 >= Version;
        case OMPC_if:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_is_device_ptr:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_map:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_uses_allocators:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_bind:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_nowait:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_dyn_cgroup_mem:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case OMPD_teams_loop:
      switch (C) {
        case OMPC_allocate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_lastprivate:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_ompx_attribute:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_private:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_shared:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_bind:
          return 50 <= Version && 2147483647 >= Version;
        case OMPC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_default:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_num_teams:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_order:
          return 1 <= Version && 2147483647 >= Version;
        case OMPC_thread_limit:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
  }
  llvm_unreachable("Invalid OpenMP Directive kind");
}

llvm::omp::Association llvm::omp::getDirectiveAssociation(llvm::omp::Directive Dir) {
  switch (Dir) {
  case llvm::omp::Directive::OMPD_allocate:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_allocators:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_assume:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_assumes:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_atomic:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_barrier:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_begin_assumes:
    return llvm::omp::Association::Delimited;
  case llvm::omp::Directive::OMPD_begin_declare_target:
    return llvm::omp::Association::Delimited;
  case llvm::omp::Directive::OMPD_begin_declare_variant:
    return llvm::omp::Association::Delimited;
  case llvm::omp::Directive::OMPD_cancel:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_cancellation_point:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_critical:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_declare_mapper:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_declare_reduction:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_declare_simd:
    return llvm::omp::Association::Declaration;
  case llvm::omp::Directive::OMPD_declare_target:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_declare_variant:
    return llvm::omp::Association::Declaration;
  case llvm::omp::Directive::OMPD_depobj:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_distribute:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_distribute_parallel_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_distribute_parallel_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_distribute_parallel_for:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_distribute_parallel_for_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_distribute_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_end_assumes:
    return llvm::omp::Association::Delimited;
  case llvm::omp::Directive::OMPD_end_declare_target:
    return llvm::omp::Association::Delimited;
  case llvm::omp::Directive::OMPD_end_declare_variant:
    return llvm::omp::Association::Delimited;
  case llvm::omp::Directive::OMPD_end_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_end_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_end_scope:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_end_sections:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_end_single:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_end_workshare:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_error:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_flush:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_for:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_for_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_interchange:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_masked_taskloop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_masked_taskloop_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_master:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_master_taskloop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_master_taskloop_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_metadirective:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_nothing:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_ordered:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_parallel:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_parallel_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_for:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_for_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_masked:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_parallel_masked_taskloop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_masked_taskloop_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_master:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_parallel_master_taskloop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_master_taskloop_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_parallel_sections:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_parallel_workshare:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_requires:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_reverse:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_scan:
    return llvm::omp::Association::Separating;
  case llvm::omp::Directive::OMPD_scope:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_section:
    return llvm::omp::Association::Separating;
  case llvm::omp::Directive::OMPD_sections:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_single:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_stripe:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_target_data:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_target_enter_data:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_target_exit_data:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_target_parallel:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_target_parallel_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_parallel_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_parallel_for:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_parallel_for_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_target_teams_distribute:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_for:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams_distribute_parallel_for_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams_distribute_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_update:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_task:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_taskgroup:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_taskloop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_taskloop_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_taskwait:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_taskyield:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_teams:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_teams_distribute:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_do_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_for:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_teams_distribute_parallel_for_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_teams_distribute_simd:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_threadprivate:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_tile:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_unknown:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_unroll:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_workshare:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_dispatch:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_interop:
    return llvm::omp::Association::None;
  case llvm::omp::Directive::OMPD_loop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_masked:
    return llvm::omp::Association::Block;
  case llvm::omp::Directive::OMPD_parallel_loop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_loop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_parallel_loop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_target_teams_loop:
    return llvm::omp::Association::Loop;
  case llvm::omp::Directive::OMPD_teams_loop:
    return llvm::omp::Association::Loop;
  } // switch (Dir)
  llvm_unreachable("Unexpected directive");
}

llvm::omp::Category llvm::omp::getDirectiveCategory(llvm::omp::Directive Dir) {
  switch (Dir) {
  case llvm::omp::OMPD_allocate:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_allocators:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_assume:
    return llvm::omp::Category::Informational;
  case llvm::omp::OMPD_assumes:
    return llvm::omp::Category::Informational;
  case llvm::omp::OMPD_atomic:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_barrier:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_begin_assumes:
    return llvm::omp::Category::Informational;
  case llvm::omp::OMPD_begin_declare_target:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_begin_declare_variant:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_cancel:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_cancellation_point:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_critical:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_declare_mapper:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_declare_reduction:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_declare_simd:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_declare_target:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_declare_variant:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_depobj:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_distribute:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_distribute_parallel_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_distribute_parallel_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_distribute_parallel_for:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_distribute_parallel_for_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_distribute_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_end_assumes:
    return llvm::omp::Category::Informational;
  case llvm::omp::OMPD_end_declare_target:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_end_declare_variant:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_end_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_end_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_end_scope:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_end_sections:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_end_single:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_end_workshare:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_error:
    return llvm::omp::Category::Utility;
  case llvm::omp::OMPD_flush:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_for:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_for_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_interchange:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_masked_taskloop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_masked_taskloop_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_master:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_master_taskloop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_master_taskloop_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_metadirective:
    return llvm::omp::Category::Meta;
  case llvm::omp::OMPD_nothing:
    return llvm::omp::Category::Utility;
  case llvm::omp::OMPD_ordered:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_for:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_for_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_masked:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_masked_taskloop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_masked_taskloop_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_master:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_master_taskloop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_master_taskloop_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_sections:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_workshare:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_requires:
    return llvm::omp::Category::Informational;
  case llvm::omp::OMPD_reverse:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_scan:
    return llvm::omp::Category::Subsidiary;
  case llvm::omp::OMPD_scope:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_section:
    return llvm::omp::Category::Subsidiary;
  case llvm::omp::OMPD_sections:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_single:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_stripe:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_data:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_enter_data:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_exit_data:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_parallel:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_parallel_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_parallel_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_parallel_for:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_parallel_for_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_distribute:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_distribute_parallel_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_distribute_parallel_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_distribute_parallel_for:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_distribute_parallel_for_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_distribute_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_update:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_task:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_taskgroup:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_taskloop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_taskloop_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_taskwait:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_taskyield:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_distribute:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_distribute_parallel_do:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_distribute_parallel_do_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_distribute_parallel_for:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_distribute_parallel_for_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_distribute_simd:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_threadprivate:
    return llvm::omp::Category::Declarative;
  case llvm::omp::OMPD_tile:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_unknown:
    return llvm::omp::Category::Utility;
  case llvm::omp::OMPD_unroll:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_workshare:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_dispatch:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_interop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_loop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_masked:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_parallel_loop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_loop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_parallel_loop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_target_teams_loop:
    return llvm::omp::Category::Executable;
  case llvm::omp::OMPD_teams_loop:
    return llvm::omp::Category::Executable;
  } // switch (Dir)
  llvm_unreachable("Unexpected directive");
}

static_assert(sizeof(llvm::omp::Directive) == sizeof(int));
[[maybe_unused]] static const llvm::omp::Directive LeafConstructTable[][8] = {
    {llvm::omp::OMPD_allocate, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_allocators, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_assume, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_assumes, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_atomic, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_barrier, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_begin_assumes, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_begin_declare_target, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_begin_declare_variant, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_cancel, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_cancellation_point, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_critical, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_declare_mapper, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_declare_reduction, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_declare_simd, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_declare_target, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_declare_variant, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_depobj, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_distribute, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_do, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_error, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_flush, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_for, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_interchange, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_master, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_metadirective, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_nothing, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_ordered, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_requires, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_reverse, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_scan, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_scope, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_section, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_sections, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_single, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_stripe, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_data, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_enter_data, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_exit_data, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_update, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_task, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_taskgroup, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_taskloop, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_taskwait, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_taskyield, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_threadprivate, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_tile, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_unknown, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_unroll, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_workshare, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_dispatch, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_interop, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_loop, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_masked, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_distribute_parallel_do, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_distribute_parallel_do_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_distribute_parallel_for, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_distribute_parallel_for_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_distribute_simd, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_distribute, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_do_simd, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_do, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_for_simd, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_for, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_master_taskloop, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_master, llvm::omp::OMPD_taskloop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_master_taskloop_simd, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_master, llvm::omp::OMPD_taskloop, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_do, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_do_simd, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_for, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_for_simd, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_master, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_master, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_master_taskloop, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_parallel, llvm::omp::OMPD_master, llvm::omp::OMPD_taskloop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_master_taskloop_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_parallel, llvm::omp::OMPD_master, llvm::omp::OMPD_taskloop, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_sections, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_sections, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_workshare, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_workshare, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_loop, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_loop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_masked, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_parallel, llvm::omp::OMPD_masked, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_masked_taskloop, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_parallel, llvm::omp::OMPD_masked, llvm::omp::OMPD_taskloop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_parallel_masked_taskloop_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_parallel, llvm::omp::OMPD_masked, llvm::omp::OMPD_taskloop, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_parallel, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_target, llvm::omp::OMPD_parallel, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_parallel_do, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_target, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_parallel_do_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_target, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_parallel_for, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_target, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_parallel_for_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_target, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_parallel_loop, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_target, llvm::omp::OMPD_parallel, llvm::omp::OMPD_loop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_simd, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_target, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_teams, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_teams_distribute, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_teams_distribute_parallel_do, static_cast<llvm::omp::Directive>(5), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_teams_distribute_parallel_do_simd, static_cast<llvm::omp::Directive>(6), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, llvm::omp::OMPD_simd,},
    {llvm::omp::OMPD_target_teams_distribute_parallel_for, static_cast<llvm::omp::Directive>(5), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_teams_distribute_parallel_for_simd, static_cast<llvm::omp::Directive>(6), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, llvm::omp::OMPD_simd,},
    {llvm::omp::OMPD_target_teams_distribute_simd, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_teams_loop, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_target, llvm::omp::OMPD_teams, llvm::omp::OMPD_loop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_target_loop, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_target, llvm::omp::OMPD_loop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_taskloop_simd, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_taskloop, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_distribute, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_distribute_parallel_do, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_distribute_parallel_do_simd, static_cast<llvm::omp::Directive>(5), llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_do, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_distribute_parallel_for, static_cast<llvm::omp::Directive>(4), llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_distribute_parallel_for_simd, static_cast<llvm::omp::Directive>(5), llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_parallel, llvm::omp::OMPD_for, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_distribute_simd, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_teams, llvm::omp::OMPD_distribute, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_teams_loop, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_teams, llvm::omp::OMPD_loop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_masked_taskloop, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_masked, llvm::omp::OMPD_taskloop, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_masked_taskloop_simd, static_cast<llvm::omp::Directive>(3), llvm::omp::OMPD_masked, llvm::omp::OMPD_taskloop, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_assumes, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_declare_target, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_declare_variant, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_do, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_scope, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_sections, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_single, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_workshare, static_cast<llvm::omp::Directive>(0), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
    {llvm::omp::OMPD_end_do_simd, static_cast<llvm::omp::Directive>(2), llvm::omp::OMPD_do, llvm::omp::OMPD_simd, static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1), static_cast<llvm::omp::Directive>(-1),},
};

[[maybe_unused]] static auto LeafConstructTableEndDirective = LeafConstructTable + 106;

[[maybe_unused]] static const int LeafConstructTableOrdering[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 58, 59, 60, 61, 62, 19, 63, 106, 107, 108, 109, 114, 110, 111, 112, 113, 20, 21, 22, 64, 23, 104, 105, 24, 65, 66, 25, 26, 27, 28, 67, 68, 69, 70, 77, 78, 79, 71, 72, 73, 74, 75, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 42, 43, 44, 45, 96, 46, 47, 48, 97, 98, 99, 100, 101, 102, 49, 50, 51, 52, 53, 54, 55, 56, 57, 76, 95, 85, 94, 103,
};

#endif // GEN_DIRECTIVES_IMPL

