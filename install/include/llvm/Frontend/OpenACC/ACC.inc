#ifdef GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#undef GEN_FLANG_DIRECTIVE_CLAUSE_SETS

namespace llvm {
namespace acc {

  // Sets for atomic

  static AccClauseSet allowedClauses_ACCD_atomic {
  };

  static AccClauseSet allowedOnceClauses_ACCD_atomic {
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_atomic {
  };

  static AccClauseSet requiredClauses_ACCD_atomic {
  };

  // Sets for cache

  static AccClauseSet allowedClauses_ACCD_cache {
  };

  static AccClauseSet allowedOnceClauses_ACCD_cache {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_cache {
  };

  static AccClauseSet requiredClauses_ACCD_cache {
  };

  // Sets for data

  static AccClauseSet allowedClauses_ACCD_data {
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_data {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_default,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_data {
  };

  static AccClauseSet requiredClauses_ACCD_data {
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_present,
  };

  // Sets for declare

  static AccClauseSet allowedClauses_ACCD_declare {
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_device_resident,
    llvm::acc::Clause::ACCC_link,
  };

  static AccClauseSet allowedOnceClauses_ACCD_declare {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_declare {
  };

  static AccClauseSet requiredClauses_ACCD_declare {
  };

  // Sets for enter data

  static AccClauseSet allowedClauses_ACCD_enter_data {
    llvm::acc::Clause::ACCC_wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_enter_data {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_enter_data {
  };

  static AccClauseSet requiredClauses_ACCD_enter_data {
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_copyin,
  };

  // Sets for exit data

  static AccClauseSet allowedClauses_ACCD_exit_data {
    llvm::acc::Clause::ACCC_finalize,
    llvm::acc::Clause::ACCC_wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_exit_data {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_exit_data {
  };

  static AccClauseSet requiredClauses_ACCD_exit_data {
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_delete,
    llvm::acc::Clause::ACCC_detach,
  };

  // Sets for host_data

  static AccClauseSet allowedClauses_ACCD_host_data {
    llvm::acc::Clause::ACCC_if_present,
  };

  static AccClauseSet allowedOnceClauses_ACCD_host_data {
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_host_data {
  };

  static AccClauseSet requiredClauses_ACCD_host_data {
    llvm::acc::Clause::ACCC_use_device,
  };

  // Sets for init

  static AccClauseSet allowedClauses_ACCD_init {
    llvm::acc::Clause::ACCC_device_type,
  };

  static AccClauseSet allowedOnceClauses_ACCD_init {
    llvm::acc::Clause::ACCC_device_num,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_init {
  };

  static AccClauseSet requiredClauses_ACCD_init {
  };

  // Sets for kernels

  static AccClauseSet allowedClauses_ACCD_kernels {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_num_gangs,
    llvm::acc::Clause::ACCC_num_workers,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_vector_length,
    llvm::acc::Clause::ACCC_wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_kernels {
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_kernels {
  };

  static AccClauseSet requiredClauses_ACCD_kernels {
  };

  // Sets for kernels loop

  static AccClauseSet allowedClauses_ACCD_kernels_loop {
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_collapse,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_num_gangs,
    llvm::acc::Clause::ACCC_num_workers,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_private,
    llvm::acc::Clause::ACCC_reduction,
    llvm::acc::Clause::ACCC_shortloop,
    llvm::acc::Clause::ACCC_tile,
    llvm::acc::Clause::ACCC_vector,
    llvm::acc::Clause::ACCC_vector_length,
    llvm::acc::Clause::ACCC_wait,
    llvm::acc::Clause::ACCC_worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_kernels_loop {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_kernels_loop {
    llvm::acc::Clause::ACCC_auto,
    llvm::acc::Clause::ACCC_independent,
    llvm::acc::Clause::ACCC_seq,
  };

  static AccClauseSet requiredClauses_ACCD_kernels_loop {
  };

  // Sets for loop

  static AccClauseSet allowedClauses_ACCD_loop {
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_private,
    llvm::acc::Clause::ACCC_reduction,
    llvm::acc::Clause::ACCC_collapse,
    llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_shortloop,
    llvm::acc::Clause::ACCC_tile,
    llvm::acc::Clause::ACCC_vector,
    llvm::acc::Clause::ACCC_worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_loop {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_loop {
    llvm::acc::Clause::ACCC_auto,
    llvm::acc::Clause::ACCC_independent,
    llvm::acc::Clause::ACCC_seq,
  };

  static AccClauseSet requiredClauses_ACCD_loop {
  };

  // Sets for parallel

  static AccClauseSet allowedClauses_ACCD_parallel {
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_num_gangs,
    llvm::acc::Clause::ACCC_num_workers,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_private,
    llvm::acc::Clause::ACCC_firstprivate,
    llvm::acc::Clause::ACCC_reduction,
    llvm::acc::Clause::ACCC_wait,
    llvm::acc::Clause::ACCC_vector_length,
  };

  static AccClauseSet allowedOnceClauses_ACCD_parallel {
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_parallel {
  };

  static AccClauseSet requiredClauses_ACCD_parallel {
  };

  // Sets for parallel loop

  static AccClauseSet allowedClauses_ACCD_parallel_loop {
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_collapse,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_firstprivate,
    llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_num_gangs,
    llvm::acc::Clause::ACCC_num_workers,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_private,
    llvm::acc::Clause::ACCC_reduction,
    llvm::acc::Clause::ACCC_shortloop,
    llvm::acc::Clause::ACCC_tile,
    llvm::acc::Clause::ACCC_vector,
    llvm::acc::Clause::ACCC_vector_length,
    llvm::acc::Clause::ACCC_wait,
    llvm::acc::Clause::ACCC_worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_parallel_loop {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_parallel_loop {
    llvm::acc::Clause::ACCC_auto,
    llvm::acc::Clause::ACCC_independent,
    llvm::acc::Clause::ACCC_seq,
  };

  static AccClauseSet requiredClauses_ACCD_parallel_loop {
  };

  // Sets for routine

  static AccClauseSet allowedClauses_ACCD_routine {
    llvm::acc::Clause::ACCC_bind,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_seq,
    llvm::acc::Clause::ACCC_vector,
    llvm::acc::Clause::ACCC_worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_routine {
    llvm::acc::Clause::ACCC_nohost,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_routine {
  };

  static AccClauseSet requiredClauses_ACCD_routine {
  };

  // Sets for serial

  static AccClauseSet allowedClauses_ACCD_serial {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_private,
    llvm::acc::Clause::ACCC_firstprivate,
    llvm::acc::Clause::ACCC_reduction,
    llvm::acc::Clause::ACCC_wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_serial {
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_serial {
  };

  static AccClauseSet requiredClauses_ACCD_serial {
  };

  // Sets for serial loop

  static AccClauseSet allowedClauses_ACCD_serial_loop {
    llvm::acc::Clause::ACCC_attach,
    llvm::acc::Clause::ACCC_collapse,
    llvm::acc::Clause::ACCC_copy,
    llvm::acc::Clause::ACCC_copyin,
    llvm::acc::Clause::ACCC_copyout,
    llvm::acc::Clause::ACCC_create,
    llvm::acc::Clause::ACCC_deviceptr,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_firstprivate,
    llvm::acc::Clause::ACCC_gang,
    llvm::acc::Clause::ACCC_no_create,
    llvm::acc::Clause::ACCC_present,
    llvm::acc::Clause::ACCC_private,
    llvm::acc::Clause::ACCC_reduction,
    llvm::acc::Clause::ACCC_shortloop,
    llvm::acc::Clause::ACCC_tile,
    llvm::acc::Clause::ACCC_vector,
    llvm::acc::Clause::ACCC_wait,
    llvm::acc::Clause::ACCC_worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_serial_loop {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_default,
    llvm::acc::Clause::ACCC_if,
    llvm::acc::Clause::ACCC_self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_serial_loop {
    llvm::acc::Clause::ACCC_auto,
    llvm::acc::Clause::ACCC_independent,
    llvm::acc::Clause::ACCC_seq,
  };

  static AccClauseSet requiredClauses_ACCD_serial_loop {
  };

  // Sets for set

  static AccClauseSet allowedClauses_ACCD_set {
  };

  static AccClauseSet allowedOnceClauses_ACCD_set {
    llvm::acc::Clause::ACCC_default_async,
    llvm::acc::Clause::ACCC_device_num,
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_set {
  };

  static AccClauseSet requiredClauses_ACCD_set {
    llvm::acc::Clause::ACCC_default_async,
    llvm::acc::Clause::ACCC_device_num,
    llvm::acc::Clause::ACCC_device_type,
  };

  // Sets for shutdown

  static AccClauseSet allowedClauses_ACCD_shutdown {
    llvm::acc::Clause::ACCC_device_type,
  };

  static AccClauseSet allowedOnceClauses_ACCD_shutdown {
    llvm::acc::Clause::ACCC_device_num,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_shutdown {
  };

  static AccClauseSet requiredClauses_ACCD_shutdown {
  };

  // Sets for unknown

  static AccClauseSet allowedClauses_ACCD_unknown {
  };

  static AccClauseSet allowedOnceClauses_ACCD_unknown {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_unknown {
  };

  static AccClauseSet requiredClauses_ACCD_unknown {
  };

  // Sets for update

  static AccClauseSet allowedClauses_ACCD_update {
    llvm::acc::Clause::ACCC_device_type,
    llvm::acc::Clause::ACCC_if_present,
    llvm::acc::Clause::ACCC_wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_update {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_update {
  };

  static AccClauseSet requiredClauses_ACCD_update {
    llvm::acc::Clause::ACCC_device,
    llvm::acc::Clause::ACCC_host,
    llvm::acc::Clause::ACCC_self,
  };

  // Sets for wait

  static AccClauseSet allowedClauses_ACCD_wait {
  };

  static AccClauseSet allowedOnceClauses_ACCD_wait {
    llvm::acc::Clause::ACCC_async,
    llvm::acc::Clause::ACCC_if,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_wait {
  };

  static AccClauseSet requiredClauses_ACCD_wait {
  };
} // namespace acc
} // namespace llvm

#endif // GEN_FLANG_DIRECTIVE_CLAUSE_SETS

#ifdef GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#undef GEN_FLANG_DIRECTIVE_CLAUSE_MAP

{
  {llvm::acc::Directive::ACCD_atomic,
    {
      llvm::acc::allowedClauses_ACCD_atomic,
      llvm::acc::allowedOnceClauses_ACCD_atomic,
      llvm::acc::allowedExclusiveClauses_ACCD_atomic,
      llvm::acc::requiredClauses_ACCD_atomic,
    }
  },
  {llvm::acc::Directive::ACCD_cache,
    {
      llvm::acc::allowedClauses_ACCD_cache,
      llvm::acc::allowedOnceClauses_ACCD_cache,
      llvm::acc::allowedExclusiveClauses_ACCD_cache,
      llvm::acc::requiredClauses_ACCD_cache,
    }
  },
  {llvm::acc::Directive::ACCD_data,
    {
      llvm::acc::allowedClauses_ACCD_data,
      llvm::acc::allowedOnceClauses_ACCD_data,
      llvm::acc::allowedExclusiveClauses_ACCD_data,
      llvm::acc::requiredClauses_ACCD_data,
    }
  },
  {llvm::acc::Directive::ACCD_declare,
    {
      llvm::acc::allowedClauses_ACCD_declare,
      llvm::acc::allowedOnceClauses_ACCD_declare,
      llvm::acc::allowedExclusiveClauses_ACCD_declare,
      llvm::acc::requiredClauses_ACCD_declare,
    }
  },
  {llvm::acc::Directive::ACCD_enter_data,
    {
      llvm::acc::allowedClauses_ACCD_enter_data,
      llvm::acc::allowedOnceClauses_ACCD_enter_data,
      llvm::acc::allowedExclusiveClauses_ACCD_enter_data,
      llvm::acc::requiredClauses_ACCD_enter_data,
    }
  },
  {llvm::acc::Directive::ACCD_exit_data,
    {
      llvm::acc::allowedClauses_ACCD_exit_data,
      llvm::acc::allowedOnceClauses_ACCD_exit_data,
      llvm::acc::allowedExclusiveClauses_ACCD_exit_data,
      llvm::acc::requiredClauses_ACCD_exit_data,
    }
  },
  {llvm::acc::Directive::ACCD_host_data,
    {
      llvm::acc::allowedClauses_ACCD_host_data,
      llvm::acc::allowedOnceClauses_ACCD_host_data,
      llvm::acc::allowedExclusiveClauses_ACCD_host_data,
      llvm::acc::requiredClauses_ACCD_host_data,
    }
  },
  {llvm::acc::Directive::ACCD_init,
    {
      llvm::acc::allowedClauses_ACCD_init,
      llvm::acc::allowedOnceClauses_ACCD_init,
      llvm::acc::allowedExclusiveClauses_ACCD_init,
      llvm::acc::requiredClauses_ACCD_init,
    }
  },
  {llvm::acc::Directive::ACCD_kernels,
    {
      llvm::acc::allowedClauses_ACCD_kernels,
      llvm::acc::allowedOnceClauses_ACCD_kernels,
      llvm::acc::allowedExclusiveClauses_ACCD_kernels,
      llvm::acc::requiredClauses_ACCD_kernels,
    }
  },
  {llvm::acc::Directive::ACCD_kernels_loop,
    {
      llvm::acc::allowedClauses_ACCD_kernels_loop,
      llvm::acc::allowedOnceClauses_ACCD_kernels_loop,
      llvm::acc::allowedExclusiveClauses_ACCD_kernels_loop,
      llvm::acc::requiredClauses_ACCD_kernels_loop,
    }
  },
  {llvm::acc::Directive::ACCD_loop,
    {
      llvm::acc::allowedClauses_ACCD_loop,
      llvm::acc::allowedOnceClauses_ACCD_loop,
      llvm::acc::allowedExclusiveClauses_ACCD_loop,
      llvm::acc::requiredClauses_ACCD_loop,
    }
  },
  {llvm::acc::Directive::ACCD_parallel,
    {
      llvm::acc::allowedClauses_ACCD_parallel,
      llvm::acc::allowedOnceClauses_ACCD_parallel,
      llvm::acc::allowedExclusiveClauses_ACCD_parallel,
      llvm::acc::requiredClauses_ACCD_parallel,
    }
  },
  {llvm::acc::Directive::ACCD_parallel_loop,
    {
      llvm::acc::allowedClauses_ACCD_parallel_loop,
      llvm::acc::allowedOnceClauses_ACCD_parallel_loop,
      llvm::acc::allowedExclusiveClauses_ACCD_parallel_loop,
      llvm::acc::requiredClauses_ACCD_parallel_loop,
    }
  },
  {llvm::acc::Directive::ACCD_routine,
    {
      llvm::acc::allowedClauses_ACCD_routine,
      llvm::acc::allowedOnceClauses_ACCD_routine,
      llvm::acc::allowedExclusiveClauses_ACCD_routine,
      llvm::acc::requiredClauses_ACCD_routine,
    }
  },
  {llvm::acc::Directive::ACCD_serial,
    {
      llvm::acc::allowedClauses_ACCD_serial,
      llvm::acc::allowedOnceClauses_ACCD_serial,
      llvm::acc::allowedExclusiveClauses_ACCD_serial,
      llvm::acc::requiredClauses_ACCD_serial,
    }
  },
  {llvm::acc::Directive::ACCD_serial_loop,
    {
      llvm::acc::allowedClauses_ACCD_serial_loop,
      llvm::acc::allowedOnceClauses_ACCD_serial_loop,
      llvm::acc::allowedExclusiveClauses_ACCD_serial_loop,
      llvm::acc::requiredClauses_ACCD_serial_loop,
    }
  },
  {llvm::acc::Directive::ACCD_set,
    {
      llvm::acc::allowedClauses_ACCD_set,
      llvm::acc::allowedOnceClauses_ACCD_set,
      llvm::acc::allowedExclusiveClauses_ACCD_set,
      llvm::acc::requiredClauses_ACCD_set,
    }
  },
  {llvm::acc::Directive::ACCD_shutdown,
    {
      llvm::acc::allowedClauses_ACCD_shutdown,
      llvm::acc::allowedOnceClauses_ACCD_shutdown,
      llvm::acc::allowedExclusiveClauses_ACCD_shutdown,
      llvm::acc::requiredClauses_ACCD_shutdown,
    }
  },
  {llvm::acc::Directive::ACCD_unknown,
    {
      llvm::acc::allowedClauses_ACCD_unknown,
      llvm::acc::allowedOnceClauses_ACCD_unknown,
      llvm::acc::allowedExclusiveClauses_ACCD_unknown,
      llvm::acc::requiredClauses_ACCD_unknown,
    }
  },
  {llvm::acc::Directive::ACCD_update,
    {
      llvm::acc::allowedClauses_ACCD_update,
      llvm::acc::allowedOnceClauses_ACCD_update,
      llvm::acc::allowedExclusiveClauses_ACCD_update,
      llvm::acc::requiredClauses_ACCD_update,
    }
  },
  {llvm::acc::Directive::ACCD_wait,
    {
      llvm::acc::allowedClauses_ACCD_wait,
      llvm::acc::allowedOnceClauses_ACCD_wait,
      llvm::acc::allowedExclusiveClauses_ACCD_wait,
      llvm::acc::requiredClauses_ACCD_wait,
    }
  },
}

#endif // GEN_FLANG_DIRECTIVE_CLAUSE_MAP

#ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES
#undef GEN_FLANG_CLAUSE_PARSER_CLASSES

WRAPPER_CLASS(Async, std::optional<ScalarIntExpr>);
WRAPPER_CLASS(Attach, AccObjectList);
EMPTY_CLASS(Auto);
WRAPPER_CLASS(Bind, AccBindClause);
EMPTY_CLASS(Capture);
WRAPPER_CLASS(Collapse, AccCollapseArg);
WRAPPER_CLASS(Copy, AccObjectList);
WRAPPER_CLASS(Copyin, AccObjectListWithModifier);
WRAPPER_CLASS(Copyout, AccObjectListWithModifier);
WRAPPER_CLASS(Create, AccObjectListWithModifier);
WRAPPER_CLASS(Default, AccDefaultClause);
WRAPPER_CLASS(DefaultAsync, ScalarIntExpr);
WRAPPER_CLASS(Delete, AccObjectList);
WRAPPER_CLASS(Detach, AccObjectList);
WRAPPER_CLASS(Device, AccObjectList);
WRAPPER_CLASS(DeviceNum, ScalarIntExpr);
WRAPPER_CLASS(Deviceptr, AccObjectList);
WRAPPER_CLASS(DeviceResident, AccObjectList);
WRAPPER_CLASS(DeviceType, AccDeviceTypeExprList);
EMPTY_CLASS(Finalize);
WRAPPER_CLASS(Firstprivate, AccObjectList);
WRAPPER_CLASS(Gang, std::optional<AccGangArgList>);
WRAPPER_CLASS(Host, AccObjectList);
WRAPPER_CLASS(If, ScalarExpr);
EMPTY_CLASS(IfPresent);
EMPTY_CLASS(Independent);
WRAPPER_CLASS(Link, AccObjectList);
WRAPPER_CLASS(NoCreate, AccObjectList);
EMPTY_CLASS(Nohost);
WRAPPER_CLASS(NumGangs, std::list<ScalarIntExpr>);
WRAPPER_CLASS(NumWorkers, ScalarIntExpr);
WRAPPER_CLASS(Present, AccObjectList);
WRAPPER_CLASS(Private, AccObjectList);
EMPTY_CLASS(Read);
WRAPPER_CLASS(Reduction, AccObjectListWithReduction);
WRAPPER_CLASS(Self, std::optional<AccSelfClause>);
EMPTY_CLASS(Seq);
EMPTY_CLASS(Shortloop);
WRAPPER_CLASS(Tile, AccTileExprList);
EMPTY_CLASS(Unknown);
WRAPPER_CLASS(UseDevice, AccObjectList);
WRAPPER_CLASS(Vector, std::optional<ScalarIntExpr>);
WRAPPER_CLASS(VectorLength, ScalarIntExpr);
WRAPPER_CLASS(Wait, std::optional<AccWaitArgument>);
WRAPPER_CLASS(Worker, std::optional<ScalarIntExpr>);
EMPTY_CLASS(Write);

#endif // GEN_FLANG_CLAUSE_PARSER_CLASSES

#ifdef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST
#undef GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST

Async
, Attach
, Auto
, Bind
, Capture
, Collapse
, Copy
, Copyin
, Copyout
, Create
, Default
, DefaultAsync
, Delete
, Detach
, Device
, DeviceNum
, Deviceptr
, DeviceResident
, DeviceType
, Finalize
, Firstprivate
, Gang
, Host
, If
, IfPresent
, Independent
, Link
, NoCreate
, Nohost
, NumGangs
, NumWorkers
, Present
, Private
, Read
, Reduction
, Self
, Seq
, Shortloop
, Tile
, Unknown
, UseDevice
, Vector
, VectorLength
, Wait
, Worker
, Write

#endif // GEN_FLANG_CLAUSE_PARSER_CLASSES_LIST

#ifdef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES
#undef GEN_FLANG_DUMP_PARSE_TREE_CLAUSES

NODE(AccClause, Async)
NODE(AccClause, Attach)
NODE(AccClause, Auto)
NODE(AccClause, Bind)
NODE(AccClause, Capture)
NODE(AccClause, Collapse)
NODE(AccClause, Copy)
NODE(AccClause, Copyin)
NODE(AccClause, Copyout)
NODE(AccClause, Create)
NODE(AccClause, Default)
NODE(AccClause, DefaultAsync)
NODE(AccClause, Delete)
NODE(AccClause, Detach)
NODE(AccClause, Device)
NODE(AccClause, DeviceNum)
NODE(AccClause, Deviceptr)
NODE(AccClause, DeviceResident)
NODE(AccClause, DeviceType)
NODE(AccClause, Finalize)
NODE(AccClause, Firstprivate)
NODE(AccClause, Gang)
NODE(AccClause, Host)
NODE(AccClause, If)
NODE(AccClause, IfPresent)
NODE(AccClause, Independent)
NODE(AccClause, Link)
NODE(AccClause, NoCreate)
NODE(AccClause, Nohost)
NODE(AccClause, NumGangs)
NODE(AccClause, NumWorkers)
NODE(AccClause, Present)
NODE(AccClause, Private)
NODE(AccClause, Read)
NODE(AccClause, Reduction)
NODE(AccClause, Self)
NODE(AccClause, Seq)
NODE(AccClause, Shortloop)
NODE(AccClause, Tile)
NODE(AccClause, Unknown)
NODE(AccClause, UseDevice)
NODE(AccClause, Vector)
NODE(AccClause, VectorLength)
NODE(AccClause, Wait)
NODE(AccClause, Worker)
NODE(AccClause, Write)

#endif // GEN_FLANG_DUMP_PARSE_TREE_CLAUSES

#ifdef GEN_FLANG_CLAUSE_UNPARSE
#undef GEN_FLANG_CLAUSE_UNPARSE

void Unparse(const AccClause::Async &x) {
  Word("ASYNC");
  Walk("(", x.v, ")");
}
void Unparse(const AccClause::Attach &x) {
  Word("ATTACH");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::Auto &) { Word("AUTO"); }
void Unparse(const AccClause::Bind &x) {
  Word("BIND");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::Capture &) { Word("CAPTURE"); }
void Unparse(const AccClause::Collapse &x) {
  Word("COLLAPSE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Copy &x) {
  Word("COPY");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Copyin &x) {
  Word("COPYIN");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Copyout &x) {
  Word("COPYOUT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Create &x) {
  Word("CREATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Default &x) {
  Word("DEFAULT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::DefaultAsync &x) {
  Word("DEFAULT_ASYNC");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Delete &x) {
  Word("DELETE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Detach &x) {
  Word("DETACH");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Device &x) {
  Word("DEVICE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::DeviceNum &x) {
  Word("DEVICE_NUM");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Deviceptr &x) {
  Word("DEVICEPTR");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::DeviceResident &x) {
  Word("DEVICE_RESIDENT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::DeviceType &x) {
  Word("DEVICE_TYPE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::Finalize &) { Word("FINALIZE"); }
void Unparse(const AccClause::Firstprivate &x) {
  Word("FIRSTPRIVATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Gang &x) {
  Word("GANG");
  Walk("(", x.v, ")");
}
void Unparse(const AccClause::Host &x) {
  Word("HOST");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::If &x) {
  Word("IF");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::IfPresent &) { Word("IF_PRESENT"); }
void Before(const AccClause::Independent &) { Word("INDEPENDENT"); }
void Unparse(const AccClause::Link &x) {
  Word("LINK");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::NoCreate &x) {
  Word("NO_CREATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::Nohost &) { Word("NOHOST"); }
void Unparse(const AccClause::NumGangs &x) {
  Word("NUM_GANGS");
  Put("(");
  Walk(x.v, ",");
  Put(")");
}
void Unparse(const AccClause::NumWorkers &x) {
  Word("NUM_WORKERS");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Present &x) {
  Word("PRESENT");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Private &x) {
  Word("PRIVATE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::Read &) { Word("READ"); }
void Unparse(const AccClause::Reduction &x) {
  Word("REDUCTION");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Self &x) {
  Word("SELF");
  Walk("(", x.v, ")");
}
void Before(const AccClause::Seq &) { Word("SEQ"); }
void Before(const AccClause::Shortloop &) { Word("SHORTLOOP"); }
void Unparse(const AccClause::Tile &x) {
  Word("TILE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Before(const AccClause::Unknown &) { Word("UNKNOWN"); }
void Unparse(const AccClause::UseDevice &x) {
  Word("USE_DEVICE");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Vector &x) {
  Word("VECTOR");
  Walk("(", x.v, ")");
}
void Unparse(const AccClause::VectorLength &x) {
  Word("VECTOR_LENGTH");
  Put("(");
  Walk(x.v);
  Put(")");
}
void Unparse(const AccClause::Wait &x) {
  Word("WAIT");
  Walk("(", x.v, ")");
}
void Unparse(const AccClause::Worker &x) {
  Word("WORKER");
  Walk("(", x.v, ")");
}
void Before(const AccClause::Write &) { Word("WRITE"); }

#endif // GEN_FLANG_CLAUSE_UNPARSE

#ifdef GEN_FLANG_CLAUSE_CHECK_ENTER
#undef GEN_FLANG_CLAUSE_CHECK_ENTER

void Enter(const parser::AccClause::Async &);
void Enter(const parser::AccClause::Attach &);
void Enter(const parser::AccClause::Auto &);
void Enter(const parser::AccClause::Bind &);
void Enter(const parser::AccClause::Capture &);
void Enter(const parser::AccClause::Collapse &);
void Enter(const parser::AccClause::Copy &);
void Enter(const parser::AccClause::Copyin &);
void Enter(const parser::AccClause::Copyout &);
void Enter(const parser::AccClause::Create &);
void Enter(const parser::AccClause::Default &);
void Enter(const parser::AccClause::DefaultAsync &);
void Enter(const parser::AccClause::Delete &);
void Enter(const parser::AccClause::Detach &);
void Enter(const parser::AccClause::Device &);
void Enter(const parser::AccClause::DeviceNum &);
void Enter(const parser::AccClause::Deviceptr &);
void Enter(const parser::AccClause::DeviceResident &);
void Enter(const parser::AccClause::DeviceType &);
void Enter(const parser::AccClause::Finalize &);
void Enter(const parser::AccClause::Firstprivate &);
void Enter(const parser::AccClause::Gang &);
void Enter(const parser::AccClause::Host &);
void Enter(const parser::AccClause::If &);
void Enter(const parser::AccClause::IfPresent &);
void Enter(const parser::AccClause::Independent &);
void Enter(const parser::AccClause::Link &);
void Enter(const parser::AccClause::NoCreate &);
void Enter(const parser::AccClause::Nohost &);
void Enter(const parser::AccClause::NumGangs &);
void Enter(const parser::AccClause::NumWorkers &);
void Enter(const parser::AccClause::Present &);
void Enter(const parser::AccClause::Private &);
void Enter(const parser::AccClause::Read &);
void Enter(const parser::AccClause::Reduction &);
void Enter(const parser::AccClause::Self &);
void Enter(const parser::AccClause::Seq &);
void Enter(const parser::AccClause::Shortloop &);
void Enter(const parser::AccClause::Tile &);
void Enter(const parser::AccClause::Unknown &);
void Enter(const parser::AccClause::UseDevice &);
void Enter(const parser::AccClause::Vector &);
void Enter(const parser::AccClause::VectorLength &);
void Enter(const parser::AccClause::Wait &);
void Enter(const parser::AccClause::Worker &);
void Enter(const parser::AccClause::Write &);

#endif // GEN_FLANG_CLAUSE_CHECK_ENTER

#ifdef GEN_FLANG_CLAUSE_PARSER_KIND_MAP
#undef GEN_FLANG_CLAUSE_PARSER_KIND_MAP

if constexpr (std::is_same_v<A, parser::AccClause::Async>)
  return llvm::acc::Clause::ACCC_async;
if constexpr (std::is_same_v<A, parser::AccClause::Attach>)
  return llvm::acc::Clause::ACCC_attach;
if constexpr (std::is_same_v<A, parser::AccClause::Auto>)
  return llvm::acc::Clause::ACCC_auto;
if constexpr (std::is_same_v<A, parser::AccClause::Bind>)
  return llvm::acc::Clause::ACCC_bind;
if constexpr (std::is_same_v<A, parser::AccClause::Capture>)
  return llvm::acc::Clause::ACCC_capture;
if constexpr (std::is_same_v<A, parser::AccClause::Collapse>)
  return llvm::acc::Clause::ACCC_collapse;
if constexpr (std::is_same_v<A, parser::AccClause::Copy>)
  return llvm::acc::Clause::ACCC_copy;
if constexpr (std::is_same_v<A, parser::AccClause::Copyin>)
  return llvm::acc::Clause::ACCC_copyin;
if constexpr (std::is_same_v<A, parser::AccClause::Copyout>)
  return llvm::acc::Clause::ACCC_copyout;
if constexpr (std::is_same_v<A, parser::AccClause::Create>)
  return llvm::acc::Clause::ACCC_create;
if constexpr (std::is_same_v<A, parser::AccClause::Default>)
  return llvm::acc::Clause::ACCC_default;
if constexpr (std::is_same_v<A, parser::AccClause::DefaultAsync>)
  return llvm::acc::Clause::ACCC_default_async;
if constexpr (std::is_same_v<A, parser::AccClause::Delete>)
  return llvm::acc::Clause::ACCC_delete;
if constexpr (std::is_same_v<A, parser::AccClause::Detach>)
  return llvm::acc::Clause::ACCC_detach;
if constexpr (std::is_same_v<A, parser::AccClause::Device>)
  return llvm::acc::Clause::ACCC_device;
if constexpr (std::is_same_v<A, parser::AccClause::DeviceNum>)
  return llvm::acc::Clause::ACCC_device_num;
if constexpr (std::is_same_v<A, parser::AccClause::Deviceptr>)
  return llvm::acc::Clause::ACCC_deviceptr;
if constexpr (std::is_same_v<A, parser::AccClause::DeviceResident>)
  return llvm::acc::Clause::ACCC_device_resident;
if constexpr (std::is_same_v<A, parser::AccClause::DeviceType>)
  return llvm::acc::Clause::ACCC_device_type;
if constexpr (std::is_same_v<A, parser::AccClause::Finalize>)
  return llvm::acc::Clause::ACCC_finalize;
if constexpr (std::is_same_v<A, parser::AccClause::Firstprivate>)
  return llvm::acc::Clause::ACCC_firstprivate;
if constexpr (std::is_same_v<A, parser::AccClause::Gang>)
  return llvm::acc::Clause::ACCC_gang;
if constexpr (std::is_same_v<A, parser::AccClause::Host>)
  return llvm::acc::Clause::ACCC_host;
if constexpr (std::is_same_v<A, parser::AccClause::If>)
  return llvm::acc::Clause::ACCC_if;
if constexpr (std::is_same_v<A, parser::AccClause::IfPresent>)
  return llvm::acc::Clause::ACCC_if_present;
if constexpr (std::is_same_v<A, parser::AccClause::Independent>)
  return llvm::acc::Clause::ACCC_independent;
if constexpr (std::is_same_v<A, parser::AccClause::Link>)
  return llvm::acc::Clause::ACCC_link;
if constexpr (std::is_same_v<A, parser::AccClause::NoCreate>)
  return llvm::acc::Clause::ACCC_no_create;
if constexpr (std::is_same_v<A, parser::AccClause::Nohost>)
  return llvm::acc::Clause::ACCC_nohost;
if constexpr (std::is_same_v<A, parser::AccClause::NumGangs>)
  return llvm::acc::Clause::ACCC_num_gangs;
if constexpr (std::is_same_v<A, parser::AccClause::NumWorkers>)
  return llvm::acc::Clause::ACCC_num_workers;
if constexpr (std::is_same_v<A, parser::AccClause::Present>)
  return llvm::acc::Clause::ACCC_present;
if constexpr (std::is_same_v<A, parser::AccClause::Private>)
  return llvm::acc::Clause::ACCC_private;
if constexpr (std::is_same_v<A, parser::AccClause::Read>)
  return llvm::acc::Clause::ACCC_read;
if constexpr (std::is_same_v<A, parser::AccClause::Reduction>)
  return llvm::acc::Clause::ACCC_reduction;
if constexpr (std::is_same_v<A, parser::AccClause::Self>)
  return llvm::acc::Clause::ACCC_self;
if constexpr (std::is_same_v<A, parser::AccClause::Seq>)
  return llvm::acc::Clause::ACCC_seq;
if constexpr (std::is_same_v<A, parser::AccClause::Shortloop>)
  return llvm::acc::Clause::ACCC_shortloop;
if constexpr (std::is_same_v<A, parser::AccClause::Tile>)
  return llvm::acc::Clause::ACCC_tile;
if constexpr (std::is_same_v<A, parser::AccClause::Unknown>)
  return llvm::acc::Clause::ACCC_unknown;
if constexpr (std::is_same_v<A, parser::AccClause::UseDevice>)
  return llvm::acc::Clause::ACCC_use_device;
if constexpr (std::is_same_v<A, parser::AccClause::Vector>)
  return llvm::acc::Clause::ACCC_vector;
if constexpr (std::is_same_v<A, parser::AccClause::VectorLength>)
  return llvm::acc::Clause::ACCC_vector_length;
if constexpr (std::is_same_v<A, parser::AccClause::Wait>)
  return llvm::acc::Clause::ACCC_wait;
if constexpr (std::is_same_v<A, parser::AccClause::Worker>)
  return llvm::acc::Clause::ACCC_worker;
if constexpr (std::is_same_v<A, parser::AccClause::Write>)
  return llvm::acc::Clause::ACCC_write;
llvm_unreachable("Invalid OpenACC Parser clause");

#endif // GEN_FLANG_CLAUSE_PARSER_KIND_MAP

#ifdef GEN_FLANG_CLAUSES_PARSER
#undef GEN_FLANG_CLAUSES_PARSER

TYPE_PARSER(
  "write" >> construct<AccClause>(construct<AccClause::Write>()) ||
  "worker" >> construct<AccClause>(construct<AccClause::Worker>(maybe(parenthesized("num:" >> scalarIntExpr || scalarIntExpr)))) ||
  "wait" >> construct<AccClause>(construct<AccClause::Wait>(maybe(parenthesized(Parser<AccWaitArgument>{})))) ||
  "vector_length" >> construct<AccClause>(construct<AccClause::VectorLength>(parenthesized(scalarIntExpr))) ||
  "vector" >> construct<AccClause>(construct<AccClause::Vector>(maybe(parenthesized("length:" >> scalarIntExpr || scalarIntExpr)))) ||
  "use_device" >> construct<AccClause>(construct<AccClause::UseDevice>(parenthesized(Parser<AccObjectList>{}))) ||
  "unknown" >> construct<AccClause>(construct<AccClause::Unknown>()) ||
  "tile" >> construct<AccClause>(construct<AccClause::Tile>(parenthesized(Parser<AccTileExprList>{}))) ||
  "shortloop" >> construct<AccClause>(construct<AccClause::Shortloop>()) ||
  "seq" >> construct<AccClause>(construct<AccClause::Seq>()) ||
  "self" >> construct<AccClause>(construct<AccClause::Self>(maybe(parenthesized(Parser<AccSelfClause>{})))) ||
  "reduction" >> construct<AccClause>(construct<AccClause::Reduction>(parenthesized(Parser<AccObjectListWithReduction>{}))) ||
  "read" >> construct<AccClause>(construct<AccClause::Read>()) ||
  "private" >> construct<AccClause>(construct<AccClause::Private>(parenthesized(Parser<AccObjectList>{}))) ||
  "present" >> construct<AccClause>(construct<AccClause::Present>(parenthesized(Parser<AccObjectList>{}))) ||
  "num_workers" >> construct<AccClause>(construct<AccClause::NumWorkers>(parenthesized(scalarIntExpr))) ||
  "num_gangs" >> construct<AccClause>(construct<AccClause::NumGangs>(parenthesized(nonemptyList(scalarIntExpr)))) ||
  "nohost" >> construct<AccClause>(construct<AccClause::Nohost>()) ||
  "no_create" >> construct<AccClause>(construct<AccClause::NoCreate>(parenthesized(Parser<AccObjectList>{}))) ||
  "link" >> construct<AccClause>(construct<AccClause::Link>(parenthesized(Parser<AccObjectList>{}))) ||
  "independent" >> construct<AccClause>(construct<AccClause::Independent>()) ||
  "if_present" >> construct<AccClause>(construct<AccClause::IfPresent>()) ||
  "if" >> construct<AccClause>(construct<AccClause::If>(parenthesized(scalarExpr))) ||
  "host" >> construct<AccClause>(construct<AccClause::Host>(parenthesized(Parser<AccObjectList>{}))) ||
  "gang" >> construct<AccClause>(construct<AccClause::Gang>(maybe(parenthesized(Parser<AccGangArgList>{})))) ||
  "firstprivate" >> construct<AccClause>(construct<AccClause::Firstprivate>(parenthesized(Parser<AccObjectList>{}))) ||
  "finalize" >> construct<AccClause>(construct<AccClause::Finalize>()) ||
  "deviceptr" >> construct<AccClause>(construct<AccClause::Deviceptr>(parenthesized(Parser<AccObjectList>{}))) ||
  ("device_type"_tok || "dtype"_tok) >> construct<AccClause>(construct<AccClause::DeviceType>(parenthesized(Parser<AccDeviceTypeExprList>{}))) ||
  "device_resident" >> construct<AccClause>(construct<AccClause::DeviceResident>(parenthesized(Parser<AccObjectList>{}))) ||
  "device_num" >> construct<AccClause>(construct<AccClause::DeviceNum>(parenthesized(scalarIntExpr))) ||
  "device" >> construct<AccClause>(construct<AccClause::Device>(parenthesized(Parser<AccObjectList>{}))) ||
  "detach" >> construct<AccClause>(construct<AccClause::Detach>(parenthesized(Parser<AccObjectList>{}))) ||
  "delete" >> construct<AccClause>(construct<AccClause::Delete>(parenthesized(Parser<AccObjectList>{}))) ||
  "default_async" >> construct<AccClause>(construct<AccClause::DefaultAsync>(parenthesized(scalarIntExpr))) ||
  "default" >> construct<AccClause>(construct<AccClause::Default>(parenthesized(Parser<AccDefaultClause>{}))) ||
  ("create"_tok || "present_or_create"_tok || "pcreate"_tok) >> construct<AccClause>(construct<AccClause::Create>(parenthesized(Parser<AccObjectListWithModifier>{}))) ||
  ("copyout"_tok || "present_or_copyout"_tok || "pcopyout"_tok) >> construct<AccClause>(construct<AccClause::Copyout>(parenthesized(Parser<AccObjectListWithModifier>{}))) ||
  ("copyin"_tok || "present_or_copyin"_tok || "pcopyin"_tok) >> construct<AccClause>(construct<AccClause::Copyin>(parenthesized(Parser<AccObjectListWithModifier>{}))) ||
  ("copy"_tok || "present_or_copy"_tok || "pcopy"_tok) >> construct<AccClause>(construct<AccClause::Copy>(parenthesized(Parser<AccObjectList>{}))) ||
  "collapse" >> construct<AccClause>(construct<AccClause::Collapse>(parenthesized(Parser<AccCollapseArg>{}))) ||
  "capture" >> construct<AccClause>(construct<AccClause::Capture>()) ||
  "bind" >> construct<AccClause>(construct<AccClause::Bind>(parenthesized(Parser<AccBindClause>{}))) ||
  "auto" >> construct<AccClause>(construct<AccClause::Auto>()) ||
  "attach" >> construct<AccClause>(construct<AccClause::Attach>(parenthesized(Parser<AccObjectList>{}))) ||
  "async" >> construct<AccClause>(construct<AccClause::Async>(maybe(parenthesized(scalarIntExpr))))
)

#endif // GEN_FLANG_CLAUSES_PARSER

#ifdef GEN_CLANG_DIRECTIVE_CLAUSE_SETS
#undef GEN_CLANG_DIRECTIVE_CLAUSE_SETS

namespace clang {
namespace acc {

  // Sets for atomic

  static AccClauseSet allowedClauses_ACCD_atomic {
  };

  static AccClauseSet allowedOnceClauses_ACCD_atomic {
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_atomic {
  };

  static AccClauseSet requiredClauses_ACCD_atomic {
  };

  // Sets for cache

  static AccClauseSet allowedClauses_ACCD_cache {
  };

  static AccClauseSet allowedOnceClauses_ACCD_cache {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_cache {
  };

  static AccClauseSet requiredClauses_ACCD_cache {
  };

  // Sets for data

  static AccClauseSet allowedClauses_ACCD_data {
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::Wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_data {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Default,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_data {
  };

  static AccClauseSet requiredClauses_ACCD_data {
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::Present,
  };

  // Sets for declare

  static AccClauseSet allowedClauses_ACCD_declare {
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::DeviceResident,
   clang::OpenACCClauseKind::Link,
  };

  static AccClauseSet allowedOnceClauses_ACCD_declare {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_declare {
  };

  static AccClauseSet requiredClauses_ACCD_declare {
  };

  // Sets for enter data

  static AccClauseSet allowedClauses_ACCD_enter_data {
   clang::OpenACCClauseKind::Wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_enter_data {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_enter_data {
  };

  static AccClauseSet requiredClauses_ACCD_enter_data {
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::CopyIn,
  };

  // Sets for exit data

  static AccClauseSet allowedClauses_ACCD_exit_data {
   clang::OpenACCClauseKind::Finalize,
   clang::OpenACCClauseKind::Wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_exit_data {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_exit_data {
  };

  static AccClauseSet requiredClauses_ACCD_exit_data {
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Delete,
   clang::OpenACCClauseKind::Detach,
  };

  // Sets for host_data

  static AccClauseSet allowedClauses_ACCD_host_data {
   clang::OpenACCClauseKind::IfPresent,
  };

  static AccClauseSet allowedOnceClauses_ACCD_host_data {
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_host_data {
  };

  static AccClauseSet requiredClauses_ACCD_host_data {
   clang::OpenACCClauseKind::UseDevice,
  };

  // Sets for init

  static AccClauseSet allowedClauses_ACCD_init {
   clang::OpenACCClauseKind::DeviceType,
  };

  static AccClauseSet allowedOnceClauses_ACCD_init {
   clang::OpenACCClauseKind::DeviceNum,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_init {
  };

  static AccClauseSet requiredClauses_ACCD_init {
  };

  // Sets for kernels

  static AccClauseSet allowedClauses_ACCD_kernels {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::NumGangs,
   clang::OpenACCClauseKind::NumWorkers,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::VectorLength,
   clang::OpenACCClauseKind::Wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_kernels {
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_kernels {
  };

  static AccClauseSet requiredClauses_ACCD_kernels {
  };

  // Sets for kernels loop

  static AccClauseSet allowedClauses_ACCD_kernels_loop {
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Collapse,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::Gang,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::NumGangs,
   clang::OpenACCClauseKind::NumWorkers,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::Private,
   clang::OpenACCClauseKind::Reduction,
   clang::OpenACCClauseKind::Shortloop,
   clang::OpenACCClauseKind::Tile,
   clang::OpenACCClauseKind::Vector,
   clang::OpenACCClauseKind::VectorLength,
   clang::OpenACCClauseKind::Wait,
   clang::OpenACCClauseKind::Worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_kernels_loop {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_kernels_loop {
   clang::OpenACCClauseKind::Auto,
   clang::OpenACCClauseKind::Independent,
   clang::OpenACCClauseKind::Seq,
  };

  static AccClauseSet requiredClauses_ACCD_kernels_loop {
  };

  // Sets for loop

  static AccClauseSet allowedClauses_ACCD_loop {
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::Private,
   clang::OpenACCClauseKind::Reduction,
   clang::OpenACCClauseKind::Collapse,
   clang::OpenACCClauseKind::Gang,
   clang::OpenACCClauseKind::Shortloop,
   clang::OpenACCClauseKind::Tile,
   clang::OpenACCClauseKind::Vector,
   clang::OpenACCClauseKind::Worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_loop {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_loop {
   clang::OpenACCClauseKind::Auto,
   clang::OpenACCClauseKind::Independent,
   clang::OpenACCClauseKind::Seq,
  };

  static AccClauseSet requiredClauses_ACCD_loop {
  };

  // Sets for parallel

  static AccClauseSet allowedClauses_ACCD_parallel {
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::NumGangs,
   clang::OpenACCClauseKind::NumWorkers,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::Private,
   clang::OpenACCClauseKind::FirstPrivate,
   clang::OpenACCClauseKind::Reduction,
   clang::OpenACCClauseKind::Wait,
   clang::OpenACCClauseKind::VectorLength,
  };

  static AccClauseSet allowedOnceClauses_ACCD_parallel {
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_parallel {
  };

  static AccClauseSet requiredClauses_ACCD_parallel {
  };

  // Sets for parallel loop

  static AccClauseSet allowedClauses_ACCD_parallel_loop {
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Collapse,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::FirstPrivate,
   clang::OpenACCClauseKind::Gang,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::NumGangs,
   clang::OpenACCClauseKind::NumWorkers,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::Private,
   clang::OpenACCClauseKind::Reduction,
   clang::OpenACCClauseKind::Shortloop,
   clang::OpenACCClauseKind::Tile,
   clang::OpenACCClauseKind::Vector,
   clang::OpenACCClauseKind::VectorLength,
   clang::OpenACCClauseKind::Wait,
   clang::OpenACCClauseKind::Worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_parallel_loop {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_parallel_loop {
   clang::OpenACCClauseKind::Auto,
   clang::OpenACCClauseKind::Independent,
   clang::OpenACCClauseKind::Seq,
  };

  static AccClauseSet requiredClauses_ACCD_parallel_loop {
  };

  // Sets for routine

  static AccClauseSet allowedClauses_ACCD_routine {
   clang::OpenACCClauseKind::Bind,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::Gang,
   clang::OpenACCClauseKind::Seq,
   clang::OpenACCClauseKind::Vector,
   clang::OpenACCClauseKind::Worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_routine {
   clang::OpenACCClauseKind::NoHost,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_routine {
  };

  static AccClauseSet requiredClauses_ACCD_routine {
  };

  // Sets for serial

  static AccClauseSet allowedClauses_ACCD_serial {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::Private,
   clang::OpenACCClauseKind::FirstPrivate,
   clang::OpenACCClauseKind::Reduction,
   clang::OpenACCClauseKind::Wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_serial {
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_serial {
  };

  static AccClauseSet requiredClauses_ACCD_serial {
  };

  // Sets for serial loop

  static AccClauseSet allowedClauses_ACCD_serial_loop {
   clang::OpenACCClauseKind::Attach,
   clang::OpenACCClauseKind::Collapse,
   clang::OpenACCClauseKind::Copy,
   clang::OpenACCClauseKind::CopyIn,
   clang::OpenACCClauseKind::CopyOut,
   clang::OpenACCClauseKind::Create,
   clang::OpenACCClauseKind::DevicePtr,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::FirstPrivate,
   clang::OpenACCClauseKind::Gang,
   clang::OpenACCClauseKind::NoCreate,
   clang::OpenACCClauseKind::Present,
   clang::OpenACCClauseKind::Private,
   clang::OpenACCClauseKind::Reduction,
   clang::OpenACCClauseKind::Shortloop,
   clang::OpenACCClauseKind::Tile,
   clang::OpenACCClauseKind::Vector,
   clang::OpenACCClauseKind::Wait,
   clang::OpenACCClauseKind::Worker,
  };

  static AccClauseSet allowedOnceClauses_ACCD_serial_loop {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::Default,
   clang::OpenACCClauseKind::If,
   clang::OpenACCClauseKind::Self,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_serial_loop {
   clang::OpenACCClauseKind::Auto,
   clang::OpenACCClauseKind::Independent,
   clang::OpenACCClauseKind::Seq,
  };

  static AccClauseSet requiredClauses_ACCD_serial_loop {
  };

  // Sets for set

  static AccClauseSet allowedClauses_ACCD_set {
  };

  static AccClauseSet allowedOnceClauses_ACCD_set {
   clang::OpenACCClauseKind::DefaultAsync,
   clang::OpenACCClauseKind::DeviceNum,
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_set {
  };

  static AccClauseSet requiredClauses_ACCD_set {
   clang::OpenACCClauseKind::DefaultAsync,
   clang::OpenACCClauseKind::DeviceNum,
   clang::OpenACCClauseKind::DeviceType,
  };

  // Sets for shutdown

  static AccClauseSet allowedClauses_ACCD_shutdown {
   clang::OpenACCClauseKind::DeviceType,
  };

  static AccClauseSet allowedOnceClauses_ACCD_shutdown {
   clang::OpenACCClauseKind::DeviceNum,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_shutdown {
  };

  static AccClauseSet requiredClauses_ACCD_shutdown {
  };

  // Sets for unknown

  static AccClauseSet allowedClauses_ACCD_unknown {
  };

  static AccClauseSet allowedOnceClauses_ACCD_unknown {
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_unknown {
  };

  static AccClauseSet requiredClauses_ACCD_unknown {
  };

  // Sets for update

  static AccClauseSet allowedClauses_ACCD_update {
   clang::OpenACCClauseKind::DeviceType,
   clang::OpenACCClauseKind::IfPresent,
   clang::OpenACCClauseKind::Wait,
  };

  static AccClauseSet allowedOnceClauses_ACCD_update {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_update {
  };

  static AccClauseSet requiredClauses_ACCD_update {
   clang::OpenACCClauseKind::Device,
   clang::OpenACCClauseKind::Host,
   clang::OpenACCClauseKind::Self,
  };

  // Sets for wait

  static AccClauseSet allowedClauses_ACCD_wait {
  };

  static AccClauseSet allowedOnceClauses_ACCD_wait {
   clang::OpenACCClauseKind::Async,
   clang::OpenACCClauseKind::If,
  };

  static AccClauseSet allowedExclusiveClauses_ACCD_wait {
  };

  static AccClauseSet requiredClauses_ACCD_wait {
  };
} // namespace acc
} // namespace llvm

#endif // GEN_CLANG_DIRECTIVE_CLAUSE_SETS

#ifdef GEN_CLANG_DIRECTIVE_CLAUSE_MAP
#undef GEN_CLANG_DIRECTIVE_CLAUSE_MAP

{
  {clang::OpenACCDirectiveKind::Atomic,
    {
      clang::acc::allowedClauses_ACCD_atomic,
      clang::acc::allowedOnceClauses_ACCD_atomic,
      clang::acc::allowedExclusiveClauses_ACCD_atomic,
      clang::acc::requiredClauses_ACCD_atomic,
    }
  },
  {clang::OpenACCDirectiveKind::Cache,
    {
      clang::acc::allowedClauses_ACCD_cache,
      clang::acc::allowedOnceClauses_ACCD_cache,
      clang::acc::allowedExclusiveClauses_ACCD_cache,
      clang::acc::requiredClauses_ACCD_cache,
    }
  },
  {clang::OpenACCDirectiveKind::Data,
    {
      clang::acc::allowedClauses_ACCD_data,
      clang::acc::allowedOnceClauses_ACCD_data,
      clang::acc::allowedExclusiveClauses_ACCD_data,
      clang::acc::requiredClauses_ACCD_data,
    }
  },
  {clang::OpenACCDirectiveKind::Declare,
    {
      clang::acc::allowedClauses_ACCD_declare,
      clang::acc::allowedOnceClauses_ACCD_declare,
      clang::acc::allowedExclusiveClauses_ACCD_declare,
      clang::acc::requiredClauses_ACCD_declare,
    }
  },
  {clang::OpenACCDirectiveKind::EnterData,
    {
      clang::acc::allowedClauses_ACCD_enter_data,
      clang::acc::allowedOnceClauses_ACCD_enter_data,
      clang::acc::allowedExclusiveClauses_ACCD_enter_data,
      clang::acc::requiredClauses_ACCD_enter_data,
    }
  },
  {clang::OpenACCDirectiveKind::ExitData,
    {
      clang::acc::allowedClauses_ACCD_exit_data,
      clang::acc::allowedOnceClauses_ACCD_exit_data,
      clang::acc::allowedExclusiveClauses_ACCD_exit_data,
      clang::acc::requiredClauses_ACCD_exit_data,
    }
  },
  {clang::OpenACCDirectiveKind::HostData,
    {
      clang::acc::allowedClauses_ACCD_host_data,
      clang::acc::allowedOnceClauses_ACCD_host_data,
      clang::acc::allowedExclusiveClauses_ACCD_host_data,
      clang::acc::requiredClauses_ACCD_host_data,
    }
  },
  {clang::OpenACCDirectiveKind::Init,
    {
      clang::acc::allowedClauses_ACCD_init,
      clang::acc::allowedOnceClauses_ACCD_init,
      clang::acc::allowedExclusiveClauses_ACCD_init,
      clang::acc::requiredClauses_ACCD_init,
    }
  },
  {clang::OpenACCDirectiveKind::Kernels,
    {
      clang::acc::allowedClauses_ACCD_kernels,
      clang::acc::allowedOnceClauses_ACCD_kernels,
      clang::acc::allowedExclusiveClauses_ACCD_kernels,
      clang::acc::requiredClauses_ACCD_kernels,
    }
  },
  {clang::OpenACCDirectiveKind::KernelsLoop,
    {
      clang::acc::allowedClauses_ACCD_kernels_loop,
      clang::acc::allowedOnceClauses_ACCD_kernels_loop,
      clang::acc::allowedExclusiveClauses_ACCD_kernels_loop,
      clang::acc::requiredClauses_ACCD_kernels_loop,
    }
  },
  {clang::OpenACCDirectiveKind::Loop,
    {
      clang::acc::allowedClauses_ACCD_loop,
      clang::acc::allowedOnceClauses_ACCD_loop,
      clang::acc::allowedExclusiveClauses_ACCD_loop,
      clang::acc::requiredClauses_ACCD_loop,
    }
  },
  {clang::OpenACCDirectiveKind::Parallel,
    {
      clang::acc::allowedClauses_ACCD_parallel,
      clang::acc::allowedOnceClauses_ACCD_parallel,
      clang::acc::allowedExclusiveClauses_ACCD_parallel,
      clang::acc::requiredClauses_ACCD_parallel,
    }
  },
  {clang::OpenACCDirectiveKind::ParallelLoop,
    {
      clang::acc::allowedClauses_ACCD_parallel_loop,
      clang::acc::allowedOnceClauses_ACCD_parallel_loop,
      clang::acc::allowedExclusiveClauses_ACCD_parallel_loop,
      clang::acc::requiredClauses_ACCD_parallel_loop,
    }
  },
  {clang::OpenACCDirectiveKind::Routine,
    {
      clang::acc::allowedClauses_ACCD_routine,
      clang::acc::allowedOnceClauses_ACCD_routine,
      clang::acc::allowedExclusiveClauses_ACCD_routine,
      clang::acc::requiredClauses_ACCD_routine,
    }
  },
  {clang::OpenACCDirectiveKind::Serial,
    {
      clang::acc::allowedClauses_ACCD_serial,
      clang::acc::allowedOnceClauses_ACCD_serial,
      clang::acc::allowedExclusiveClauses_ACCD_serial,
      clang::acc::requiredClauses_ACCD_serial,
    }
  },
  {clang::OpenACCDirectiveKind::SerialLoop,
    {
      clang::acc::allowedClauses_ACCD_serial_loop,
      clang::acc::allowedOnceClauses_ACCD_serial_loop,
      clang::acc::allowedExclusiveClauses_ACCD_serial_loop,
      clang::acc::requiredClauses_ACCD_serial_loop,
    }
  },
  {clang::OpenACCDirectiveKind::Set,
    {
      clang::acc::allowedClauses_ACCD_set,
      clang::acc::allowedOnceClauses_ACCD_set,
      clang::acc::allowedExclusiveClauses_ACCD_set,
      clang::acc::requiredClauses_ACCD_set,
    }
  },
  {clang::OpenACCDirectiveKind::Shutdown,
    {
      clang::acc::allowedClauses_ACCD_shutdown,
      clang::acc::allowedOnceClauses_ACCD_shutdown,
      clang::acc::allowedExclusiveClauses_ACCD_shutdown,
      clang::acc::requiredClauses_ACCD_shutdown,
    }
  },
  {clang::OpenACCDirectiveKind::Invalid,
    {
      clang::acc::allowedClauses_ACCD_unknown,
      clang::acc::allowedOnceClauses_ACCD_unknown,
      clang::acc::allowedExclusiveClauses_ACCD_unknown,
      clang::acc::requiredClauses_ACCD_unknown,
    }
  },
  {clang::OpenACCDirectiveKind::Update,
    {
      clang::acc::allowedClauses_ACCD_update,
      clang::acc::allowedOnceClauses_ACCD_update,
      clang::acc::allowedExclusiveClauses_ACCD_update,
      clang::acc::requiredClauses_ACCD_update,
    }
  },
  {clang::OpenACCDirectiveKind::Wait,
    {
      clang::acc::allowedClauses_ACCD_wait,
      clang::acc::allowedOnceClauses_ACCD_wait,
      clang::acc::allowedExclusiveClauses_ACCD_wait,
      clang::acc::requiredClauses_ACCD_wait,
    }
  },
}

#endif // GEN_CLANG_DIRECTIVE_CLAUSE_MAP

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
  CLAUSE(ACCC_##Name, #Name, /* Implicit */ false) \
  CLAUSE_CLASS(ACCC_##Name, #Name, Class)
#define __CLAUSE_NO_CLASS(Name)                    \
  CLAUSE(ACCC_##Name, #Name, /* Implicit */ false) \
  CLAUSE_NO_CLASS(ACCC_##Name, #Name)
#define __IMPLICIT_CLAUSE_CLASS(Name, Str, Class)  \
  CLAUSE(ACCC_##Name, Str, /* Implicit */ true)    \
  CLAUSE_CLASS(ACCC_##Name, Str, Class)
#define __IMPLICIT_CLAUSE_NO_CLASS(Name, Str)      \
  CLAUSE(ACCC_##Name, Str, /* Implicit */ true)    \
  CLAUSE_NO_CLASS(ACCC_##Name, Str)

__CLAUSE_NO_CLASS(async)
__CLAUSE_NO_CLASS(attach)
__CLAUSE_NO_CLASS(auto)
__CLAUSE_NO_CLASS(bind)
__CLAUSE_NO_CLASS(capture)
__CLAUSE_NO_CLASS(collapse)
__CLAUSE_NO_CLASS(copy)
__CLAUSE_NO_CLASS(copyin)
__CLAUSE_NO_CLASS(copyout)
__CLAUSE_NO_CLASS(create)
__CLAUSE_NO_CLASS(default)
__CLAUSE_NO_CLASS(default_async)
__CLAUSE_NO_CLASS(delete)
__CLAUSE_NO_CLASS(detach)
__CLAUSE_NO_CLASS(device)
__CLAUSE_NO_CLASS(device_num)
__CLAUSE_NO_CLASS(deviceptr)
__CLAUSE_NO_CLASS(device_resident)
__CLAUSE_NO_CLASS(device_type)
__CLAUSE_NO_CLASS(finalize)
__CLAUSE_NO_CLASS(firstprivate)
__CLAUSE_NO_CLASS(gang)
__CLAUSE_NO_CLASS(host)
__CLAUSE_NO_CLASS(if)
__CLAUSE_NO_CLASS(if_present)
__CLAUSE_NO_CLASS(independent)
__CLAUSE_NO_CLASS(link)
__CLAUSE_NO_CLASS(no_create)
__CLAUSE_NO_CLASS(nohost)
__CLAUSE_NO_CLASS(num_gangs)
__CLAUSE_NO_CLASS(num_workers)
__CLAUSE_NO_CLASS(present)
__CLAUSE_NO_CLASS(private)
__CLAUSE_NO_CLASS(read)
__CLAUSE_NO_CLASS(reduction)
__CLAUSE_NO_CLASS(self)
__CLAUSE_NO_CLASS(seq)
__CLAUSE_NO_CLASS(shortloop)
__CLAUSE_NO_CLASS(tile)
__CLAUSE_NO_CLASS(unknown)
__CLAUSE_NO_CLASS(use_device)
__CLAUSE_NO_CLASS(vector)
__CLAUSE_NO_CLASS(vector_length)
__CLAUSE_NO_CLASS(wait)
__CLAUSE_NO_CLASS(worker)
__CLAUSE_NO_CLASS(write)

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

Directive llvm::acc::getOpenACCDirectiveKind(llvm::StringRef Str) {
  return llvm::StringSwitch<Directive>(Str)
    .Case("atomic",ACCD_atomic)
    .Case("cache",ACCD_cache)
    .Case("data",ACCD_data)
    .Case("declare",ACCD_declare)
    .Case("enter data",ACCD_enter_data)
    .Case("exit data",ACCD_exit_data)
    .Case("host_data",ACCD_host_data)
    .Case("init",ACCD_init)
    .Case("kernels",ACCD_kernels)
    .Case("kernels loop",ACCD_kernels_loop)
    .Case("loop",ACCD_loop)
    .Case("parallel",ACCD_parallel)
    .Case("parallel loop",ACCD_parallel_loop)
    .Case("routine",ACCD_routine)
    .Case("serial",ACCD_serial)
    .Case("serial loop",ACCD_serial_loop)
    .Case("set",ACCD_set)
    .Case("shutdown",ACCD_shutdown)
    .Case("unknown",ACCD_unknown)
    .Case("update",ACCD_update)
    .Case("wait",ACCD_wait)
    .Default(ACCD_unknown);
}

llvm::StringRef llvm::acc::getOpenACCDirectiveName(Directive Kind) {
  switch (Kind) {
    case ACCD_atomic:
      return "atomic";
    case ACCD_cache:
      return "cache";
    case ACCD_data:
      return "data";
    case ACCD_declare:
      return "declare";
    case ACCD_enter_data:
      return "enter data";
    case ACCD_exit_data:
      return "exit data";
    case ACCD_host_data:
      return "host_data";
    case ACCD_init:
      return "init";
    case ACCD_kernels:
      return "kernels";
    case ACCD_kernels_loop:
      return "kernels loop";
    case ACCD_loop:
      return "loop";
    case ACCD_parallel:
      return "parallel";
    case ACCD_parallel_loop:
      return "parallel loop";
    case ACCD_routine:
      return "routine";
    case ACCD_serial:
      return "serial";
    case ACCD_serial_loop:
      return "serial loop";
    case ACCD_set:
      return "set";
    case ACCD_shutdown:
      return "shutdown";
    case ACCD_unknown:
      return "unknown";
    case ACCD_update:
      return "update";
    case ACCD_wait:
      return "wait";
  }
  llvm_unreachable("Invalid OpenACC Directive kind");
}

Clause llvm::acc::getOpenACCClauseKind(llvm::StringRef Str) {
  return llvm::StringSwitch<Clause>(Str)
    .Case("async",ACCC_async)
    .Case("attach",ACCC_attach)
    .Case("auto",ACCC_auto)
    .Case("bind",ACCC_bind)
    .Case("capture",ACCC_capture)
    .Case("collapse",ACCC_collapse)
    .Case("copy",ACCC_copy)
    .Case("copyin",ACCC_copyin)
    .Case("copyout",ACCC_copyout)
    .Case("create",ACCC_create)
    .Case("default",ACCC_default)
    .Case("default_async",ACCC_default_async)
    .Case("delete",ACCC_delete)
    .Case("detach",ACCC_detach)
    .Case("device",ACCC_device)
    .Case("device_num",ACCC_device_num)
    .Case("deviceptr",ACCC_deviceptr)
    .Case("device_resident",ACCC_device_resident)
    .Case("device_type",ACCC_device_type)
    .Case("finalize",ACCC_finalize)
    .Case("firstprivate",ACCC_firstprivate)
    .Case("gang",ACCC_gang)
    .Case("host",ACCC_host)
    .Case("if",ACCC_if)
    .Case("if_present",ACCC_if_present)
    .Case("independent",ACCC_independent)
    .Case("link",ACCC_link)
    .Case("no_create",ACCC_no_create)
    .Case("nohost",ACCC_nohost)
    .Case("num_gangs",ACCC_num_gangs)
    .Case("num_workers",ACCC_num_workers)
    .Case("present",ACCC_present)
    .Case("private",ACCC_private)
    .Case("read",ACCC_read)
    .Case("reduction",ACCC_reduction)
    .Case("self",ACCC_self)
    .Case("seq",ACCC_seq)
    .Case("shortloop",ACCC_shortloop)
    .Case("tile",ACCC_tile)
    .Case("unknown",ACCC_unknown)
    .Case("use_device",ACCC_use_device)
    .Case("vector",ACCC_vector)
    .Case("vector_length",ACCC_vector_length)
    .Case("wait",ACCC_wait)
    .Case("worker",ACCC_worker)
    .Case("write",ACCC_write)
    .Default(ACCC_unknown);
}

llvm::StringRef llvm::acc::getOpenACCClauseName(Clause Kind) {
  switch (Kind) {
    case ACCC_async:
      return "async";
    case ACCC_attach:
      return "attach";
    case ACCC_auto:
      return "auto";
    case ACCC_bind:
      return "bind";
    case ACCC_capture:
      return "capture";
    case ACCC_collapse:
      return "collapse";
    case ACCC_copy:
      return "copy";
    case ACCC_copyin:
      return "copyin";
    case ACCC_copyout:
      return "copyout";
    case ACCC_create:
      return "create";
    case ACCC_default:
      return "default";
    case ACCC_default_async:
      return "default_async";
    case ACCC_delete:
      return "delete";
    case ACCC_detach:
      return "detach";
    case ACCC_device:
      return "device";
    case ACCC_device_num:
      return "device_num";
    case ACCC_deviceptr:
      return "deviceptr";
    case ACCC_device_resident:
      return "device_resident";
    case ACCC_device_type:
      return "device_type";
    case ACCC_finalize:
      return "finalize";
    case ACCC_firstprivate:
      return "firstprivate";
    case ACCC_gang:
      return "gang";
    case ACCC_host:
      return "host";
    case ACCC_if:
      return "if";
    case ACCC_if_present:
      return "if_present";
    case ACCC_independent:
      return "independent";
    case ACCC_link:
      return "link";
    case ACCC_no_create:
      return "no_create";
    case ACCC_nohost:
      return "nohost";
    case ACCC_num_gangs:
      return "num_gangs";
    case ACCC_num_workers:
      return "num_workers";
    case ACCC_present:
      return "present";
    case ACCC_private:
      return "private";
    case ACCC_read:
      return "read";
    case ACCC_reduction:
      return "reduction";
    case ACCC_self:
      return "self";
    case ACCC_seq:
      return "seq";
    case ACCC_shortloop:
      return "shortloop";
    case ACCC_tile:
      return "tile";
    case ACCC_unknown:
      return "unknown";
    case ACCC_use_device:
      return "use_device";
    case ACCC_vector:
      return "vector";
    case ACCC_vector_length:
      return "vector_length";
    case ACCC_wait:
      return "wait";
    case ACCC_worker:
      return "worker";
    case ACCC_write:
      return "write";
  }
  llvm_unreachable("Invalid OpenACC Clause kind");
}

DefaultValue llvm::acc::getDefaultValue(llvm::StringRef Str) {
  return llvm::StringSwitch<DefaultValue>(Str)
    .Case("present",ACC_Default_present)
    .Case("none",ACC_Default_none)
    .Default(ACC_Default_none);
}

llvm::StringRef llvm::acc::getOpenACCDefaultValueName(llvm::acc::DefaultValue x) {
  switch (x) {
    case ACC_Default_present:
      return "present";
    case ACC_Default_none:
      return "none";
  }
  llvm_unreachable("Invalid OpenACC DefaultValue kind");
}

bool llvm::acc::isAllowedClauseForDirective(Directive D, Clause C, unsigned Version) {
  assert(unsigned(D) <= llvm::acc::Directive_enumSize);
  assert(unsigned(C) <= llvm::acc::Clause_enumSize);
  switch (D) {
    case ACCD_atomic:
      switch (C) {
        case ACCC_if:
          return 34 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_cache:
      return false;
      break;
    case ACCD_data:
      switch (C) {
        case ACCC_device_type:
          return 32 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 32 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 32 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_declare:
      switch (C) {
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_resident:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_link:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_enter_data:
      switch (C) {
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_exit_data:
      switch (C) {
        case ACCC_finalize:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_delete:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_detach:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_host_data:
      switch (C) {
        case ACCC_if_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_use_device:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_init:
      switch (C) {
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_num:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_kernels:
      switch (C) {
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_gangs:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_workers:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector_length:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_kernels_loop:
      switch (C) {
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_gang:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_gangs:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_workers:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_private:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_shortloop:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_tile:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector_length:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_worker:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_auto:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_independent:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_seq:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_loop:
      switch (C) {
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_private:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_gang:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_shortloop:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_tile:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_worker:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_auto:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_independent:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_seq:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_parallel:
      switch (C) {
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_gangs:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_workers:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_private:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector_length:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_parallel_loop:
      switch (C) {
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_gang:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_gangs:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_num_workers:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_private:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_shortloop:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_tile:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector_length:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_worker:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_auto:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_independent:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_seq:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_routine:
      switch (C) {
        case ACCC_bind:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_gang:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_seq:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_worker:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_nohost:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_serial:
      switch (C) {
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_private:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_serial_loop:
      switch (C) {
        case ACCC_attach:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_collapse:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copy:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyin:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_copyout:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_deviceptr:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_firstprivate:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_gang:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_no_create:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_private:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_reduction:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_shortloop:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_tile:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_vector:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_worker:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_default:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_auto:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_independent:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_seq:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_set:
      switch (C) {
        case ACCC_default_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_num:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_shutdown:
      switch (C) {
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device_num:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_unknown:
      return false;
      break;
    case ACCD_update:
      switch (C) {
        case ACCC_device_type:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if_present:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_wait:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_device:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_host:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_self:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
    case ACCD_wait:
      switch (C) {
        case ACCC_async:
          return 1 <= Version && 2147483647 >= Version;
        case ACCC_if:
          return 1 <= Version && 2147483647 >= Version;
        default:
          return false;
      }
      break;
  }
  llvm_unreachable("Invalid OpenACC Directive kind");
}

llvm::acc::Association llvm::acc::getDirectiveAssociation(llvm::acc::Directive Dir) {
  switch (Dir) {
  case llvm::acc::Directive::ACCD_atomic:
    return llvm::acc::Association::Block;
  case llvm::acc::Directive::ACCD_cache:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_data:
    return llvm::acc::Association::Block;
  case llvm::acc::Directive::ACCD_declare:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_enter_data:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_exit_data:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_host_data:
    return llvm::acc::Association::Block;
  case llvm::acc::Directive::ACCD_init:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_kernels:
    return llvm::acc::Association::Block;
  case llvm::acc::Directive::ACCD_kernels_loop:
    return llvm::acc::Association::Loop;
  case llvm::acc::Directive::ACCD_loop:
    return llvm::acc::Association::Loop;
  case llvm::acc::Directive::ACCD_parallel:
    return llvm::acc::Association::Block;
  case llvm::acc::Directive::ACCD_parallel_loop:
    return llvm::acc::Association::Loop;
  case llvm::acc::Directive::ACCD_routine:
    return llvm::acc::Association::Declaration;
  case llvm::acc::Directive::ACCD_serial:
    return llvm::acc::Association::Block;
  case llvm::acc::Directive::ACCD_serial_loop:
    return llvm::acc::Association::Loop;
  case llvm::acc::Directive::ACCD_set:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_shutdown:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_unknown:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_update:
    return llvm::acc::Association::None;
  case llvm::acc::Directive::ACCD_wait:
    return llvm::acc::Association::None;
  } // switch (Dir)
  llvm_unreachable("Unexpected directive");
}

llvm::acc::Category llvm::acc::getDirectiveCategory(llvm::acc::Directive Dir) {
  switch (Dir) {
  case llvm::acc::ACCD_atomic:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_cache:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_data:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_declare:
    return llvm::acc::Category::Declarative;
  case llvm::acc::ACCD_enter_data:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_exit_data:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_host_data:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_init:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_kernels:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_kernels_loop:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_loop:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_parallel:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_parallel_loop:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_routine:
    return llvm::acc::Category::Declarative;
  case llvm::acc::ACCD_serial:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_serial_loop:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_set:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_shutdown:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_unknown:
    return llvm::acc::Category::Utility;
  case llvm::acc::ACCD_update:
    return llvm::acc::Category::Executable;
  case llvm::acc::ACCD_wait:
    return llvm::acc::Category::Executable;
  } // switch (Dir)
  llvm_unreachable("Unexpected directive");
}

static_assert(sizeof(llvm::acc::Directive) == sizeof(int));
[[maybe_unused]] static const llvm::acc::Directive LeafConstructTable[][4] = {
    {llvm::acc::ACCD_atomic, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_cache, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_data, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_declare, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_enter_data, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_exit_data, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_host_data, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_init, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_kernels, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_loop, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_parallel, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_routine, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_serial, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_set, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_shutdown, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_unknown, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_update, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_wait, static_cast<llvm::acc::Directive>(0), static_cast<llvm::acc::Directive>(-1), static_cast<llvm::acc::Directive>(-1),},
    {llvm::acc::ACCD_kernels_loop, static_cast<llvm::acc::Directive>(2), llvm::acc::ACCD_kernels, llvm::acc::ACCD_loop,},
    {llvm::acc::ACCD_parallel_loop, static_cast<llvm::acc::Directive>(2), llvm::acc::ACCD_parallel, llvm::acc::ACCD_loop,},
    {llvm::acc::ACCD_serial_loop, static_cast<llvm::acc::Directive>(2), llvm::acc::ACCD_serial, llvm::acc::ACCD_loop,},
};

[[maybe_unused]] static auto LeafConstructTableEndDirective = LeafConstructTable + 21;

[[maybe_unused]] static const int LeafConstructTableOrdering[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 9, 10, 19, 11, 12, 20, 13, 14, 15, 16, 17,
};

#endif // GEN_DIRECTIVES_IMPL

