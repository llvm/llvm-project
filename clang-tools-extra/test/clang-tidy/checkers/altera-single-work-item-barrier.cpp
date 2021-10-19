// RUN: %check_clang_tidy -check-suffix=OLDCLOLDAOC %s altera-single-work-item-barrier %t -- -header-filter=.* "--" -cl-std=CL1.2 -c --include opencl-c.h -DOLDCLOLDAOC
// RUN: %check_clang_tidy -check-suffix=NEWCLOLDAOC %s altera-single-work-item-barrier %t -- -header-filter=.* "--" -cl-std=CL2.0 -c --include opencl-c.h -DNEWCLOLDAOC
// RUN: %check_clang_tidy -check-suffix=OLDCLNEWAOC %s altera-single-work-item-barrier %t -- -config='{CheckOptions: [{key: altera-single-work-item-barrier.AOCVersion, value: 1701}]}' -header-filter=.* "--" -cl-std=CL1.2 -c --include opencl-c.h -DOLDCLNEWAOC
// RUN: %check_clang_tidy -check-suffix=NEWCLNEWAOC %s altera-single-work-item-barrier %t -- -config='{CheckOptions: [{key: altera-single-work-item-barrier.AOCVersion, value: 1701}]}' -header-filter=.* "--" -cl-std=CL2.0 -c --include opencl-c.h -DNEWCLNEWAOC

#ifdef OLDCLOLDAOC  // OpenCL 1.2 Altera Offline Compiler < 17.1
void __kernel error_barrier_no_id(__global int * foo, int size) {
  // CHECK-MESSAGES-OLDCLOLDAOC: :[[@LINE-1]]:15: warning: kernel function 'error_barrier_no_id' does not call 'get_global_id' or 'get_local_id' and will be treated as a single work-item [altera-single-work-item-barrier]
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  // CHECK-MESSAGES-OLDCLOLDAOC: :[[@LINE-1]]:3: note: barrier call is in a single work-item and may error out
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void __kernel success_barrier_global_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void __kernel success_barrier_local_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void __kernel success_barrier_both_ids(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}

void success_nokernel_barrier_no_id(__global int * foo, int size) {
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void success_nokernel_barrier_global_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void success_nokernel_barrier_local_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void success_nokernel_barrier_both_ids(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}
#endif

#ifdef NEWCLOLDAOC  // OpenCL 2.0 Altera Offline Compiler < 17.1
void __kernel error_barrier_no_id(__global int * foo, int size) {
  // CHECK-MESSAGES-NEWCLOLDAOC: :[[@LINE-1]]:15: warning: kernel function 'error_barrier_no_id' does not call 'get_global_id' or 'get_local_id' and will be treated as a single work-item [altera-single-work-item-barrier]
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  // CHECK-MESSAGES-NEWCLOLDAOC: :[[@LINE-1]]:3: note: barrier call is in a single work-item and may error out
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void __kernel success_barrier_global_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void __kernel success_barrier_local_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void __kernel success_barrier_both_ids(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}

void success_nokernel_barrier_no_id(__global int * foo, int size) {
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void success_nokernel_barrier_global_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void success_nokernel_barrier_local_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void success_nokernel_barrier_both_ids(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}
#endif

#ifdef OLDCLNEWAOC  // OpenCL 1.2 Altera Offline Compiler >= 17.1
void __kernel error_barrier_no_id(__global int * foo, int size) {
  // CHECK-MESSAGES-OLDCLNEWAOC: :[[@LINE-1]]:15: warning: kernel function 'error_barrier_no_id' does not call an ID function and may be a viable single work-item, but will be forced to execute as an NDRange [altera-single-work-item-barrier]
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  // CHECK-MESSAGES-OLDCLNEWAOC: :[[@LINE-1]]:3: note: barrier call will force NDRange execution; if single work-item semantics are desired a mem_fence may be more efficient
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

__attribute__ ((reqd_work_group_size(1,1,1)))
void __kernel error_barrier_no_id_work_group_size(__global int * foo, int size) {
  // CHECK-MESSAGES-OLDCLNEWAOC: :[[@LINE-1]]:15: warning: kernel function 'error_barrier_no_id_work_group_size' does not call an ID function and may be a viable single work-item, but will be forced to execute as an NDRange [altera-single-work-item-barrier]
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  // CHECK-MESSAGES-OLDCLNEWAOC: :[[@LINE-1]]:3: note: barrier call will force NDRange execution; if single work-item semantics are desired a mem_fence may be more efficient
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

__attribute__ ((reqd_work_group_size(2,1,1)))
void __kernel success_barrier_no_id_work_group_size(__global int * foo, int size) {
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void __kernel success_barrier_global_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void __kernel success_barrier_local_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void __kernel success_barrier_both_ids(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}

void success_nokernel_barrier_no_id(__global int * foo, int size) {
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void success_nokernel_barrier_global_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void success_nokernel_barrier_local_id(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void success_nokernel_barrier_both_ids(__global int * foo, int size) {
  barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}
#endif

#ifdef NEWCLNEWAOC  // OpenCL 2.0 Altera Offline Compiler >= 17.1
void __kernel error_barrier_no_id(__global int * foo, int size) {
  // CHECK-MESSAGES-NEWCLNEWAOC: :[[@LINE-1]]:15: warning: kernel function 'error_barrier_no_id' does not call an ID function and may be a viable single work-item, but will be forced to execute as an NDRange [altera-single-work-item-barrier]
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  // CHECK-MESSAGES-NEWCLNEWAOC: :[[@LINE-1]]:3: note: barrier call will force NDRange execution; if single work-item semantics are desired a mem_fence may be more efficient
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

__attribute__ ((reqd_work_group_size(1,1,1)))
void __kernel error_barrier_no_id_work_group_size(__global int * foo, int size) {
  // CHECK-MESSAGES-NEWCLNEWAOC: :[[@LINE-1]]:15: warning: kernel function 'error_barrier_no_id_work_group_size' does not call an ID function and may be a viable single work-item, but will be forced to execute as an NDRange [altera-single-work-item-barrier]
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  // CHECK-MESSAGES-NEWCLNEWAOC: :[[@LINE-1]]:3: note: barrier call will force NDRange execution; if single work-item semantics are desired a mem_fence may be more efficient
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

__attribute__ ((reqd_work_group_size(2,1,1)))
void __kernel success_barrier_no_id_work_group_size(__global int * foo, int size) {
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void __kernel success_barrier_global_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void __kernel success_barrier_local_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void __kernel success_barrier_both_ids(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}

void success_nokernel_barrier_no_id(__global int * foo, int size) {
  for (int j = 0; j < 256; j++) {
	for (int i = 256; i < size; i+= 256) {
      foo[j] += foo[j+i];
    }
  }
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  for (int i = 1; i < 256; i++) {
	foo[0] += foo[i];
  }
}

void success_nokernel_barrier_global_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_global_id(0);
}

void success_nokernel_barrier_local_id(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int tid = get_local_id(0);
}

void success_nokernel_barrier_both_ids(__global int * foo, int size) {
  work_group_barrier(CLK_GLOBAL_MEM_FENCE);
  int gid = get_global_id(0);
  int lid = get_local_id(0);
}
#endif
