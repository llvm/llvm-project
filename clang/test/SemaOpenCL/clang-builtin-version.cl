// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100
// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100 -cl-std=CL1.2
// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100 -cl-std=CL3.0 -cl-ext=-__opencl_c_device_enqueue,-__opencl_c_generic_address_space,-__opencl_c_pipes

// Confirm CL2.0 Clang builtins are not available in earlier versions and in OpenCL C 3.0 without required features.

kernel void dse_builtins(void) {
  int tmp;
  enqueue_kernel(tmp, tmp, tmp, ^(void) { // expected-error{{implicit declaration of function 'enqueue_kernel' is invalid in OpenCL}}
    return;
  });
  unsigned size = get_kernel_work_group_size(^(void) { // expected-error{{implicit declaration of function 'get_kernel_work_group_size' is invalid in OpenCL}}
    return;
  });
  size = get_kernel_preferred_work_group_size_multiple(^(void) { // expected-error{{implicit declaration of function 'get_kernel_preferred_work_group_size_multiple' is invalid in OpenCL}}
    return;
  });
#if (__OPENCL_C_VERSION__ >= CL_VERSION_3_0) && !defined(__opencl_c_device_enqueue)
// expected-error@-10{{support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
// expected-error@-8{{support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
// expected-error@-6{{support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
#endif
}

void pipe_builtins(void) {
  int tmp;

  foo(void); // expected-error{{implicit declaration of function 'foo' is invalid in OpenCL}}
  // expected-error@-1{{expected expression}}
  boo(); // expected-error{{implicit declaration of function 'boo' is invalid in OpenCL}}

  read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'read_pipe' is invalid in OpenCL}}
  write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'write_pipe' is invalid in OpenCL}}

  reserve_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'reserve_read_pipe' is invalid in OpenCL}}
  reserve_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'reserve_write_pipe' is invalid in OpenCL}}

  work_group_reserve_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'work_group_reserve_read_pipe' is invalid in OpenCL}}
  work_group_reserve_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'work_group_reserve_write_pipe' is invalid in OpenCL}}

  sub_group_reserve_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'sub_group_reserve_write_pipe' is invalid in OpenCL}}
  sub_group_reserve_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'sub_group_reserve_read_pipe' is invalid in OpenCL}}

  commit_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'commit_read_pipe' is invalid in OpenCL}}
  commit_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'commit_write_pipe' is invalid in OpenCL}}

  work_group_commit_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'work_group_commit_read_pipe' is invalid in OpenCL}}
  work_group_commit_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'work_group_commit_write_pipe' is invalid in OpenCL}}

  sub_group_commit_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'sub_group_commit_write_pipe' is invalid in OpenCL}}
  sub_group_commit_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'sub_group_commit_read_pipe' is invalid in OpenCL}}

  get_pipe_num_packets(tmp); // expected-error{{implicit declaration of function 'get_pipe_num_packets' is invalid in OpenCL}}
  get_pipe_max_packets(tmp); // expected-error{{implicit declaration of function 'get_pipe_max_packets' is invalid in OpenCL}}
}
