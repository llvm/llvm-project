// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100
// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100 -cl-std=CL1.2
// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100 -cl-std=CL3.0 -cl-ext=-__opencl_c_device_enqueue,-__opencl_c_generic_address_space,-__opencl_c_pipes

// Confirm CL2.0 Clang builtins are not available in earlier versions and in OpenCL C 3.0 without required features.

kernel void dse_builtins(void) {
  int tmp;
  enqueue_kernel(tmp, tmp, tmp, ^(void) { // expected-error{{use of undeclared identifier 'enqueue_kernel'}}
    return;
  });
  unsigned size = get_kernel_work_group_size(^(void) { // expected-error{{use of undeclared identifier 'get_kernel_work_group_size'}}
    return;
  });
  size = get_kernel_preferred_work_group_size_multiple(^(void) { // expected-error{{use of undeclared identifier 'get_kernel_preferred_work_group_size_multiple'}}
    return;
  });
#if (__OPENCL_C_VERSION__ >= CL_VERSION_3_0) && !defined(__opencl_c_device_enqueue)
// expected-error@-10{{support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
// FIXME: the typo correction for the undeclared identifiers finds alternative
// suggestions, but instantiating the typo correction causes us to
// re-instantiate the argument to the call, which triggers the support
// diagnostic a second time.
// expected-error@-12 2{{support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
// expected-error@-10 2{{support disabled - compile with -fblocks or for OpenCL C 2.0 or OpenCL C 3.0 with __opencl_c_device_enqueue feature}}
#endif
}

void pipe_builtins(void) {
  int tmp;

  // FIXME: the typo correction for this case goes off the rails and tries to
  // convert this mistake into a for loop instead of a local function
  // declaration.
  foo(void); // expected-error{{use of undeclared identifier 'foo'; did you mean 'for'?}}
             // expected-error@-1{{expected identifier or '('}}
             // expected-note@-2{{to match this '('}}
  boo(); // expected-error{{use of undeclared identifier 'boo'}}
         // expected-error@-1{{expected ';' in 'for' statement specifier}}

  read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'read_pipe'}}
                        // expected-error@-1{{expected ')'}}
  write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'write_pipe'}}

  reserve_read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'reserve_read_pipe'}}
  reserve_write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'reserve_write_pipe'}}

  work_group_reserve_read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'work_group_reserve_read_pipe'}}
  work_group_reserve_write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'work_group_reserve_write_pipe'}}

  sub_group_reserve_write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'sub_group_reserve_write_pipe'}}
  sub_group_reserve_read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'sub_group_reserve_read_pipe'}}

  commit_read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'commit_read_pipe'}}
  commit_write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'commit_write_pipe'}}

  work_group_commit_read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'work_group_commit_read_pipe'}}
  work_group_commit_write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'work_group_commit_write_pipe'}}

  sub_group_commit_write_pipe(tmp, tmp); // expected-error{{use of undeclared identifier 'sub_group_commit_write_pipe'}}
  sub_group_commit_read_pipe(tmp, tmp);  // expected-error{{use of undeclared identifier 'sub_group_commit_read_pipe'}}

  get_pipe_num_packets(tmp); // expected-error{{use of undeclared identifier 'get_pipe_num_packets'}}
  get_pipe_max_packets(tmp); // expected-error{{use of undeclared identifier 'get_pipe_max_packets'}}
}
