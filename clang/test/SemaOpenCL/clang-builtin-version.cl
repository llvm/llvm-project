// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100

// Confirm CL2.0 Clang builtins are not available in earlier versions

kernel void dse_builtins() {
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
}

void pipe_builtins() {
  int tmp;

  foo(void); // expected-error{{implicit declaration of function 'foo' is invalid in OpenCL}}
  // expected-note@-1{{'foo' declared here}}
  // expected-error@-2{{expected expression}}
  boo(); // expected-error{{implicit declaration of function 'boo' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'foo'?}}

  read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'read_pipe' is invalid in OpenCL}}
  write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'write_pipe' is invalid in OpenCL}}

  reserve_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'reserve_read_pipe' is invalid in OpenCL}}
  // expected-note@-1{{'reserve_read_pipe' declared here}}
  reserve_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'reserve_write_pipe' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'reserve_read_pipe'?}}

  work_group_reserve_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'work_group_reserve_read_pipe' is invalid in OpenCL}}
  // expected-note@-1 2{{'work_group_reserve_read_pipe' declared here}}
  work_group_reserve_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'work_group_reserve_write_pipe' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'work_group_reserve_read_pipe'?}}
  // expected-note@-2{{'work_group_reserve_write_pipe' declared here}}

  sub_group_reserve_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'sub_group_reserve_write_pipe' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'work_group_reserve_write_pipe'?}}
  sub_group_reserve_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'sub_group_reserve_read_pipe' is invalid in OpenCL}}

  commit_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'commit_read_pipe' is invalid in OpenCL}}
  // expected-note@-1{{'commit_read_pipe' declared here}}
  commit_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'commit_write_pipe' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'commit_read_pipe'?}}

  work_group_commit_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'work_group_commit_read_pipe' is invalid in OpenCL}}
  // expected-note@-1{{'work_group_commit_read_pipe' declared here}}
  // expected-note@-2{{did you mean 'work_group_reserve_read_pipe'?}}
  work_group_commit_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'work_group_commit_write_pipe' is invalid in OpenCL}}
  // expected-note@-1{{'work_group_commit_write_pipe' declared here}}
  // expected-note@-2{{did you mean 'work_group_commit_read_pipe'?}}

  sub_group_commit_write_pipe(tmp, tmp); // expected-error{{implicit declaration of function 'sub_group_commit_write_pipe' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'work_group_commit_write_pipe'?}}
  sub_group_commit_read_pipe(tmp, tmp);  // expected-error{{implicit declaration of function 'sub_group_commit_read_pipe' is invalid in OpenCL}}

  get_pipe_num_packets(tmp); // expected-error{{implicit declaration of function 'get_pipe_num_packets' is invalid in OpenCL}}
  // expected-note@-1{{'get_pipe_num_packets' declared here}}
  get_pipe_max_packets(tmp); // expected-error{{implicit declaration of function 'get_pipe_max_packets' is invalid in OpenCL}}
  // expected-note@-1{{did you mean 'get_pipe_num_packets'?}}
}
