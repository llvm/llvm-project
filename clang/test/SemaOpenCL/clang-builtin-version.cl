// RUN: %clang_cc1 %s -fblocks -verify -pedantic -fsyntax-only -ferror-limit 100

// Confirm CL2.0 Clang builtins are not available in earlier versions

kernel void dse_builtins() {
  int tmp;
  enqueue_kernel(tmp, tmp, tmp, ^(void) { // expected-warning{{implicit declaration of function 'enqueue_kernel' is invalid in C99}}
    return;
  });
  unsigned size = get_kernel_work_group_size(^(void) { // expected-warning{{implicit declaration of function 'get_kernel_work_group_size' is invalid in C99}}
    return;
  });
  size = get_kernel_preferred_work_group_size_multiple(^(void) { // expected-warning{{implicit declaration of function 'get_kernel_preferred_work_group_size_multiple' is invalid in C99}}
    return;
  });
}

void pipe_builtins() {
  int tmp;

  read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'read_pipe' is invalid in C99}}
  write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'write_pipe' is invalid in C99}}

  reserve_read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'reserve_read_pipe' is invalid in C99}}
  reserve_write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'reserve_write_pipe' is invalid in C99}}

  work_group_reserve_read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'work_group_reserve_read_pipe' is invalid in C99}}
  work_group_reserve_write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'work_group_reserve_write_pipe' is invalid in C99}}

  sub_group_reserve_write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'sub_group_reserve_write_pipe' is invalid in C99}}
  sub_group_reserve_read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'sub_group_reserve_read_pipe' is invalid in C99}}

  commit_read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'commit_read_pipe' is invalid in C99}}
  commit_write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'commit_write_pipe' is invalid in C99}}

  work_group_commit_read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'work_group_commit_read_pipe' is invalid in C99}}
  work_group_commit_write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'work_group_commit_write_pipe' is invalid in C99}}

  sub_group_commit_write_pipe(tmp, tmp); // expected-warning{{implicit declaration of function 'sub_group_commit_write_pipe' is invalid in C99}}
  sub_group_commit_read_pipe(tmp, tmp);  // expected-warning{{implicit declaration of function 'sub_group_commit_read_pipe' is invalid in C99}}

  get_pipe_num_packets(tmp); // expected-warning{{implicit declaration of function 'get_pipe_num_packets' is invalid in C99}}
  get_pipe_max_packets(tmp); // expected-warning{{implicit declaration of function 'get_pipe_max_packets' is invalid in C99}}
}
