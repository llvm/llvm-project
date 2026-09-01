// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only \
// RUN:   -cl-std=CL2.0 -cl-ext=+cl_khr_subgroups

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

void read_pipe_builtins(read_only pipe int p, global int *ptr) {
  int tmp;
  reserve_id_t rid;

  read_pipe(p, &tmp);
  read_pipe(p, ptr);
  // expected-error@+1 {{first argument to 'read_pipe' must be a pipe type}}
  read_pipe(tmp, p);
  // expected-error@+1 {{invalid number of arguments to function: 'read_pipe'}}
  read_pipe(p);
  read_pipe(p, rid, tmp, ptr);
  // expected-error@+1 {{invalid argument type to function 'read_pipe' (expecting 'reserve_id_t' having '__private int')}}
  read_pipe(p, tmp, tmp, ptr);
  // expected-error@+1 {{invalid argument type to function 'read_pipe' (expecting 'unsigned int' having '__private reserve_id_t')}}
  read_pipe(p, rid, rid, ptr);
  // expected-error@+1 {{invalid argument type to function 'read_pipe' (expecting 'int *' having '__private int')}}
  read_pipe(p, tmp);
  // expected-error@+1 {{invalid pipe access modifier (expecting write_only)}}
  write_pipe(p, ptr);
  // expected-error@+1 {{invalid pipe access modifier (expecting write_only)}}
  write_pipe(p, rid, tmp, ptr);

  reserve_read_pipe(p, tmp);
  // expected-error@+1 {{invalid argument type to function 'reserve_read_pipe' (expecting 'unsigned int' having '__global int *__private')}}
  reserve_read_pipe(p, ptr);
  // expected-error@+1 {{first argument to 'work_group_reserve_read_pipe' must be a pipe type}}
  work_group_reserve_read_pipe(tmp, tmp);
  // expected-error@+1 {{invalid pipe access modifier (expecting write_only)}}
  sub_group_reserve_write_pipe(p, tmp);

  commit_read_pipe(p, rid);
  // expected-error@+1 {{first argument to 'commit_read_pipe' must be a pipe type}}
  commit_read_pipe(tmp, rid);
  // expected-error@+1 {{invalid argument type to function 'work_group_commit_read_pipe' (expecting 'reserve_id_t' having '__private int')}}
  work_group_commit_read_pipe(p, tmp);
  // expected-error@+1 {{invalid pipe access modifier (expecting write_only)}}
  sub_group_commit_write_pipe(p, tmp);
}

void write_pipe_builtins(write_only pipe int p, global int *ptr) {
  int tmp;
  reserve_id_t rid;

  write_pipe(p, &tmp);
  write_pipe(p, ptr);
  // expected-error@+1 {{first argument to 'write_pipe' must be a pipe type}}
  write_pipe(tmp, p);
  // expected-error@+1 {{invalid number of arguments to function: 'write_pipe'}}
  write_pipe(p);
  write_pipe(p, rid, tmp, ptr);
  // expected-error@+1 {{invalid argument type to function 'write_pipe' (expecting 'reserve_id_t' having '__private int')}}
  write_pipe(p, tmp, tmp, ptr);
  // expected-error@+1 {{invalid argument type to function 'write_pipe' (expecting 'unsigned int' having '__private reserve_id_t')}}
  write_pipe(p, rid, rid, ptr);
  // expected-error@+1 {{invalid argument type to function 'write_pipe' (expecting 'int *' having '__private int')}}
  write_pipe(p, tmp);
  // expected-error@+1 {{invalid pipe access modifier (expecting read_only)}}
  read_pipe(p, ptr);
  // expected-error@+1 {{invalid pipe access modifier (expecting read_only)}}
  read_pipe(p, rid, tmp, ptr);

  reserve_write_pipe(p, tmp);
  // expected-error@+1 {{invalid argument type to function 'reserve_write_pipe' (expecting 'unsigned int' having '__global int *__private')}}
  reserve_write_pipe(p, ptr);
  // expected-error@+1 {{first argument to 'work_group_reserve_write_pipe' must be a pipe type}}
  work_group_reserve_write_pipe(tmp, tmp);
  // expected-error@+1 {{invalid pipe access modifier (expecting read_only)}}
  sub_group_reserve_read_pipe(p, tmp);

  commit_write_pipe(p, rid);
  // expected-error@+1 {{first argument to 'commit_write_pipe' must be a pipe type}}
  commit_write_pipe(tmp, rid);
  // expected-error@+1 {{invalid argument type to function 'work_group_commit_write_pipe' (expecting 'reserve_id_t' having '__private int')}}
  work_group_commit_write_pipe(p, tmp);
  // expected-error@+1 {{invalid pipe access modifier (expecting read_only)}}
  sub_group_commit_read_pipe(p, tmp);
}

void pipe_query_builtins(void) {
  int tmp;
  // expected-error@+1 {{first argument to 'get_pipe_num_packets' must be a pipe type}}
  get_pipe_num_packets(tmp);
  // expected-error@+1 {{first argument to 'get_pipe_max_packets' must be a pipe type}}
  get_pipe_max_packets(tmp);
}
