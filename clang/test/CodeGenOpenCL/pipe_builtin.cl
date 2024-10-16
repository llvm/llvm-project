// RUN: %clang_cc1 -triple spir-unknown-unknown -emit-llvm -cl-ext=+cl_khr_subgroups -O0 -cl-std=clc++ -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

void test1(read_only pipe int p, global int *ptr) {
  // CHECK: call spir_func i32 @__read_pipe_2(target("spirv.Pipe", 0) %{{.*}}, ptr addrspace(4) %{{.*}}, i32 4, i32 4)
  read_pipe(p, ptr);
  // CHECK: call spir_func target("spirv.ReserveId") @__reserve_read_pipe(target("spirv.Pipe", 0) %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = reserve_read_pipe(p, 2);
  // CHECK: call spir_func i32 @__read_pipe_4(target("spirv.Pipe", 0) %{{.*}}, ptr addrspace(4) %{{.*}}, i32 4, i32 4)
  read_pipe(p, rid, 2, ptr);
  // CHECK: call spir_func void @__commit_read_pipe(target("spirv.Pipe", 0) %{{.*}}, target("spirv.ReserveId") %{{.*}}, i32 4, i32 4)
  commit_read_pipe(p, rid);
}

void test2(write_only pipe int p, global int *ptr) {
  // CHECK: call spir_func i32 @__write_pipe_2(target("spirv.Pipe", 1) %{{.*}}, ptr addrspace(4) %{{.*}}, i32 4, i32 4)
  write_pipe(p, ptr);
  // CHECK: call spir_func target("spirv.ReserveId") @__reserve_write_pipe(target("spirv.Pipe", 1) %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = reserve_write_pipe(p, 2);
  // CHECK: call spir_func i32 @__write_pipe_4(target("spirv.Pipe", 1) %{{.*}}, ptr addrspace(4) %{{.*}}, i32 4, i32 4)
  write_pipe(p, rid, 2, ptr);
  // CHECK: call spir_func void @__commit_write_pipe(target("spirv.Pipe", 1) %{{.*}}, target("spirv.ReserveId") %{{.*}}, i32 4, i32 4)
  commit_write_pipe(p, rid);
}

void test3(read_only pipe int p, global int *ptr) {
  // CHECK: call spir_func target("spirv.ReserveId") @__work_group_reserve_read_pipe(target("spirv.Pipe", 0) %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = work_group_reserve_read_pipe(p, 2);
  // CHECK: call spir_func void @__work_group_commit_read_pipe(target("spirv.Pipe", 0) %{{.*}}, target("spirv.ReserveId") %{{.*}}, i32 4, i32 4)
  work_group_commit_read_pipe(p, rid);
}

void test4(write_only pipe int p, global int *ptr) {
  // CHECK: call spir_func target("spirv.ReserveId") @__work_group_reserve_write_pipe(target("spirv.Pipe", 1) %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = work_group_reserve_write_pipe(p, 2);
  // CHECK: call spir_func void @__work_group_commit_write_pipe(target("spirv.Pipe", 1) %{{.*}}, target("spirv.ReserveId") %{{.*}}, i32 4, i32 4)
  work_group_commit_write_pipe(p, rid);
}

void test5(read_only pipe int p, global int *ptr) {
  // CHECK: call spir_func target("spirv.ReserveId") @__sub_group_reserve_read_pipe(target("spirv.Pipe", 0) %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = sub_group_reserve_read_pipe(p, 2);
  // CHECK: call spir_func void @__sub_group_commit_read_pipe(target("spirv.Pipe", 0) %{{.*}}, target("spirv.ReserveId") %{{.*}}, i32 4, i32 4)
  sub_group_commit_read_pipe(p, rid);
}

void test6(write_only pipe int p, global int *ptr) {
  // CHECK: call spir_func target("spirv.ReserveId") @__sub_group_reserve_write_pipe(target("spirv.Pipe", 1) %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = sub_group_reserve_write_pipe(p, 2);
  // CHECK: call spir_func void @__sub_group_commit_write_pipe(target("spirv.Pipe", 1) %{{.*}}, target("spirv.ReserveId") %{{.*}}, i32 4, i32 4)
  sub_group_commit_write_pipe(p, rid);
}

void test7(read_only pipe int p, global int *ptr) {
  // CHECK: call spir_func i32 @__get_pipe_num_packets_ro(target("spirv.Pipe", 0) %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_num_packets(p);
  // CHECK: call spir_func i32 @__get_pipe_max_packets_ro(target("spirv.Pipe", 0) %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_max_packets(p);
}

void test8(write_only pipe int p, global int *ptr) {
  // CHECK: call spir_func i32 @__get_pipe_num_packets_wo(target("spirv.Pipe", 1) %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_num_packets(p);
  // CHECK: call spir_func i32 @__get_pipe_max_packets_wo(target("spirv.Pipe", 1) %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_max_packets(p);
}
