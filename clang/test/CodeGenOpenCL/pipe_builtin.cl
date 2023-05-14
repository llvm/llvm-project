// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -cl-ext=+cl_khr_subgroups -O0 -cl-std=clc++ -o - %s | FileCheck %s
// FIXME: Add MS ABI manglings of OpenCL things and remove %itanium_abi_triple
// above to support OpenCL in the MS C++ ABI.

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

void test1(read_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__read_pipe_2(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  read_pipe(p, ptr);
  // CHECK: call ptr @__reserve_read_pipe(ptr %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = reserve_read_pipe(p, 2);
  // CHECK: call i32 @__read_pipe_4(ptr %{{.*}}, ptr %{{.*}}, i32 {{.*}}, ptr %{{.*}}, i32 4, i32 4)
  read_pipe(p, rid, 2, ptr);
  // CHECK: call void @__commit_read_pipe(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  commit_read_pipe(p, rid);
}

void test2(write_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__write_pipe_2(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  write_pipe(p, ptr);
  // CHECK: call ptr @__reserve_write_pipe(ptr %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = reserve_write_pipe(p, 2);
  // CHECK: call i32 @__write_pipe_4(ptr %{{.*}}, ptr %{{.*}}, i32 {{.*}}, ptr %{{.*}}, i32 4, i32 4)
  write_pipe(p, rid, 2, ptr);
  // CHECK: call void @__commit_write_pipe(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  commit_write_pipe(p, rid);
}

void test3(read_only pipe int p, global int *ptr) {
  // CHECK: call ptr @__work_group_reserve_read_pipe(ptr %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = work_group_reserve_read_pipe(p, 2);
  // CHECK: call void @__work_group_commit_read_pipe(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  work_group_commit_read_pipe(p, rid);
}

void test4(write_only pipe int p, global int *ptr) {
  // CHECK: call ptr @__work_group_reserve_write_pipe(ptr %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = work_group_reserve_write_pipe(p, 2);
  // CHECK: call void @__work_group_commit_write_pipe(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  work_group_commit_write_pipe(p, rid);
}

void test5(read_only pipe int p, global int *ptr) {
  // CHECK: call ptr @__sub_group_reserve_read_pipe(ptr %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = sub_group_reserve_read_pipe(p, 2);
  // CHECK: call void @__sub_group_commit_read_pipe(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  sub_group_commit_read_pipe(p, rid);
}

void test6(write_only pipe int p, global int *ptr) {
  // CHECK: call ptr @__sub_group_reserve_write_pipe(ptr %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = sub_group_reserve_write_pipe(p, 2);
  // CHECK: call void @__sub_group_commit_write_pipe(ptr %{{.*}}, ptr %{{.*}}, i32 4, i32 4)
  sub_group_commit_write_pipe(p, rid);
}

void test7(read_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__get_pipe_num_packets_ro(ptr %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_num_packets(p);
  // CHECK: call i32 @__get_pipe_max_packets_ro(ptr %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_max_packets(p);
}

void test8(write_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__get_pipe_num_packets_wo(ptr %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_num_packets(p);
  // CHECK: call i32 @__get_pipe_max_packets_wo(ptr %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_max_packets(p);
}
