// RUN: %clang_cc1 -emit-llvm -cl-ext=+cl_khr_subgroups -O0 -cl-std=CL2.0 -o - %s | FileCheck %s

// CHECK-DAG: %opencl.pipe_ro_t = type opaque
// CHECK-DAG: %opencl.pipe_wo_t = type opaque
// CHECK-DAG: %opencl.reserve_id_t = type opaque

#pragma OPENCL EXTENSION cl_khr_subgroups : enable

void test1(read_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__read_pipe_2(%opencl.pipe_ro_t* %{{.*}}, i8* %{{.*}}, i32 4, i32 4)
  read_pipe(p, ptr);
  // CHECK: call %opencl.reserve_id_t* @__reserve_read_pipe(%opencl.pipe_ro_t* %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = reserve_read_pipe(p, 2);
  // CHECK: call i32 @__read_pipe_4(%opencl.pipe_ro_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 {{.*}}, i8* %{{.*}}, i32 4, i32 4)
  read_pipe(p, rid, 2, ptr);
  // CHECK: call void @__commit_read_pipe(%opencl.pipe_ro_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 4, i32 4)
  commit_read_pipe(p, rid);
}

void test2(write_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__write_pipe_2(%opencl.pipe_wo_t* %{{.*}}, i8* %{{.*}}, i32 4, i32 4)
  write_pipe(p, ptr);
  // CHECK: call %opencl.reserve_id_t* @__reserve_write_pipe(%opencl.pipe_wo_t* %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = reserve_write_pipe(p, 2);
  // CHECK: call i32 @__write_pipe_4(%opencl.pipe_wo_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 {{.*}}, i8* %{{.*}}, i32 4, i32 4)
  write_pipe(p, rid, 2, ptr);
  // CHECK: call void @__commit_write_pipe(%opencl.pipe_wo_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 4, i32 4)
  commit_write_pipe(p, rid);
}

void test3(read_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__work_group_reserve_read_pipe(%opencl.pipe_ro_t* %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = work_group_reserve_read_pipe(p, 2);
  // CHECK: call void @__work_group_commit_read_pipe(%opencl.pipe_ro_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 4, i32 4)
  work_group_commit_read_pipe(p, rid);
}

void test4(write_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__work_group_reserve_write_pipe(%opencl.pipe_wo_t* %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = work_group_reserve_write_pipe(p, 2);
  // CHECK: call void @__work_group_commit_write_pipe(%opencl.pipe_wo_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 4, i32 4)
  work_group_commit_write_pipe(p, rid);
}

void test5(read_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__sub_group_reserve_read_pipe(%opencl.pipe_ro_t* %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = sub_group_reserve_read_pipe(p, 2);
  // CHECK: call void @__sub_group_commit_read_pipe(%opencl.pipe_ro_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 4, i32 4)
  sub_group_commit_read_pipe(p, rid);
}

void test6(write_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__sub_group_reserve_write_pipe(%opencl.pipe_wo_t* %{{.*}}, i32 {{.*}}, i32 4, i32 4)
  reserve_id_t rid = sub_group_reserve_write_pipe(p, 2);
  // CHECK: call void @__sub_group_commit_write_pipe(%opencl.pipe_wo_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 4, i32 4)
  sub_group_commit_write_pipe(p, rid);
}

void test7(read_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__get_pipe_num_packets_ro(%opencl.pipe_ro_t* %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_num_packets(p);
  // CHECK: call i32 @__get_pipe_max_packets_ro(%opencl.pipe_ro_t* %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_max_packets(p);
}

void test8(write_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__get_pipe_num_packets_wo(%opencl.pipe_wo_t* %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_num_packets(p);
  // CHECK: call i32 @__get_pipe_max_packets_wo(%opencl.pipe_wo_t* %{{.*}}, i32 4, i32 4)
  *ptr = get_pipe_max_packets(p);
}

void test9(read_only pipe int r, write_only pipe int w, global int *ptr) {
  // verify that return type is correctly casted to i1 value
  // CHECK: %[[R:[0-9]+]] = call i32 @__read_pipe_2
  // CHECK: icmp ne i32 %[[R]], 0
  if (read_pipe(r, ptr)) *ptr = -1;
  // CHECK: %[[W:[0-9]+]] = call i32 @__write_pipe_2
  // CHECK: icmp ne i32 %[[W]], 0
  if (write_pipe(w, ptr)) *ptr = -1;
  // CHECK: %[[NR:[0-9]+]] = call i32 @__get_pipe_num_packets_ro
  // CHECK: icmp ne i32 %[[NR]], 0
  if (get_pipe_num_packets(r)) *ptr = -1;
  // CHECK: %[[NW:[0-9]+]] = call i32 @__get_pipe_num_packets_wo
  // CHECK: icmp ne i32 %[[NW]], 0
  if (get_pipe_num_packets(w)) *ptr = -1;
  // CHECK: %[[MR:[0-9]+]] = call i32 @__get_pipe_max_packets_ro
  // CHECK: icmp ne i32 %[[MR]], 0
  if (get_pipe_max_packets(r)) *ptr = -1;
  // CHECK: %[[MW:[0-9]+]] = call i32 @__get_pipe_max_packets_wo
  // CHECK: icmp ne i32 %[[MW]], 0
  if (get_pipe_max_packets(w)) *ptr = -1;
}
