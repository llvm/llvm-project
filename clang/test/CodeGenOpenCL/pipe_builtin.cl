// RUN: %clang_cc1 -emit-llvm -O0 -cl-std=CL2.0 -o - %s | FileCheck %s

// CHECK: %opencl.pipe_t = type opaque
// CHECK: %opencl.reserve_id_t = type opaque

void test1(read_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__read_pipe_2(%opencl.pipe_t* %{{.*}}, i8* %{{.*}})
  read_pipe(p, ptr);
  // CHECK: call %opencl.reserve_id_t* @__reserve_read_pipe(%opencl.pipe_t* %{{.*}}, i32 {{.*}})
  reserve_id_t rid = reserve_read_pipe(p, 2);
  // CHECK: call i32 @__read_pipe_4(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 {{.*}}, i8* %{{.*}})
  read_pipe(p, rid, 2, ptr);
  // CHECK: call void @__commit_read_pipe(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}})
  commit_read_pipe(p, rid);
}

void test2(write_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__write_pipe_2(%opencl.pipe_t* %{{.*}}, i8* %{{.*}})
  write_pipe(p, ptr);
  // CHECK: call %opencl.reserve_id_t* @__reserve_write_pipe(%opencl.pipe_t* %{{.*}}, i32 {{.*}})
  reserve_id_t rid = reserve_write_pipe(p, 2);
  // CHECK: call i32 @__write_pipe_4(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}}, i32 {{.*}}, i8* %{{.*}})
  write_pipe(p, rid, 2, ptr);
  // CHECK: call void @__commit_write_pipe(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}})
  commit_write_pipe(p, rid);
}

void test3(read_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__work_group_reserve_read_pipe(%opencl.pipe_t* %{{.*}}, i32 {{.*}})
  reserve_id_t rid = work_group_reserve_read_pipe(p, 2);
  // CHECK: call void @__work_group_commit_read_pipe(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}})
  work_group_commit_read_pipe(p, rid);
}

void test4(write_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__work_group_reserve_write_pipe(%opencl.pipe_t* %{{.*}}, i32 {{.*}})
  reserve_id_t rid = work_group_reserve_write_pipe(p, 2);
  // CHECK: call void @__work_group_commit_write_pipe(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}})
  work_group_commit_write_pipe(p, rid);
}

void test5(read_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__sub_group_reserve_read_pipe(%opencl.pipe_t* %{{.*}}, i32 {{.*}})
  reserve_id_t rid = sub_group_reserve_read_pipe(p, 2);
  // CHECK: call void @__sub_group_commit_read_pipe(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}})
  sub_group_commit_read_pipe(p, rid);
}

void test6(write_only pipe int p, global int *ptr) {
  // CHECK: call %opencl.reserve_id_t* @__sub_group_reserve_write_pipe(%opencl.pipe_t* %{{.*}}, i32 {{.*}})
  reserve_id_t rid = sub_group_reserve_write_pipe(p, 2);
  // CHECK: call void @__sub_group_commit_write_pipe(%opencl.pipe_t* %{{.*}}, %opencl.reserve_id_t* %{{.*}})
  sub_group_commit_write_pipe(p, rid);
}

void test7(write_only pipe int p, global int *ptr) {
  // CHECK: call i32 @__get_pipe_num_packets(%opencl.pipe_t* %{{.*}})
  *ptr = get_pipe_num_packets(p);
  // CHECK: call i32 @__get_pipe_max_packets(%opencl.pipe_t* %{{.*}})
  *ptr = get_pipe_max_packets(p);
}

void test8(read_only pipe int r, write_only pipe int w, global int *ptr) {
  // verify that return type is correctly casted to i1 value
  // CHECK: %[[R:[0-9]+]] = call i32 @__read_pipe_2
  // CHECK: icmp ne i32 %[[R]], 0
  if (read_pipe(r, ptr)) *ptr = -1;
  // CHECK: %[[W:[0-9]+]] = call i32 @__write_pipe_2
  // CHECK: icmp ne i32 %[[W]], 0
  if (write_pipe(w, ptr)) *ptr = -1;
  // CHECK: %[[N:[0-9]+]] = call i32 @__get_pipe_num_packets
  // CHECK: icmp ne i32 %[[N]], 0
  if (get_pipe_num_packets(r)) *ptr = -1;
  // CHECK: %[[M:[0-9]+]] = call i32 @__get_pipe_max_packets
  // CHECK: icmp ne i32 %[[M]], 0
  if (get_pipe_max_packets(w)) *ptr = -1;
}
