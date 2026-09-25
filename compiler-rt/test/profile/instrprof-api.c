// Testing profile generate.
// RUN: %clang_profgen %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFGEN
// RUN: %clang_pgogen %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFGEN

// Testing profile use. Generate some profile file first.
// RUN: rm -rf rawprof.profraw
// RUN: %clang_profgen -o %t1 %s
// RUN: %run %t1
// RUN: llvm-profdata merge -o %t1.profdata rawprof.profraw
// RUN: %clang_profuse=%t1.profdata %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFUSE
// RUN: rm -rf rawprof.profraw
// RUN: %clang_pgogen -o %t2 %s
// RUN: %run %t2
// RUN: llvm-profdata merge -o %t2.profdata rawprof.profraw
// RUN: %clang_pgouse=%t2.profdata %s -S -emit-llvm -o - | FileCheck %s --check-prefix=PROFUSE
#include "profile/instr_prof_interface.h"

__attribute__((noinline)) int bar() { return 4; }

int foo() {
  __llvm_profile_reset_counters();
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_reset_counters()
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_reset_counters()
  return bar();
}

// PROFUSE-NOT: declare {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_reset_counters()

int other_apis(char *buf, uint64_t size) {
  __llvm_profile_initialize_file();
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_initialize_file()
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_initialize_file()
  if (__llvm_profile_write_file())
    return 1;
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_write_file()
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_write_file()
  const char *filename = __llvm_profile_get_filename();
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}ptr @__llvm_profile_get_filename()
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}ptr @__llvm_profile_get_filename()
  uint64_t buf_size = __llvm_profile_get_size_for_buffer();
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}i64 @__llvm_profile_get_size_for_buffer()
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}i64 @__llvm_profile_get_size_for_buffer()
  if (__llvm_profile_write_buffer(buf))
    return 2;
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_write_buffer(ptr noundef %{{.*}})
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_write_buffer(ptr noundef %{{.*}})
  if (__llvm_profile_check_compatibility(buf, size))
    return 3;
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_check_compatibility(ptr noundef %{{.*}}, i64 noundef %{{.*}})
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_check_compatibility(ptr noundef %{{.*}}, i64 noundef %{{.*}})
  if (__llvm_profile_merge_from_buffer(buf, size))
    return 4;
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_merge_from_buffer(ptr noundef %{{.*}}, i64 noundef %{{.*}})
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_merge_from_buffer(ptr noundef %{{.*}}, i64 noundef %{{.*}})
  return filename != 0 || buf_size != 0;
}

int main() {
  int z = foo() + 3;
  __llvm_profile_set_filename("rawprof.profraw");
  // PROFGEN: call {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_set_filename(ptr noundef @{{.*}})
  // PROFUSE-NOT: call {{(arm_aapcs_vfpcc )?}}void @__llvm_profile_set_filename(ptr noundef @{{.*}})
  if (__llvm_profile_dump())
    return 2;
  // PROFGEN: %{{.*}} = call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_dump()
  // PROFUSE-NOT: %{{.*}} = call {{(arm_aapcs_vfpcc )?}}{{(signext )*}}i32 @__llvm_profile_dump()
  return z + bar() - 11;
}

// PROFUSE-NOT: declare void @__llvm_profile_initialize_file()
// PROFUSE-NOT: declare signext i32 @__llvm_profile_write_file()
// PROFUSE-NOT: declare ptr @__llvm_profile_get_filename()
// PROFUSE-NOT: declare i64 @__llvm_profile_get_size_for_buffer()
// PROFUSE-NOT: declare signext i32 @__llvm_profile_write_buffer(ptr noundef)
// PROFUSE-NOT: declare signext i32 @__llvm_profile_check_compatibility(ptr noundef, i64 noundef)
// PROFUSE-NOT: declare signext i32 @__llvm_profile_merge_from_buffer(ptr noundef, i64 noundef)
// PROFUSE-NOT: declare void @__llvm_profile_set_filename(ptr noundef)
// PROFUSE-NOT: declare signext i32 @__llvm_profile_dump()
