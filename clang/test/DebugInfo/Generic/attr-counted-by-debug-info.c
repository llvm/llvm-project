// RUN: %clang -emit-llvm -DCOUNTED_BY -S -g %s -o - | FileCheck %s
// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

#ifdef COUNTED_BY
#define __counted_by(member)    __attribute__((__counted_by__(member)))
#else
#define __counted_by(member)
#endif

struct {
  int num_counters;
  long value[] __counted_by(num_counters);
} agent_send_response_port_num;

// CHECK: !DICompositeType(tag: DW_TAG_array_type, baseType: ![[BT:.*]], elements: ![[ELEMENTS:.*]])
// CHECK: ![[BT]] = !DIBasicType(name: "long", size: {{.*}}, encoding: DW_ATE_signed)
// CHECK: ![[ELEMENTS]] = !{![[COUNT:.*]]}
// CHECK: ![[COUNT]] = !DISubrange(count: -1)