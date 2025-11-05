// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm \
// RUN:   -debug-info-kind=standalone -std=c++26 %s -o - | FileCheck %s


// CHECK: ![[PACK1:[0-9]+]] = distinct !DISubprogram(name: "capture_pack<int>"
// CHECK: ![[PACK2:[0-9]+]] = distinct !DISubprogram(name: "capture_pack<int, int>"
// CHECK: ![[PACK3:[0-9]+]] = distinct !DISubprogram(name: "capture_pack_and_locals<int>"
// CHECK: ![[PACK4:[0-9]+]] = distinct !DISubprogram(name: "capture_pack_and_locals<int, int>"
// CHECK: ![[PACK5:[0-9]+]] = distinct !DISubprogram(name: "capture_pack_and_this<int>"
// CHECK: ![[PACK6:[0-9]+]] = distinct !DISubprogram(name: "capture_pack_and_this<int, int>"
// CHECK: ![[PACK7:[0-9]+]] = distinct !DISubprogram(name: "capture_binding_and_param_pack<int, int>"

template<typename... Args>
auto capture_pack(Args... args) {
  return [args..., ...params = args] {
    return 0;
  }();
}

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK1]]
// CHECK-SAME:                           elements: ![[PACK1_ELEMS:[0-9]+]]
// CHECK-DAG:  ![[PACK1_ELEMS]] = !{![[PACK1_ARGS:[0-9]+]], ![[PACK1_PARAMS:[0-9]+]]}
// CHECK-DAG:  ![[PACK1_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK1_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK2]]
// CHECK-SAME:                           elements: ![[PACK2_ELEMS:[0-9]+]]
// CHECK:      ![[PACK2_ELEMS]] = !{![[PACK2_ARGS:[0-9]+]]
// CHECK-SAME:                      ![[PACK2_ARGS]]
// CHECK-SAME:                      ![[PACK2_PARAMS:[0-9]+]]
// CHECK-SAME:                      ![[PACK2_PARAMS]]}
// CHECK-DAG:  ![[PACK2_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK2_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-NOT:  DW_TAG_member

template<typename... Args>
auto capture_pack_and_locals(int x, Args... args) {
  int w = 0;
  return [=, &args..., &x, ...params = args] {
    return w;
  }();
}

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK3]]
// CHECK-SAME:                           elements: ![[PACK3_ELEMS:[0-9]+]]
// CHECK:      ![[PACK3_ELEMS]] = !{![[PACK3_ARGS:[0-9]+]]
// CHECK-SAME:                      ![[PACK3_X:[0-9]+]]
// CHECK-SAME:                      ![[PACK3_PARAMS:[0-9]+]]
// CHECK-SAME:                      ![[PACK3_W:[0-9]+]]}
// CHECK-DAG:  ![[PACK3_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  !DIDerivedType(tag: DW_TAG_reference_type
// CHECK-DAG:  ![[PACK3_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-DAG:  ![[PACK3_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-DAG:  ![[PACK3_W]] = !DIDerivedType(tag: DW_TAG_member, name: "w"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK4]]
// CHECK-SAME:                           elements: ![[PACK4_ELEMS:[0-9]+]]
// CHECK:      ![[PACK4_ELEMS]] = !{![[PACK4_ARGS:[0-9]+]]
// CHECK-SAME:                      ![[PACK4_ARGS]]
// CHECK-SAME:                      ![[PACK4_X:[0-9]+]]
// CHECK-SAME:                      ![[PACK4_PARAMS:[0-9]+]]
// CHECK-SAME:                      ![[PACK4_PARAMS]]
// CHECK-SAME:                      ![[PACK4_W:[0-9]+]]}
// CHECK-DAG:  ![[PACK4_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK4_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-DAG:  ![[PACK4_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-DAG:  ![[PACK4_W]] = !DIDerivedType(tag: DW_TAG_member, name: "w"
// CHECK-NOT:  DW_TAG_member

struct Foo {
  template<typename... Args>
  auto capture_pack_and_this(Args... args) {
    auto val1 = [this, args..., ...params = args] {
      return w;
    }();

    auto val2 = [args..., this, ...params = args] {
      return w;
    }();

    auto val3 = [args..., ...params = args, this] {
      return w;
    }();

    return val1 + val2 + val3;
  }

  int w = 10;
} f;

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK5]]
// CHECK-SAME:                           elements: ![[PACK5a_ELEMS:[0-9]+]]
// CHECK:      ![[PACK5a_ELEMS]] = !{![[PACK5a_THIS:[0-9]+]]
// CHECK-SAME:                       ![[PACK5a_ARGS:[0-9]+]]
// CHECK-SAME:                       ![[PACK5a_PARAMS:[0-9]+]]}
// CHECK-DAG:  ![[PACK5a_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this"
// CHECK-DAG:  ![[PACK5a_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK5a_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK5]]
// CHECK-SAME:                           elements: ![[PACK5b_ELEMS:[0-9]+]]
// CHECK:      ![[PACK5b_ELEMS]] = !{![[PACK5b_ARGS:[0-9]+]]
// CHECK-SAME:                       ![[PACK5b_THIS:[0-9]+]]
// CHECK-SAME:                       ![[PACK5b_PARAMS:[0-9]+]]}
// CHECK-DAG:  ![[PACK5b_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK5b_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this"
// CHECK-DAG:  ![[PACK5b_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK5]]
// CHECK:                                elements: ![[PACK5c_ELEMS:[0-9]+]]
// CHECK-NEXT: ![[PACK5c_ELEMS]] = !{![[PACK5c_ARGS:[0-9]+]]
// CHECK-SAME:                       ![[PACK5c_PARAMS:[0-9]+]]
// CHECK-SAME:                       ![[PACK5c_THIS:[0-9]+]]}
// CHECK-DAG:  ![[PACK5c_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK5c_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-DAG:  ![[PACK5c_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK6]]
// CHECK-SAME:                           elements: ![[PACK6a_ELEMS:[0-9]+]]
// CHECK:      ![[PACK6a_ELEMS]] = !{![[PACK6a_THIS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6a_ARGS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6a_ARGS]]
// CHECK-SAME:                       ![[PACK6a_PARAMS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6a_PARAMS]]
// CHECK-DAG:  ![[PACK6a_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this"
// CHECK-DAG:  ![[PACK6a_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK6a_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK6]]
// CHECK-SAME:                           elements: ![[PACK6b_ELEMS:[0-9]+]]
// CHECK:      ![[PACK6b_ELEMS]] = !{![[PACK6b_ARGS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6b_ARGS]]
// CHECK-SAME:                       ![[PACK6b_THIS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6b_PARAMS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6b_PARAMS]]}
// CHECK-DAG:  ![[PACK6b_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK6b_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this"
// CHECK-DAG:  ![[PACK6b_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-NOT:  DW_TAG_member

// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK6]]
// CHECK-SAME:                           elements: ![[PACK6c_ELEMS:[0-9]+]]
// CHECK:      ![[PACK6c_ELEMS]] = !{![[PACK6c_ARGS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6c_ARGS]]
// CHECK-SAME:                       ![[PACK6c_PARAMS:[0-9]+]]
// CHECK-SAME:                       ![[PACK6c_PARAMS]]
// CHECK-SAME:                       ![[PACK6c_THIS:[0-9]+]]}
// CHECK-DAG:  ![[PACK6c_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK6c_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-DAG:  ![[PACK6c_THIS]] = !DIDerivedType(tag: DW_TAG_member, name: "this"
// CHECK-NOT:  DW_TAG_member

template<typename... Args>
auto capture_binding_and_param_pack(Args... args) {
  struct C { int x = 2; int y = 3; };

  auto [x, ...e] = C();

  return [&, args..., x, ...params = args,
          ...es = e] {
    return e...[0] + es...[0];
  }();
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "C"
// CHECK:      distinct !DICompositeType(tag: DW_TAG_class_type, scope: ![[PACK7]]
// CHECK-SAME:                           elements: ![[PACK7_ELEMS:[0-9]+]]
// CHECK:      ![[PACK7_ELEMS]] = !{![[PACK7_ARGS:[0-9]+]]
// CHECK-SAME:                      ![[PACK7_ARGS]]
// CHECK-SAME:                      ![[PACK7_X:[0-9]+]]
// CHECK-SAME:                      ![[PACK7_PARAMS:[0-9]+]]
// CHECK-SAME:                      ![[PACK7_PARAMS]]
// CHECK-SAME:                      ![[PACK7_ES:[0-9]+]]
// CHECK-SAME:                      ![[PACK7_E:[0-9]+]]}
// CHECK-DAG:  ![[PACK7_ARGS]] = !DIDerivedType(tag: DW_TAG_member, name: "args"
// CHECK-DAG:  ![[PACK7_X]] = !DIDerivedType(tag: DW_TAG_member, name: "x"
// CHECK-DAG:  ![[PACK7_PARAMS]] = !DIDerivedType(tag: DW_TAG_member, name: "params"
// CHECK-DAG:  ![[PACK7_ES]] = !DIDerivedType(tag: DW_TAG_member, name: "es"
// CHECK-DAG:  ![[PACK7_E]] = !DIDerivedType(tag: DW_TAG_member, name: "e"
// CHECK-NOT:  DW_TAG_member

int main() {
  return capture_pack(1)
         + capture_pack(1, 2)
         + capture_pack_and_locals(1, 2)
         + capture_pack_and_locals(1, 2, 3)
         + f.capture_pack_and_this(1)
         + f.capture_pack_and_this(1, 2)
         + capture_binding_and_param_pack(1, 2);
}
