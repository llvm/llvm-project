// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg.declare"

// CHECK: #dbg_declare(ptr %{{[a-z]+}}, ![[VAR_0:[0-9]+]], !DIExpression(),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_1:[0-9]+]], !DIExpression(),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_2:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 4),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_3:[0-9]+]], !DIExpression(DW_OP_deref),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_4:[0-9]+]], !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4),
// CHECK: #dbg_declare(ptr %z1, ![[VAR_5:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %z2, ![[VAR_6:[0-9]+]], !DIExpression()
// CHECK: ![[VAR_0]] = !DILocalVariable(name: "a"
// CHECK: ![[VAR_1]] = !DILocalVariable(name: "x1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_2]] = !DILocalVariable(name: "y1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_3]] = !DILocalVariable(name: "x2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_4]] = !DILocalVariable(name: "y2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_5]] = !DILocalVariable(name: "z1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_6]] = !DILocalVariable(name: "z2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})

struct A {
  int x;
  int y;
};

struct B {
  int w;
  int z;
  template<int> int get();
  template<> int get<0>() { return w; }
  template<> int get<1>() { return z; }
};

// Note: the following declarations are necessary for decomposition of tuple-like
// structured bindings
namespace std {
template<typename T> struct tuple_size {
};
template<>
struct tuple_size<B> {
    static constexpr unsigned value = 2;
};

template<unsigned, typename T> struct tuple_element { using type = int; };
} // namespace std

int f() {
  A a{10, 20};
  auto [x1, y1] = a;
  auto &[x2, y2] = a;
  auto [z1, z2] = B{1, 2};
  return x1 + y1 + x2 + y2 + z1 + z2;
}
