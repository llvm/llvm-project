// RUN: %clang_cc1 -std=c++23 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg.declare"

// CHECK: define {{.*}} i32 @_Z1fv
// CHECK: #dbg_declare(ptr %{{[a-z]+}}, ![[VAR_0:[0-9]+]], !DIExpression(),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_1:[0-9]+]], !DIExpression(),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_2:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 4),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_3:[0-9]+]], !DIExpression(DW_OP_deref),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_4:[0-9]+]], !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4),
// CHECK: #dbg_declare(ptr %z1, ![[VAR_5:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %z2, ![[VAR_6:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %k, ![[VAR_7:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %v{{[0-9]*}}, ![[VAR_8:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %w{{[0-9]*}}, ![[VAR_9:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %m, ![[VAR_10:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %n, ![[VAR_11:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %s, ![[VAR_12:[0-9]+]], !DIExpression()
// CHECK: #dbg_declare(ptr %p, ![[VAR_13:[0-9]+]], !DIExpression()
// CHECK: getelementptr inbounds nuw %struct.A, ptr {{.*}}, i32 0, i32 1, !dbg ![[Y1_DEBUG_LOC:[0-9]+]]
// CHECK: getelementptr inbounds nuw %struct.A, ptr {{.*}}, i32 0, i32 1, !dbg ![[Y2_DEBUG_LOC:[0-9]+]]
// CHECK: load ptr, ptr %z2, {{.*}}!dbg ![[Z2_DEBUG_LOC:[0-9]+]]
// CHECK: getelementptr inbounds [2 x i32], ptr {{.*}}, i{{64|32}} 0, i{{64|32}} 1, !dbg ![[A2_DEBUG_LOC:[0-9]+]]
// CHECK: getelementptr inbounds nuw { i32, i32 }, ptr {{.*}}, i32 0, i32 1, !dbg ![[C2_DEBUG_LOC:[0-9]+]]
// CHECK: extractelement <2 x i32> {{.*}}, i32 1, !dbg ![[V2_DEBUG_LOC:[0-9]+]]
// CHECK: ![[VAR_0]] = !DILocalVariable(name: "a"
// CHECK: ![[VAR_1]] = !DILocalVariable(name: "x1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_2]] = !DILocalVariable(name: "y1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_3]] = !DILocalVariable(name: "x2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_4]] = !DILocalVariable(name: "y2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_5]] = !DILocalVariable(name: "z1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_6]] = !DILocalVariable(name: "z2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_7]] = !DILocalVariable(name: "k", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_8]] = !DILocalVariable(name: "v", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_9]] = !DILocalVariable(name: "w", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_10]] = !DILocalVariable(name: "m", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_11]] = !DILocalVariable(name: "n", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_12]] = !DILocalVariable(name: "s", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})
// CHECK: ![[VAR_13]] = !DILocalVariable(name: "p", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: !{{[0-9]+}})

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

struct C {
  int w;
  int z;
  template<int> int get(this C&& self);
  template<> int get<0>(this C&& self) { return self.w; }
  template<> int get<1>(this C&& self) { return self.z; }
};

struct D {
  int w;
  int z;
  template<int> int get(int unused = 0);
  template<> int get<0>(int unused) { return w; }
  template<> int get<1>(int unused) { return z; }
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

template<>
struct tuple_size<C> {
    static constexpr unsigned value = 2;
};

template<>
struct tuple_size<D> {
    static constexpr unsigned value = 2;
};

template<unsigned, typename T> struct tuple_element { using type = int; };

// Decomposition of tuple-like bindings but where the `get` methods
// are declared as free functions.
struct triple {
  int k;
  int v;
  int w;
};

template<>
struct tuple_size<triple> {
    static constexpr unsigned value = 3;
};

template <unsigned I> int get(triple);
template <> int get<0>(triple p) { return p.k; }
template <> int get<1>(triple p) { return p.v; }
template <> int get<2>(triple p) { return p.w; }
} // namespace std

int f() {
  A a{10, 20};
  auto [x1, y1] = a;
  auto &[x2, y2] = a;
  auto [z1, z2] = B{1, 2};
  int array[2] = {3, 4};
  auto &[a1, a2] = array;
  _Complex int cmplx = {1, 2};
  auto &[c1, c2] = cmplx;
  int vctr __attribute__ ((vector_size (sizeof(int)*2)))= {1, 2};
  auto &[v1, v2] = vctr;
  auto [k, v, w] = std::triple{3, 4, 5};
  auto [m, n] = C{2, 3};
  auto [s, p] = D{2, 3};
  return //
     x1 //
     +  //
// CHECK: ![[Y1_DEBUG_LOC]] = !DILocation(line: [[@LINE+1]]
     y1 //
     +  //
     x2 //
     +  //
// CHECK: ![[Y2_DEBUG_LOC]] = !DILocation(line: [[@LINE+1]]
     y2 //
     +  //
     z1 //
     +  //
// CHECK: ![[Z2_DEBUG_LOC]] = !DILocation(line: [[@LINE+1]]
     z2 //
     +  //
     a1 //
     +  //
// CHECK: ![[A2_DEBUG_LOC]] = !DILocation(line: [[@LINE+1]]
     a2 //
     +  //
     c1 //
     +  //
// CHECK: ![[C2_DEBUG_LOC]] = !DILocation(line: [[@LINE+1]]
     c2 //
     +  //
     v1 //
     +  //
// CHECK: ![[V2_DEBUG_LOC]] = !DILocation(line: [[@LINE+1]]
     v2 //
     ;
}
