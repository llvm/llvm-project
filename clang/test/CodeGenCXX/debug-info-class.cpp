struct foo;
void func(foo *f) {
}
class bar;
void func(bar *f) {
}
union baz;
void func(baz *f) {
}

class B {
public:
  virtual ~B();
};

B::~B() {
}

struct C {
  static int s;
  virtual ~C();
};

C::~C() {
}

struct D {
  D();
  virtual ~D();
  void func() {
  }
};

struct E {
  E();
  virtual ~E();
  virtual void func() {
  }
};

struct F {
  struct inner {
  };
  static const int i = 2;
  virtual ~F();
};

struct G {
  virtual void func();
  struct inner {
    int j;
  };
};

struct H {};
struct I : virtual H {};
struct J : I {};
J j;

struct A {
  int one;
  static const int HdrSize = 52;
  int two;
  A() {
    int x = 1;
  }
};

void f1() {
  D x;
  x.func();
  E y;
  int i = F::i;
  F::inner z;
}

int main(int argc, char **argv) {
  B b;
  G::inner c_i;
  if (argc) {
    A a;
  }
  return 0;
}

// RUN: %clang -target x86_64-unknown_unknown -emit-llvm -g -S %s -o - | FileCheck %s
// RUN: %clang -target i686-cygwin -emit-llvm -g -S %s -o - | FileCheck %s
// RUN: %clang -target armv7l-unknown-linux-gnueabihf -emit-llvm -g -S %s -o - | FileCheck %s

// CHECK: invoke {{.+}} @_ZN1BD1Ev(%class.B* %b)
// CHECK-NEXT: unwind label %{{.+}}, !dbg ![[EXCEPTLOC:.*]]
// CHECK: store i32 0, i32* %{{.+}}, !dbg ![[RETLOC:.*]]
// CHECK: DW_TAG_structure_type ] [foo]
// CHECK: DW_TAG_class_type ] [bar]
// CHECK: DW_TAG_union_type ] [baz]
// CHECK: DW_TAG_class_type ] [B] {{.*}} [def]
// CHECK: !"0xd\00_vptr$B\00{{.*}}\0064", {{.*}} ; [ DW_TAG_member ]

// CHECK: [[C:![0-9]*]] = {{.*}} [[C_MEM:![0-9]*]], !"_ZTS1C", null, !"_ZTS1C"} ; [ DW_TAG_structure_type ] [C] {{.*}} [def]
// CHECK: [[C_MEM]] = !{[[C_VPTR:![0-9]*]], [[C_S:![0-9]*]], [[C_DTOR:![0-9]*]]}
// CHECK: [[C_VPTR]] = {{.*}} ; [ DW_TAG_member ] [_vptr$C] {{.*}} [artificial]
// CHECK: [[C_S]] = {{.*}} ; [ DW_TAG_member ] [s] {{.*}} [static] [from int]
// CHECK: [[C_DTOR]] = {{.*}} ; [ DW_TAG_subprogram ] {{.*}} [~C]

// CHECK: null, null, null, !"_ZTS1D"} ; [ DW_TAG_structure_type ] [D] {{.*}} [decl]
// CHECK: null, null, null, !"_ZTS1E"} ; [ DW_TAG_structure_type ] [E] {{.*}} [decl]
// CHECK: [[F:![0-9]*]] = {{.*}} null, null, null, !"_ZTS1F"} ; [ DW_TAG_structure_type ] [F] {{.*}} [decl]

// CHECK: null, null, null, !"_ZTS1G"} ; [ DW_TAG_structure_type ] [G] {{.*}} [decl]
// CHECK: [[G_INNER_MEM:![0-9]*]], null, null, !"_ZTSN1G5innerE"} ; [ DW_TAG_structure_type ] [inner] [line 50, {{.*}} [def]
// CHECK: [[G_INNER_MEM]] = !{[[G_INNER_I:![0-9]*]]}
// CHECK: [[G_INNER_I]] = {{.*}} ; [ DW_TAG_member ] [j] {{.*}} [from int]

// CHECK: ; [ DW_TAG_structure_type ] [A]
// CHECK: HdrSize
// CHECK: ; [ DW_TAG_structure_type ] [I] {{.*}} [def]
//
// CHECK: !"_ZTS1D", {{.*}}, [[D_FUNC_DECL:![0-9]*]], {{![0-9]*}}} ; [ DW_TAG_subprogram ] {{.*}} [def] [func]
// CHECK: [[D_FUNC_DECL]] = !{!"0x2e\00func\00{{.*}}\000\00{{[0-9]+}}"{{.*}}, !"_ZTS1D", {{.*}}, null} ; [ DW_TAG_subprogram ] {{.*}} [func]

// CHECK: ![[EXCEPTLOC]] = !MDLocation(line: 84,
// CHECK: ![[RETLOC]] = !MDLocation(line: 83,
