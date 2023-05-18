// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -std=c++11 -fblocks -S -o - -emit-llvm %s | FileCheck %s -check-prefix CHECK-X86
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -std=c++11 -fblocks -S -o - -emit-llvm %s | FileCheck %s -check-prefix CHECK-X64

extern int e(void);

void (^b)() = ^{
  static int i = 0;
};

// CHECK-X86-DAG: @"?i@?1??_block_invoke@@YAXPAU__block_literal@@@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?1??_block_invoke@@YAXPEAU__block_literal@@@Z@4HA" ={{.*}} global i32 0

void f(void) {
  static int i = 0;
  ^{ static int i = e(); }();

// CHECK-X86-DAG: @"?i@?1??_block_invoke_1@@YAXPAU__block_literal_1@@@Z?1??f@@YAXXZ@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?1??_block_invoke_1@@YAXPEAU__block_literal_1@@@Z?1??f@@YAXXZ@4HA" ={{.*}} global i32 0

  ^{ static int i = e(); }();

// CHECK-X86-DAG: @"?i@?1??_block_invoke_2@@YAXPAU__block_literal_2@@@Z?1??f@@YAXXZ@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?1??_block_invoke_2@@YAXPEAU__block_literal_2@@@Z?1??f@@YAXXZ@4HA" ={{.*}} global i32 0

  ^{ ^{ static int i = e(); }(); }();

// CHECK-X86-DAG: @"?i@?1??_block_invoke_3@@YAXPAU__block_literal_3@@@Z?1??_block_invoke_4@@YAXPAU__block_literal_4@@@Z?1??f@@YAXXZ@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?1??_block_invoke_3@@YAXPEAU__block_literal_3@@@Z?1??_block_invoke_4@@YAXPEAU__block_literal_4@@@Z?1??f@@YAXXZ@4HA" ={{.*}} global i32 0
}


template <typename>
void g(void) {
  ^{ static int i = e(); }();
}

template void g<char>(void);

// CHECK-X86-DAG: @"?i@?2??_block_invoke_1@@YAXPAU__block_literal_1@@@Z?2???$g@D@@YAXXZ@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?2??_block_invoke_1@@YAXPEAU__block_literal_1@@@Z?2???$g@D@@YAXXZ@4HA" ={{.*}} global i32 0

template void g<int>(void);

// CHECK-X86-DAG: @"?i@?2??_block_invoke_1@@YAXPAU__block_literal_1@@@Z?2???$g@H@@YAXXZ@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?2??_block_invoke_1@@YAXPEAU__block_literal_1@@@Z?2???$g@H@@YAXXZ@4HA" ={{.*}} global i32 0

inline void h(void) {
  ^{ static int i = e(); }();
}

// CHECK-X86-DAG: @"?i@?2??_block_invoke_1@@YAXPAU__block_literal_1@@@Z?2??h@@YAXXZ@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?2??_block_invoke_1@@YAXPEAU__block_literal_1@@@Z?2??h@@YAXXZ@4HA" ={{.*}} global i32 0

struct s {
  int i = ^{ static int i = e(); return ++i; }();

// CHECK-X86-DAG: @"?i@?0??_block_invoke_1@0s@@YAXPAU__block_literal_1@@@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1@0s@@YAXPEAU__block_literal_1@@@Z@4HA" ={{.*}} global i32 0

  int j = ^{ static int i = e(); return ++i; }();

// CHECK-X86-DAG: @"?i@?0??_block_invoke_1@j@s@@YAXPAU__block_literal_1@@@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1@j@s@@YAXPEAU__block_literal_1@@@Z@4HA" ={{.*}} global i32 0

  void m(int i = ^{ static int i = e(); return ++i; }(),
         int j = ^{ static int i = e(); return ++i; }()) {}

// CHECK-X86-DAG: @"?i@?0??_block_invoke_1_1@@YAXPAU__block_literal_1_1@@@Z?0??m@s@@QAEXHH@Z@4HA" ={{.*}} global i32 0
// CHECK-X86-DAG: @"?i@?0??_block_invoke_1_2@@YAXPAU__block_literal_1_2@@@Z?0??m@s@@QAEXHH@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1_1@@YAXPEAU__block_literal_1_1@@@Z?0??m@s@@QEAAXHH@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1_2@@YAXPEAU__block_literal_1_2@@@Z?0??m@s@@QEAAXHH@Z@4HA" ={{.*}} global i32 0

  void n(int = ^{ static int i = e(); return ++i; }(),
         int = ^{ static int i = e(); return ++i; }()) {}

// CHECK-X86-DAG: @"?i@?0??_block_invoke_1_1@@YAXPAU__block_literal_1_1@@@Z?0??n@s@@QAEXHH@Z@4HA" ={{.*}} global i32 0
// CHECK-X86-DAG: @"?i@?0??_block_invoke_1_2@@YAXPAU__block_literal_1_2@@@Z?0??n@s@@QAEXHH@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1_1@@YAXPEAU__block_literal_1_1@@@Z?0??n@s@@QEAAXHH@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1_2@@YAXPEAU__block_literal_1_2@@@Z?0??n@s@@QEAAXHH@Z@4HA" ={{.*}} global i32 0

};

struct t {
  struct u {
    int i = ^{ static int i = e(); return ++i; }();

// CHECK-X86-DAG: @"?i@?0??_block_invoke_1@0u@t@@YAXPAU__block_literal_1@@@Z@4HA" ={{.*}} global i32 0
// CHECK-X64-DAG: @"?i@?0??_block_invoke_1@0u@t@@YAXPEAU__block_literal_1@@@Z@4HA" ={{.*}} global i32 0

  };
};

void j(void) {
  h();
  struct s s;
  s.m();
  s.n();
  struct t::u t;
}

#if 0
template <typename T>
struct v {
  static T i;
};

template <typename T>
T v<T>::i = ^{ static T i = T(); return i; }();

template class v<char>;
template class v<int>;
#endif

