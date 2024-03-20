// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -S -x c++ -std=c++11 -triple aarch64-linux-android31 \
// RUN:   -fsanitize=memtag-globals -o %t.out %s
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-A
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-B
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-C
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-D
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-E
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-F
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-G
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-H
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-I
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-J
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-K
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-L
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-M
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-N
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-O
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-P
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-Q
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-R

// RUN: %clang_cc1 -O3 -S -x c++ -std=c++11 -triple aarch64-linux-android31 \
// RUN:   -fsanitize=memtag-globals -o %t.out %s
// RUN: FileCheck %s --input-file=%t.out
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-A
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-B
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-C
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-D
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-E
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-F
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-G
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-H
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-I
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-J
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-K
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-L
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-M
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-N
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-O
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-P
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-Q
// RUN: FileCheck %s --input-file=%t.out --check-prefix=CHECK-R

/// Ensure that emulated TLS also doesn't get sanitized.
// RUN: %clang_cc1 -S -x c++ -std=c++11 -triple aarch64-linux-android31 \
// RUN:   -fsanitize=memtag-globals -o - %s | FileCheck %s

// CHECK-A: .memtag global_int
// CHECK-A: .globl global_int
// CHECK-A: .p2align 4, 0x0
// CHECK-A: .size global_int, 16
int global_int;
// CHECK-B: .memtag _ZL9local_int
// CHECK-B: .local _ZL9local_int
// CHECK-B: .comm _ZL9local_int,16,16
static int local_int;

// CHECK-C: .memtag _ZL12local_buffer
// CHECK-C: .local _ZL12local_buffer
// CHECK-C: .comm _ZL12local_buffer,16,16
static char local_buffer[16];
// CHECK-D: .memtag _ZL22local_buffer_local_end
// CHECK-D: .p2align 4, 0x0
// CHECK-D: _ZL22local_buffer_local_end:
// CHECK-D: .xword _ZL12local_buffer+16
// CHECK-D: .size _ZL22local_buffer_local_end, 16
static char* local_buffer_local_end = &local_buffer[16];
// CHECK-E: .memtag local_buffer_global_end
// CHECK-E: .globl local_buffer_global_end
// CHECK-E  .p2align 4, 0x0
// CHECK-E: local_buffer_global_end:
// CHECK-E: .xword _ZL12local_buffer+16
// CHECK-E: .size local_buffer_global_end, 16
char* local_buffer_global_end = &local_buffer[16];

// CHECK-F: .memtag global_buffer
// CHECK-F: .globl global_buffer
// CHECK-F: .p2align 4, 0x0
// CHECK-F: .size global_buffer, 16
char global_buffer[16];
// CHECK-G: .memtag _ZL23global_buffer_local_end
// CHECK-G: .p2align 4, 0x0
// CHECK-G: _ZL23global_buffer_local_end:
// CHECK-G: .xword global_buffer+16
// CHECK-G: .size _ZL23global_buffer_local_end, 16
static char* global_buffer_local_end = &global_buffer[16];
// CHECK-H: .memtag global_buffer_global_end
// CHECK-H: .p2align 4, 0x0
// CHECK-H: global_buffer_global_end:
// CHECK-H: .xword global_buffer+16
// CHECK-H: .size global_buffer_global_end, 16
char* global_buffer_global_end = &global_buffer[16];

class MyClass {
 public:
  virtual ~MyClass() {}
  static int my_class_int;
  static const int my_class_const_int;
  virtual int virtual_func() { return 1; }
};
// CHECK-I: .memtag _ZN7MyClass12my_class_intE
// CHECK-I: .globl _ZN7MyClass12my_class_intE
// CHECK-I: .p2align 4, 0x0
// CHECK-I: .size _ZN7MyClass12my_class_intE, 16
int MyClass::my_class_int;
// CHECK-NOT: .memtag _ZN7MyClass18my_class_const_intE
const int MyClass::my_class_const_int = 1;

// CHECK-J: .memtag global_my_class
// CHECK-J: .globl global_my_class
// CHECK-J: .p2align 4, 0x0
// CHECK-J: .size global_my_class, 16
MyClass global_my_class;
// CHECK-K: .memtag _ZL14local_my_class
// CHECK-K: .p2align 4, 0x0
// CHECK-K: .size _ZL14local_my_class, 16
static MyClass local_my_class;

// CHECK-NOT: .memtag _ZL18local_const_string
static const char local_const_string[] = "this is a local string";
// CHECK-L: .memtag _ZL12local_string
// CHECK-L: .p2align 4, 0x0
// CHECK-L: .size _ZL12local_string, 32
static char local_string[] = "this is a local string";

// CHECK-M: .memtag global_atomic_int
// CHECK-M: .globl global_atomic_int
// CHECK-M: .p2align 4, 0x0
// CHECK-M: .size global_atomic_int, 16
_Atomic(int) global_atomic_int;
// CHECK-N: .memtag _ZL16local_atomic_int
// CHECK-N: .local _ZL16local_atomic_int
// CHECK-N: .comm _ZL16local_atomic_int,16,16
static _Atomic(int) local_atomic_int;

union MyUnion {
  int i;
  char c;
};

// CHECK-O: .memtag global_union
// CHECK-O: .globl global_union
// CHECK-O: .p2align 4, 0x0
// CHECK-O: .size global_union, 16
MyUnion global_union;
// CHECK-P: .memtag _ZL11local_union
// CHECK-P: .local _ZL11local_union
// CHECK-P: .comm _ZL11local_union,16,16
static MyUnion local_union;

// CHECK-NOT: .memtag {{.*}}global_tls
thread_local int global_tls;
// CHECK-NOT: .memtag {{.*}}local_tls
static thread_local int local_tls;

/// Prevent the compiler from realising that non-const local variables are not
/// modified, and constant inlining into f().
const void* export_pointers(int c) {
  switch (c) {
    case 0:  return &local_int;
    case 1:  return &local_buffer;
    case 2:  return &local_buffer_local_end;
    case 3:  return &global_buffer_local_end;
    case 4:  return &MyClass::my_class_int;
    case 6:  return &local_my_class;
    case 8:  return &local_string;
    case 9:  return &local_atomic_int;
    case 10: return &local_union;
    case 11: return &local_tls;
  }
  return nullptr;
}

/// Ensure that all tagged globals are loaded/referenced via. the GOT.
// CHECK-NOT:      .memtag _Z1fi
// CHECK-Q:        _Z1fi:
int f(int x) {
  // CHECK-R: .memtag _ZZ1fiE12function_int
  // CHECK-R: .local _ZZ1fiE12function_int
  // CHECK-R: .comm _ZZ1fiE12function_int,16,16
  static int function_int = 0;
  /// Prevent non-const `f` from being promoted to a constant and inlined.
  function_int += x;

  return
  // CHECK-Q-DAG: adrp [[REG_A:x[0-9]+]], :got:global_int
  // CHECK-Q-DAG: ldr  [[REG_A2:x[0-9]+]], [[[REG_A]], :got_lo12:global_int]
  // CHECK-Q-DAG: ldr  {{.*}}, [[[REG_A2]]]
      global_int +
  // CHECK-Q-DAG: adrp [[REG_B:x[0-9]+]], :got:_ZL9local_int
  // CHECK-Q-DAG: ldr  [[REG_B2:x[0-9]+]], [[[REG_B]], :got_lo12:_ZL9local_int]
  // CHECK-Q-DAG: ldr  {{.*}}, [[[REG_B2]]]
      local_int +
  // CHECK-Q-DAG: adrp  [[REG_C:x[0-9]+]], :got:_ZL12local_buffer
  // CHECK-Q-DAG: ldr   [[REG_C2:x[0-9]+]], [[[REG_C]], :got_lo12:_ZL12local_buffer]
  // CHECK-Q-DAG: ldrsb {{.*}}, [[[REG_C2]]]
      local_buffer[0] +
  // CHECK-Q-DAG: adrp   [[REG_D:x[0-9]+]], :got:_ZL22local_buffer_local_end
  // CHECK-Q-DAG: ldr    [[REG_D2:x[0-9]+]], [[[REG_D]], :got_lo12:_ZL22local_buffer_local_end]
  // CHECK-Q-DAG: ldr    [[REG_D3:x[0-9]+]], [[[REG_D2]]]
  // CHECK-Q-DAG: ldursb {{.*}}, [[[REG_D3]], #-16]
      local_buffer_local_end[-16] +
  // CHECK-Q-DAG: adrp   [[REG_E:x[0-9]+]], :got:local_buffer_global_end
  // CHECK-Q-DAG: ldr    [[REG_E2:x[0-9]+]], [[[REG_E]], :got_lo12:local_buffer_global_end]
  // CHECK-Q-DAG: ldr    [[REG_E3:x[0-9]+]], [[[REG_E2]]]
  // CHECK-Q-DAG: ldursb {{.*}}, [[[REG_E3]], #-16]
      local_buffer_global_end[-16] +
  // CHECK-Q-DAG: adrp  [[REG_F:x[0-9]+]], :got:global_buffer{{$}}
  // CHECK-Q-DAG: ldr   [[REG_F2:x[0-9]+]], [[[REG_F]], :got_lo12:global_buffer]
  // CHECK-Q-DAG: ldrsb {{.*}}, [[[REG_F2]]]
      global_buffer[0] +
  // CHECK-Q-DAG: adrp   [[REG_G:x[0-9]+]], :got:_ZL23global_buffer_local_end
  // CHECK-Q-DAG: ldr    [[REG_G2:x[0-9]+]], [[[REG_G]], :got_lo12:_ZL23global_buffer_local_end]
  // CHECK-Q-DAG: ldr    [[REG_G3:x[0-9]+]], [[[REG_G2]]]
  // CHECK-Q-DAG: ldursb {{.*}}, [[[REG_G3]], #-16]
      global_buffer_local_end[-16] +
  // CHECK-Q-DAG: adrp   [[REG_H:x[0-9]+]], :got:global_buffer_global_end
  // CHECK-Q-DAG: ldr    [[REG_H2:x[0-9]+]], [[[REG_H]], :got_lo12:global_buffer_global_end]
  // CHECK-Q-DAG: ldr    [[REG_H3:x[0-9]+]], [[[REG_H2]]]
  // CHECK-Q-DAG: ldursb {{.*}}, [[[REG_H3]], #-16]
      global_buffer_global_end[-16] +
  // CHECK-Q-DAG: adrp [[REG_I:x[0-9]+]], :got:_ZN7MyClass12my_class_intE
  // CHECK-Q-DAG: ldr  [[REG_I2:x[0-9]+]], [[[REG_I]], :got_lo12:_ZN7MyClass12my_class_intE]
  // CHECK-Q-DAG: ldr  {{.*}}, [[[REG_I2]]]
      MyClass::my_class_int +
  /// Constant values - ignore.
      MyClass::my_class_const_int +
      global_my_class.virtual_func() +
      local_my_class.virtual_func() +
      local_const_string[0] +
  // CHECK-Q-DAG: adrp  [[REG_J:x[0-9]+]], :got:_ZL12local_string
  // CHECK-Q-DAG: ldr   [[REG_J2:x[0-9]+]], [[[REG_J]], :got_lo12:_ZL12local_string]
  // CHECK-Q-DAG: ldrsb {{.*}}, [[[REG_J2]]]
      local_string[0] +
  // CHECK-Q-DAG: adrp  [[REG_K:x[0-9]+]], :got:_ZL16local_atomic_int
  // CHECK-Q-DAG: ldr   [[REG_K2:x[0-9]+]], [[[REG_K]], :got_lo12:_ZL16local_atomic_int]
  // CHECK-Q-DAG: ldar {{.*}}, [[[REG_K2]]]
      local_atomic_int +
  // CHECK-Q-DAG: adrp [[REG_L:x[0-9]+]], :got:global_atomic_int
  // CHECK-Q-DAG: ldr  [[REG_L2:x[0-9]+]], [[[REG_L]], :got_lo12:global_atomic_int]
  // CHECK-Q-DAG: ldar {{.*}}, [[[REG_L2]]]
      global_atomic_int +
  // CHECK-Q-DAG: adrp [[REG_M:x[0-9]+]], :got:global_union
  // CHECK-Q-DAG: ldr  [[REG_M2:x[0-9]+]], [[[REG_M]], :got_lo12:global_union]
  // CHECK-Q-DAG: ldr  {{.*}}, [[[REG_M2]]]
      global_union.i +
  // CHECK-Q-DAG: adrp  [[REG_N:x[0-9]+]], :got:_ZL11local_union
  // CHECK-Q-DAG: ldr   [[REG_N2:x[0-9]+]], [[[REG_N]], :got_lo12:_ZL11local_union]
  // CHECK-Q-DAG: ldrsb {{.*}}, [[[REG_N2]]]
      local_union.c +
  /// Global variables - ignore.
      global_tls +
      local_tls +
  // CHECK-Q-DAG: adrp  [[REG_O:x[0-9]+]], :got:_ZZ1fiE12function_int
  // CHECK-Q-DAG: ldr   [[REG_O2:x[0-9]+]], [[[REG_O]], :got_lo12:_ZZ1fiE12function_int]
  // CHECK-Q-DAG: ldr   {{.*}}, [[[REG_O2]]]
      function_int;
}

typedef void (*func_t)(void);
#define CONSTRUCTOR(section_name) \
  __attribute__((used)) __attribute__((section(section_name)))

__attribute__((constructor(0))) void func_constructor() {}
CONSTRUCTOR(".init") func_t func_init = func_constructor;
CONSTRUCTOR(".fini") func_t func_fini = func_constructor;
CONSTRUCTOR(".ctors") func_t func_ctors = func_constructor;
CONSTRUCTOR(".dtors") func_t func_dtors = func_constructor;
CONSTRUCTOR(".init_array") func_t func_init_array = func_constructor;
CONSTRUCTOR(".fini_array") func_t func_fini_array = func_constructor;
CONSTRUCTOR(".preinit_array") func_t preinit_array = func_constructor;
CONSTRUCTOR("array_of_globals") int global1;
CONSTRUCTOR("array_of_globals") int global2;
CONSTRUCTOR("array_of_globals") int global_string;

// CHECK-NOT: .memtag func_constructor
// CHECK-NOT: .memtag func_init
// CHECK-NOT: .memtag func_fini
// CHECK-NOT: .memtag func_ctors
// CHECK-NOT: .memtag func_dtors
// CHECK-NOT: .memtag func_init_array
// CHECK-NOT: .memtag func_fini_array
// CHECK-NOT: .memtag preinit_array
// CHECK-NOT: .memtag global1
// CHECK-NOT: .memtag global2
// CHECK-NOT: .memtag global_string
