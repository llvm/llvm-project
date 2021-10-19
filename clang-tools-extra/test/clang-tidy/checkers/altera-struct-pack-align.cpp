// RUN: %check_clang_tidy %s altera-struct-pack-align %t -- -header-filter=.*

// Struct needs both alignment and packing
struct error {
  char a;
  double b;
  char c;
};
// CHECK-MESSAGES: :[[@LINE-5]]:8: warning: accessing fields in struct 'error' is inefficient due to padding; only needs 10 bytes but is using 24 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-6]]:8: note: use "__attribute__((packed))" to reduce the amount of padding applied to struct 'error'
// CHECK-MESSAGES: :[[@LINE-7]]:8: warning: accessing fields in struct 'error' is inefficient due to poor alignment; currently aligned to 8 bytes, but recommended alignment is 16 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-8]]:8: note: use "__attribute__((aligned(16)))" to align struct 'error' to 16 bytes
// CHECK-FIXES: __attribute__((packed))
// CHECK-FIXES: __attribute__((aligned(16)));

// Struct is explicitly packed, but needs alignment
struct error_packed {
  char a;
  double b;
  char c;
} __attribute__((packed));
// CHECK-MESSAGES: :[[@LINE-5]]:8: warning: accessing fields in struct 'error_packed' is inefficient due to poor alignment; currently aligned to 1 bytes, but recommended alignment is 16 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-6]]:8: note: use "__attribute__((aligned(16)))" to align struct 'error_packed' to 16 bytes
// CHECK-FIXES: __attribute__((aligned(16)))

// Struct is properly packed, but needs alignment
struct align_only {
  char a;
  char b;
  char c;
  char d;
  int e;
  double f;
};
// CHECK-MESSAGES: :[[@LINE-8]]:8: warning: accessing fields in struct 'align_only' is inefficient due to poor alignment; currently aligned to 8 bytes, but recommended alignment is 16 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-9]]:8: note: use "__attribute__((aligned(16)))" to align struct 'align_only' to 16 bytes
// CHECK-FIXES: __attribute__((aligned(16)));

// Struct is perfectly packed but wrongly aligned
struct bad_align {
  char a;
  double b;
  char c;
} __attribute__((packed)) __attribute__((aligned(8)));
// CHECK-MESSAGES: :[[@LINE-5]]:8: warning: accessing fields in struct 'bad_align' is inefficient due to poor alignment; currently aligned to 8 bytes, but recommended alignment is 16 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-6]]:8: note: use "__attribute__((aligned(16)))" to align struct 'bad_align' to 16 bytes
// CHECK-FIXES: __attribute__((aligned(16)));

struct bad_align2 {
  char a;
  double b;
  char c;
} __attribute__((packed)) __attribute__((aligned(32)));
// CHECK-MESSAGES: :[[@LINE-5]]:8: warning: accessing fields in struct 'bad_align2' is inefficient due to poor alignment; currently aligned to 32 bytes, but recommended alignment is 16 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-6]]:8: note: use "__attribute__((aligned(16)))" to align struct 'bad_align2' to 16 bytes
// CHECK-FIXES: __attribute__((aligned(16)));

struct bad_align3 {
  char a;
  double b;
  char c;
} __attribute__((packed)) __attribute__((aligned(4)));
// CHECK-MESSAGES: :[[@LINE-5]]:8: warning: accessing fields in struct 'bad_align3' is inefficient due to poor alignment; currently aligned to 4 bytes, but recommended alignment is 16 bytes [altera-struct-pack-align]
// CHECK-MESSAGES: :[[@LINE-6]]:8: note: use "__attribute__((aligned(16)))" to align struct 'bad_align3' to 16 bytes
// CHECK-FIXES: __attribute__((aligned(16)));

// Struct is both perfectly packed and aligned
struct success {
  char a;
  double b;
  char c;
} __attribute__((packed)) __attribute__((aligned(16)));
//Should take 10 bytes and be aligned to 16 bytes

// Struct is properly packed, and explicitly aligned
struct success2 {
  int a;
  int b;
  int c;
} __attribute__((aligned(16)));

// If struct is properly aligned, packing not needed
struct success3 {
  char a;
  double b;
  char c;
} __attribute__((aligned(16)));

// If struct is templated, warnings should not be triggered
template <typename A, typename B>
struct success4 {
  A a;
  B b;
  int c;
};

// Warnings should not trigger on struct instantiations
void no_trigger_on_instantiation() {
  struct bad_align3 instantiated { 'a', 0.001, 'b' };
}

