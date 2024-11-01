// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.RetainCount,alpha.core -fblocks -analyzer-output=plist -o %t %s
// RUN: %normalize_plist <%t | diff -ub %S/Inputs/expected-plists/plist-output-alternate.m.plist -

void test_null_init(void) {
  int *p = 0;
  *p = 0xDEADBEEF;
}

void test_null_assign(void) {
  int *p;
  p = 0;
  *p = 0xDEADBEEF;
}

void test_null_assign_transitive(void) {
  int *p;
  p = 0;
  int *q = p;
  *q = 0xDEADBEEF;
}

void test_null_cond(int *p) {
  if (!p) {
    *p = 0xDEADBEEF;
  }
}

void test_null_cond_transitive(int *q) {
  if (!q) {
    int *p = q;
    *p = 0xDEADBEEF;
  }
}

void test_null_field(void) {
  struct s { int *p; } x;
  x.p = 0;
  *(x.p) = 0xDEADBEEF;
}

// leak reports should not show paths that end with exit() (but ones that don't end with exit())
void panic(void) __attribute__((noreturn));
enum { kCFNumberSInt8Type = 1,     kCFNumberSInt16Type = 2,     kCFNumberSInt32Type = 3,     kCFNumberSInt64Type = 4,     kCFNumberFloat32Type = 5,     kCFNumberFloat64Type = 6,      kCFNumberCharType = 7,     kCFNumberShortType = 8,     kCFNumberIntType = 9,     kCFNumberLongType = 10,     kCFNumberLongLongType = 11,     kCFNumberFloatType = 12,     kCFNumberDoubleType = 13,      kCFNumberCFIndexType = 14,      kCFNumberNSIntegerType = 15,     kCFNumberCGFloatType = 16,     kCFNumberMaxType = 16    };
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef signed long CFIndex;
typedef CFIndex CFNumberType;
typedef const struct __CFNumber * CFNumberRef;

extern CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);

void rdar8331641(int x) {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z); // expected-warning{{leak}}
  if (x)
    panic();
  (void) value;
}

