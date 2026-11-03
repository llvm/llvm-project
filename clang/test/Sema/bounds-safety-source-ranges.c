// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info -fdiagnostics-parseable-fixits %s 2>&1 | \
// RUN:   FileCheck %s --implicit-check-not error --implicit-check-not note --implicit-check-not fix --implicit-check-not warning

#define __counted_by(f)  __attribute__((counted_by(f)))

struct IncompleteTy;
typedef struct IncompleteTy Incomplete_t; 

struct Container__cb {
  int count;
  struct IncompleteTy* buf __counted_by(count);
  Incomplete_t* buf_typedef __counted_by(count);
};

void consume_struct_IncompleteTy(struct IncompleteTy* buf);
int idx(void);

void test_Container__cb(struct Container__cb* ptr) {
  struct Container__cb uninit;
  uninit.count = 0;

  uninit.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:14:{[[@LINE-1]]:16-[[@LINE-1]]:19}: error: cannot assign to 'Container__cb::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"

  struct IncompleteTy* addr_elt_zero_ptr = &ptr->buf[0];
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:45:{[[@LINE-1]]:45-[[@LINE-1]]:53}: error: cannot use 'ptr->buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"

  struct IncompleteTy* addr_elt_idx_ptr = &ptr->buf[idx()];
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:44:{[[@LINE-1]]:44-[[@LINE-1]]:52}: error: cannot use 'ptr->buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"
  
  consume_struct_IncompleteTy(uninit.buf);
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:31:{[[@LINE-1]]:31-[[@LINE-1]]:41}: error: cannot use 'uninit.buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"
  
  consume_struct_IncompleteTy(ptr->buf);
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:31:{[[@LINE-1]]:31-[[@LINE-1]]:39}: error: cannot use 'ptr->buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"

  uninit.buf[0] = uninit.buf[1];
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:3:{[[@LINE-1]]:3-[[@LINE-1]]:13}: error: cannot use 'uninit.buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"
  
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-6]]:19:{[[@LINE-6]]:19-[[@LINE-6]]:29}: error: cannot use 'uninit.buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"
  
  ptr->buf[0] = ptr->buf[1];
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:3:{[[@LINE-1]]:3-[[@LINE-1]]:11}: error: cannot use 'ptr->buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"
  
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-6]]:17:{[[@LINE-6]]:17-[[@LINE-6]]:25}: error: cannot use 'ptr->buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:11:28:{11:28-11:40}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{11:28-11:40}:"__sized_by"
}


struct Container__cb___nomacro {
  int count;
  struct IncompleteTy* buf __attribute__((__counted_by__(count)));
};

void test_Container__cb___nomacro(struct Container__cb___nomacro* ptr) {
  struct Container__cb___nomacro local;
  local.count = 0;
  local.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:13:{[[@LINE-1]]:15-[[@LINE-1]]:18}: error: cannot assign to 'Container__cb___nomacro::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-9]]:43:{[[@LINE-9]]:43-[[@LINE-9]]:57}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{[[@LINE-10]]:43-[[@LINE-10]]:57}:"__sized_by__"
}


struct Containercb_nomacro {
  int count;
  struct IncompleteTy* buf __attribute__((counted_by(count)));
};

void test_Containercb_nomacro(struct Containercb_nomacro* ptr) {
  struct Containercb_nomacro local;
  local.count = 0;
  local.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:13:{[[@LINE-1]]:15-[[@LINE-1]]:18}: error: cannot assign to 'Containercb_nomacro::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-9]]:43:{[[@LINE-9]]:43-[[@LINE-9]]:53}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{[[@LINE-10]]:43-[[@LINE-10]]:53}:"sized_by"
}


#define aaaaaaaa(x) __attribute__((counted_by(x)))
struct Container_aaaaaaaa {
  int count;
  struct IncompleteTy* buf aaaaaaaa(count);
};

void test_Container_aaaaaaaa(struct Container_aaaaaaaa* ptr) {
  struct Container_aaaaaaaa local;
  local.count = 0;
  local.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:13:{[[@LINE-1]]:15-[[@LINE-1]]:18}: error: cannot assign to 'Container_aaaaaaaa::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-9]]:28:{[[@LINE-9]]:28-[[@LINE-9]]:36}: note: consider using '__sized_by' instead of '__counted_by'
  // no fix-it
}


struct Container_macro1 {
  int count;
  struct IncompleteTy* buf __attribute__((counted_by(count)));
};

#define A() \
void test_Contain_macro1(struct Container_macro1* ptr) { \
  struct Container_macro1 local; \
  local.count = 0; \
  local.buf = 0x0; \
}

A()
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:1:{[[@LINE-1]]:1-[[@LINE-1]]:4}: error: cannot assign to 'Container_macro1::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-5]]:13:{[[@LINE-5]]:15-[[@LINE-5]]:18}: note: expanded from macro 'A'
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-14]]:43:{[[@LINE-14]]:43-[[@LINE-14]]:53}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{[[@LINE-15]]:43-[[@LINE-15]]:53}:"sized_by"


#define B() \
struct Container_macro2 { \
  int count; \
  struct IncompleteTy* buf __attribute__((counted_by(count))); \
};
B()

void test_Contain_macro2(struct Container_macro2* ptr) {
  struct Container_macro2 local;
  local.count = 0;
  local.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:13:{[[@LINE-1]]:15-[[@LINE-1]]:18}: error: cannot assign to 'Container_macro2::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-8]]:1:{[[@LINE-8]]:1-[[@LINE-8]]:2}: note: consider using '__sized_by' instead of '__counted_by'
  // no fix-it
}


#define counted_by(x) __attribute__((counted_by(x)))

struct Containercb {
  int count;
  struct IncompleteTy* buf counted_by(count);
};

void test_Containercb(struct Containercb* ptr) {
  struct Containercb local;
  local.count = 0;
  local.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:13:{[[@LINE-1]]:15-[[@LINE-1]]:18}: error: cannot assign to 'Containercb::buf' with '__counted_by' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-9]]:28:{[[@LINE-9]]:28-[[@LINE-9]]:38}: note: consider using '__sized_by' instead of '__counted_by'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{[[@LINE-10]]:28-[[@LINE-10]]:38}:"sized_by"
}


struct Containercbon {
  int count;
  struct IncompleteTy* buf __attribute__((counted_by_or_null(count)));
};

void test_Containercbon(struct Containercbon* ptr) {
  struct Containercbon local;
  local.count = 0;
  local.buf = 0x0;
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-1]]:13:{[[@LINE-1]]:15-[[@LINE-1]]:18}: error: cannot assign to 'Containercbon::buf' with '__counted_by_or_null' {{.*}} because the pointee type 'struct IncompleteTy' is incomplete
  // CHECK: bounds-safety-source-ranges.c:6:1: note: consider providing a complete definition for 'struct IncompleteTy'
  // CHECK: bounds-safety-source-ranges.c:[[@LINE-9]]:43:{[[@LINE-9]]:43-[[@LINE-9]]:61}: note: consider using '__sized_by_or_null' instead of '__counted_by_or_null'
  // CHECK: fix-it:"{{.*}}bounds-safety-source-ranges.c":{[[@LINE-10]]:43-[[@LINE-10]]:61}:"sized_by_or_null"
}

// prevent 'error' from being unmatched
// CHECK: errors generated
