// RUN: %clang_analyze_cc1 -analyzer-checker=core,fuchsia.HandleChecker -analyzer-output=text \
// RUN:     -verify %s

typedef __typeof__(sizeof(int)) size_t;
typedef int zx_status_t;
typedef __typeof__(sizeof(int)) zx_handle_t;
typedef unsigned int uint32_t;
#define NULL ((void *)0)
#define ZX_HANDLE_INVALID 0

#if defined(__clang__)
#define ZX_HANDLE_ACQUIRE __attribute__((acquire_handle("Fuchsia")))
#define ZX_HANDLE_RELEASE __attribute__((release_handle("Fuchsia")))
#define ZX_HANDLE_USE __attribute__((use_handle("Fuchsia")))
#define ZX_HANDLE_ACQUIRE_UNOWNED __attribute__((acquire_handle("FuchsiaUnowned")))
#else
#define ZX_HANDLE_ACQUIRE
#define ZX_HANDLE_RELEASE
#define ZX_HANDLE_USE
#define ZX_HANDLE_ACQUIRE_UNOWNED
#endif

zx_status_t zx_channel_create(
    uint32_t options,
    zx_handle_t *out0 ZX_HANDLE_ACQUIRE,
    zx_handle_t *out1 ZX_HANDLE_ACQUIRE);

zx_status_t zx_handle_close(
    zx_handle_t handle ZX_HANDLE_RELEASE);

ZX_HANDLE_ACQUIRE_UNOWNED
zx_handle_t zx_process_self();

void zx_process_self_param(zx_handle_t *out ZX_HANDLE_ACQUIRE_UNOWNED);

ZX_HANDLE_ACQUIRE
zx_handle_t return_handle();

void escape1(zx_handle_t *in);
void escape2(zx_handle_t in);
void (*escape3)(zx_handle_t) = escape2;

void use1(const zx_handle_t *in ZX_HANDLE_USE);
void use2(zx_handle_t in ZX_HANDLE_USE);

void moreArgs(zx_handle_t, int, ...);
void lessArgs(zx_handle_t, int a = 5);

// To test if argument indexes are OK for operator calls.
struct MyType {
  ZX_HANDLE_ACQUIRE
  zx_handle_t operator+(zx_handle_t ZX_HANDLE_RELEASE replace);
};

void checkUnownedHandle01() {
  zx_handle_t h0;
  h0 = zx_process_self(); // expected-note {{Function 'zx_process_self' returns an unowned handle}}
  zx_handle_close(h0);    // expected-warning {{Releasing an unowned handle}}
                          // expected-note@-1 {{Releasing an unowned handle}}
}

void checkUnownedHandle02() {
  zx_handle_t h0;
  zx_process_self_param(&h0); // expected-note {{Unowned handle allocated through 1st parameter}}
  zx_handle_close(h0);        // expected-warning {{Releasing an unowned handle}}
                              // expected-note@-1 {{Releasing an unowned handle}}
}

void checkInvalidHandle01() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  if (sa == ZX_HANDLE_INVALID)
    ;
  // Will we ever see a warning like below?
  // We eagerly replace the symbol with a constant and lose info...
  use2(sa); // TODOexpected-warning {{Use of an invalid handle}}
  zx_handle_close(sb);
  zx_handle_close(sa);
}

void checkInvalidHandle2() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  if (sb != ZX_HANDLE_INVALID)
    zx_handle_close(sb);
  if (sa != ZX_HANDLE_INVALID)
    zx_handle_close(sa);
}

void handleDieBeforeErrorSymbol01() {
  zx_handle_t sa, sb;
  zx_status_t status = zx_channel_create(0, &sa, &sb);
  if (status < 0)
    return;
  __builtin_trap();
}

void handleDieBeforeErrorSymbol02() {
  zx_handle_t sa, sb;
  zx_status_t status = zx_channel_create(0, &sa, &sb);
  // FIXME: There appears to be non-determinism in choosing
  // which handle to report.
  // expected-note-re@-3 {{Handle allocated through {{(2nd|3rd)}} parameter}}
  if (status == 0) { // expected-note {{Assuming 'status' is equal to 0}}
                     // expected-note@-1 {{Taking true branch}}
    return;          // expected-warning {{Potential leak of handle}}
                     // expected-note@-1 {{Potential leak of handle}}
  }
  __builtin_trap();
}

void checkNoCrash01() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  moreArgs(sa, 1, 2, 3, 4, 5);
  lessArgs(sa);
  zx_handle_close(sa);
  zx_handle_close(sb);
}

void checkNoLeak01() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  zx_handle_close(sa);
  zx_handle_close(sb);
}

void checkNoLeak02() {
  zx_handle_t ay[2];
  zx_channel_create(0, &ay[0], &ay[1]);
  zx_handle_close(ay[0]);
  zx_handle_close(ay[1]);
}

void checkNoLeak03() {
  zx_handle_t ay[2];
  zx_channel_create(0, &ay[0], &ay[1]);
  for (int i = 0; i < 2; i++)
    zx_handle_close(ay[i]);
}

zx_handle_t checkNoLeak04() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  zx_handle_close(sa);
  return sb; // no warning
}

zx_handle_t checkNoLeak05(zx_handle_t *out1) {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  *out1 = sa;
  return sb; // no warning
}

void checkNoLeak06() {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb))
    return;
  zx_handle_close(sa);
  zx_handle_close(sb);
}

void checkLeak01(int tag) {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb)) // expected-note    {{Handle allocated through 2nd parameter}}
    return;                           // expected-note@-1 {{Assuming the condition is false}}
                                      // expected-note@-2 {{Taking false branch}}
  use1(&sa);
  if (tag) // expected-note {{Assuming 'tag' is 0}}
    zx_handle_close(sa);
  // expected-note@-2 {{Taking false branch}}
  use2(sb); // expected-warning {{Potential leak of handle}}
  // expected-note@-1 {{Potential leak of handle}}
  zx_handle_close(sb);
}

void checkLeakFromReturn01(int tag) {
  zx_handle_t sa = return_handle(); // expected-note {{Function 'return_handle' returns an open handle}}
  (void)sa;
} // expected-note {{Potential leak of handle}}
// expected-warning@-1 {{Potential leak of handle}}

void checkReportLeakOnOnePath(int tag) {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb)) // expected-note {{Handle allocated through 2nd parameter}}
    return;                           // expected-note@-1 {{Assuming the condition is false}}
                                      // expected-note@-2 {{Taking false branch}}
  zx_handle_close(sb);
  switch (tag) { // expected-note {{Control jumps to the 'default' case at line}}
  case 0:
    use2(sa);
    return;
  case 1:
    use2(sa);
    return;
  case 2:
    use2(sa);
    return;
  case 3:
    use2(sa);
    return;
  case 4:
    use2(sa);
    return;
  default:
    use2(sa);
    return; // expected-warning {{Potential leak of handle}}
            // expected-note@-1 {{Potential leak of handle}}
  }
}

void checkDoubleRelease01(int tag) {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  // expected-note@-1 {{Handle allocated through 2nd parameter}}
  if (tag)               // expected-note {{Assuming 'tag' is not equal to 0}}
    zx_handle_close(sa); // expected-note {{Handle released through 1st parameter}}
  // expected-note@-2 {{Taking true branch}}
  zx_handle_close(sa); // expected-warning {{Releasing a previously released handle}}
  // expected-note@-1 {{Releasing a previously released handle}}
  zx_handle_close(sb);
}

void checkUseAfterFree01(int tag) {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  // expected-note@-1 {{Handle allocated through 2nd parameter}}
  // expected-note@-2 {{Handle allocated through 3rd parameter}}
  // expected-note@+2 {{Taking true branch}}
  // expected-note@+1 {{Taking false branch}}
  if (tag) {
    // expected-note@-1 {{Assuming 'tag' is not equal to 0}}
    zx_handle_close(sa); // expected-note {{Handle released through 1st parameter}}
    use1(&sa);           // expected-warning {{Using a previously released handle}}
    // expected-note@-1 {{Using a previously released handle}}
  }
  // expected-note@-6 {{Assuming 'tag' is 0}}
  zx_handle_close(sb); // expected-note {{Handle released through 1st parameter}}
  use2(sb);            // expected-warning {{Using a previously released handle}}
  // expected-note@-1 {{Using a previously released handle}}
}

void checkMemberOperatorIndices() {
  zx_handle_t sa, sb, sc;
  zx_channel_create(0, &sa, &sb);
  zx_handle_close(sb);
  MyType t;
  sc = t + sa;
  zx_handle_close(sc);
}

struct HandleStruct {
  zx_handle_t h;
};

void close_handle_struct(HandleStruct hs ZX_HANDLE_RELEASE);

void use_handle_struct(HandleStruct hs ZX_HANDLE_USE);

void checkHandleInStructureUseAfterFree() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb); // expected-note {{Handle allocated through 3rd parameter}}
  HandleStruct hs;
  hs.h = sb;
  use_handle_struct(hs);
  close_handle_struct(hs); // expected-note {{Handle released through 1st parameter}}
  zx_handle_close(sa);

  use2(sb); // expected-warning {{Using a previously released handle}}
  // expected-note@-1 {{Using a previously released handle}}
}

void checkHandleInStructureUseAfterFree2() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb); // expected-note {{Handle allocated through 3rd parameter}}
  HandleStruct hs;
  hs.h = sb;
  use_handle_struct(hs);
  zx_handle_close(sb); // expected-note {{Handle released through 1st parameter}}
  zx_handle_close(sa);

  use_handle_struct(hs); // expected-warning {{Using a previously released handle}}
  // expected-note@-1 {{Using a previously released handle}}
}

void checkHandleInStructureLeak() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb); // expected-note {{Handle allocated through 3rd parameter}}
  HandleStruct hs;
  hs.h = sb;
  zx_handle_close(sa); // expected-warning {{Potential leak of handle}}
  // expected-note@-1 {{Potential leak of handle}}
}

struct HandlePtrStruct {
  zx_handle_t *h;
};

void close_handle_struct(HandlePtrStruct hs ZX_HANDLE_RELEASE);

void use_handle_struct(HandlePtrStruct hs ZX_HANDLE_USE);

void checkHandlePtrInStructureUseAfterFree() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  HandlePtrStruct hs;
  hs.h = &sb;
  use_handle_struct(hs);
  close_handle_struct(hs); // expected-note {{Handle released through 1st parameter}}
  zx_handle_close(sa);

  use2(sb); // expected-warning {{Using a previously released handle}}
  // expected-note@-1 {{Using a previously released handle}}
}

void checkHandlePtrInStructureUseAfterFree2() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  HandlePtrStruct hs;
  hs.h = &sb;
  use_handle_struct(hs);
  zx_handle_close(sb); // expected-note {{Handle released through 1st parameter}}
  zx_handle_close(sa);

  use_handle_struct(hs); // expected-warning {{Using a previously released handle}}
  // expected-note@-1 {{Using a previously released handle}}
}

void checkHandlePtrInStructureLeak() {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb); // expected-note {{Handle allocated through 3rd parameter}}
  HandlePtrStruct hs;
  hs.h = &sb;
  zx_handle_close(sa); // expected-warning {{Potential leak of handle}}
  // expected-note@-1 {{Potential leak of handle}}
}

// Assume this function's declaration that has the release annotation is in one
// header file while its implementation is in another file. We have to annotate
// the declaration because it might be used outside the TU.
// We also want to make sure it is okay to call the function within the same TU.
zx_status_t test_release_handle(zx_handle_t handle ZX_HANDLE_RELEASE) {
  return zx_handle_close(handle);
}

void checkReleaseImplementedFunc() {
  zx_handle_t a, b;
  zx_channel_create(0, &a, &b);
  zx_handle_close(a);
  test_release_handle(b);
}

void use_handle(zx_handle_t handle) {
  // Do nothing.
}

void test_call_by_value() {
  zx_handle_t a, b;
  zx_channel_create(0, &a, &b);
  zx_handle_close(a);
  use_handle(b);
  zx_handle_close(b);
}

void test_call_by_value_leak() {
  zx_handle_t a, b;
  zx_channel_create(0, &a, &b); // expected-note {{Handle allocated through 3rd parameter}}
  zx_handle_close(a);
  // Here we are passing handle b as integer value to a function that could be
  // analyzed by the analyzer, thus the handle should not be considered escaped.
  // After the function 'use_handle', handle b is still tracked and should be
  // reported leaked.
  use_handle(b);
} // expected-warning {{Potential leak of handle}}
// expected-note@-1 {{Potential leak of handle}}

// RAII

template <typename T>
struct HandleWrapper {
  ~HandleWrapper() { close(); }
  void close() {
    if (handle != ZX_HANDLE_INVALID)
      zx_handle_close(handle);
  }
  T *get_handle_address() { return &handle; }

private:
  T handle;
};

void doNotWarnOnRAII() {
  HandleWrapper<zx_handle_t> w1;
  zx_handle_t sb;
  if (zx_channel_create(0, w1.get_handle_address(), &sb))
    return;
  zx_handle_close(sb);
}

template <typename T>
struct HandleWrapperUnkonwDtor {
  ~HandleWrapperUnkonwDtor();
  void close() {
    if (handle != ZX_HANDLE_INVALID)
      zx_handle_close(handle);
  }
  T *get_handle_address() { return &handle; }

private:
  T handle;
};

void doNotWarnOnUnknownDtor() {
  HandleWrapperUnkonwDtor<zx_handle_t> w1;
  zx_handle_t sb;
  if (zx_channel_create(0, w1.get_handle_address(), &sb))
    return;
  zx_handle_close(sb);
}

// Various escaping scenarios

zx_handle_t *get_handle_address();

void escape_store_to_escaped_region01() {
  zx_handle_t sb;
  if (zx_channel_create(0, get_handle_address(), &sb))
    return;
  zx_handle_close(sb);
}

struct object {
  zx_handle_t *get_handle_address();
};

void escape_store_to_escaped_region02(object &o) {
  zx_handle_t sb;
  // Same as above.
  if (zx_channel_create(0, o.get_handle_address(), &sb))
    return;
  zx_handle_close(sb);
}

void escape_store_to_escaped_region03(object o) {
  zx_handle_t sb;
  // Should we consider the pointee of get_handle_address escaped?
  // Maybe we only should it consider escaped if o escapes?
  if (zx_channel_create(0, o.get_handle_address(), &sb))
    return;
  zx_handle_close(sb);
}

void escape_through_call(int tag) {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb))
    return;
  escape1(&sa);
  if (tag)
    escape2(sb);
  else
    escape3(sb);
}

struct have_handle {
  zx_handle_t h;
  zx_handle_t *hp;
};

void escape_through_store01(have_handle *handle) {
  zx_handle_t sa;
  if (zx_channel_create(0, &sa, handle->hp))
    return;
  handle->h = sa;
}

have_handle global;
void escape_through_store02() {
  zx_handle_t sa;
  if (zx_channel_create(0, &sa, global.hp))
    return;
  global.h = sa;
}

have_handle escape_through_store03() {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb))
    return {0, nullptr};
  zx_handle_close(sb);
  return {sa, nullptr};
}

void escape_structs(have_handle *);
void escape_transitively01() {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb))
    return;
  have_handle hs[2];
  hs[1] = {sa, &sb};
  escape_structs(hs);
}

void escape_top_level_pointees(zx_handle_t *h) {
  zx_handle_t h2;
  if (zx_channel_create(0, h, &h2))
    return;
  zx_handle_close(h2);
} // *h should be escaped here. Right?
