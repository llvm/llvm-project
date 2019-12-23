// RUN: %clang_analyze_cc1 -analyzer-checker=core,fuchsia.HandleChecker -analyzer-output=text \
// RUN:     -verify %s

typedef __typeof__(sizeof(int)) size_t;
typedef int zx_status_t;
typedef __typeof__(sizeof(int)) zx_handle_t;
typedef unsigned int uint32_t;
#define NULL ((void *)0)
#define ZX_HANDLE_INVALID 0

#if defined(__clang__)
#define ZX_HANDLE_ACQUIRE  __attribute__((acquire_handle("Fuchsia")))
#define ZX_HANDLE_RELEASE  __attribute__((release_handle("Fuchsia")))
#define ZX_HANDLE_USE  __attribute__((use_handle("Fuchsia")))
#else
#define ZX_HANDLE_ACQUIRE
#define ZX_HANDLE_RELEASE
#define ZX_HANDLE_USE
#endif

zx_status_t zx_channel_create(
    uint32_t options,
    zx_handle_t *out0 ZX_HANDLE_ACQUIRE,
    zx_handle_t *out1 ZX_HANDLE_ACQUIRE);

zx_status_t zx_handle_close(
    zx_handle_t handle ZX_HANDLE_RELEASE);

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
  if (zx_channel_create(0, &sa, &sb)) // expected-note    {{Handle allocated here}}
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

void checkReportLeakOnOnePath(int tag) {
  zx_handle_t sa, sb;
  if (zx_channel_create(0, &sa, &sb)) // expected-note {{Handle allocated here}}
    return;                           // expected-note@-1 {{Assuming the condition is false}}
                                      // expected-note@-2 {{Taking false branch}}
  zx_handle_close(sb);
  switch(tag) { // expected-note {{Control jumps to the 'default' case at line}} 
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
  // expected-note@-1 {{Handle allocated here}}
  if (tag) // expected-note {{Assuming 'tag' is not equal to 0}}
    zx_handle_close(sa); // expected-note {{Handle released here}}
  // expected-note@-2 {{Taking true branch}}
  zx_handle_close(sa); // expected-warning {{Releasing a previously released handle}}
  // expected-note@-1 {{Releasing a previously released handle}}
  zx_handle_close(sb);
}

void checkUseAfterFree01(int tag) {
  zx_handle_t sa, sb;
  zx_channel_create(0, &sa, &sb);
  // expected-note@-1 {{Handle allocated here}}
  // expected-note@-2 {{Handle allocated here}}
  // expected-note@+2 {{Taking true branch}}
  // expected-note@+1 {{Taking false branch}}
  if (tag) {
    // expected-note@-1 {{Assuming 'tag' is not equal to 0}}
    zx_handle_close(sa); // expected-note {{Handle released here}}
    use1(&sa); // expected-warning {{Using a previously released handle}}
    // expected-note@-1 {{Using a previously released handle}}
  }
  // expected-note@-6 {{Assuming 'tag' is 0}}
  zx_handle_close(sb); // expected-note {{Handle released here}}
  use2(sb); // expected-warning {{Using a previously released handle}}
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

void doNotWarnOnUnkownDtor() {
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
