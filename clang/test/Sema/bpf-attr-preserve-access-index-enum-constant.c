// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=err %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=warn \
// RUN:   -Wno-error=bpf-enum-not-relocated %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=none \
// RUN:   -Wno-bpf-enum-not-relocated %s

// Where a constant is required, an enumerator of an enum with the
// preserve_access_index attribute is not relocated, and its value is used.
// This is diagnosed with -Wbpf-enum-not-relocated, an error by default, which
// -Wno-error=, -Wno- or #pragma clang diagnostic opts out of.

// none-no-diagnostics

enum bpf_map_type { BPF_MAP_TYPE_UNSPEC, BPF_MAP_TYPE_HASH } __attribute__((preserve_access_index));
enum ops_flags { OPS_ENQ_LAST = 2 } __attribute__((preserve_access_index));

#define __uint(name, val) int (*name)[val]

struct {
  __uint(type, BPF_MAP_TYPE_HASH); // err-error {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}} \
                                   // warn-warning {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}}
} m;

struct ops { unsigned long flags; };
struct ops ops = { .flags = OPS_ENQ_LAST }; // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}} \
                                            // warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}}

int f(int x) {
  asm volatile("r0 = %0" :: "i"(OPS_ENQ_LAST) : "r0"); // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}} \
                                                       // warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}}
  // Relocated, so not diagnosed.
  return x == BPF_MAP_TYPE_HASH;
}

// In a function, the initializer of a static local variable and an array
// designator are required to be constants too. A constexpr local variable is
// checked in bpf-preserve-enum-value-parity.c.
int locals(void) {
  static int s = OPS_ENQ_LAST; // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}} \
                               // warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}}
  static struct ops so = { .flags = OPS_ENQ_LAST }; // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}} \
                                                    // warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}}
  int a[4] = {[BPF_MAP_TYPE_HASH] = 1}; // err-error {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}} \
                                        // warn-warning {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}}
  int r[4] = {[BPF_MAP_TYPE_HASH ... OPS_ENQ_LAST] = 1}; // err-error {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}} \
                                                         // warn-warning {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}} \
                                                         // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}} \
                                                         // warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated where a constant is required, although 'enum ops_flags' has the 'preserve_access_index' attribute}}
  static int sa[4] = {[BPF_MAP_TYPE_HASH] = 1}; // err-error {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}} \
                                                // warn-warning {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}}
  // Relocated, so not diagnosed.
  int l = OPS_ENQ_LAST;
  return s + (int)so.flags + a[0] + r[0] + sa[0] + l;
}

int locals_ignored(void) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbpf-enum-not-relocated"
  static int s = OPS_ENQ_LAST;
  int a[4] = {[BPF_MAP_TYPE_HASH] = 1};
#pragma clang diagnostic pop
  return s + a[0];
}

// No diagnostic in a region where the warning is ignored.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbpf-enum-not-relocated"
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
} m2;
#pragma clang diagnostic pop
