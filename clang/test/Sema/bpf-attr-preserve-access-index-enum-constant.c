// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=err %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=warn \
// RUN:   -Wno-error=bpf-preserve-access-index-enumerator %s
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -fsyntax-only -verify=none \
// RUN:   -Wno-bpf-preserve-access-index-enumerator %s

// Where a constant is required, an enumerator of an enum with the
// preserve_access_index attribute is not relocated, and its value is used.
// This is diagnosed with -Wbpf-preserve-access-index-enumerator, an error by
// default, which -Wno-error=, -Wno- or #pragma clang diagnostic opts out of.

// none-no-diagnostics

enum bpf_map_type { BPF_MAP_TYPE_UNSPEC, BPF_MAP_TYPE_HASH } __attribute__((preserve_access_index));
enum ops_flags { OPS_ENQ_LAST = 2 } __attribute__((preserve_access_index));

#define __uint(name, val) int (*name)[val]

struct {
  __uint(type, BPF_MAP_TYPE_HASH); // err-error {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}} \
                                   // warn-warning {{enumerator 'BPF_MAP_TYPE_HASH' is not relocated where a constant is required, although 'enum bpf_map_type' has the 'preserve_access_index' attribute}}
} m;

struct ops { unsigned long flags; };
struct ops ops = { .flags = OPS_ENQ_LAST }; // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated}} warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated}}

int f(int x) {
  asm volatile("r0 = %0" :: "i"(OPS_ENQ_LAST) : "r0"); // err-error {{enumerator 'OPS_ENQ_LAST' is not relocated}} warn-warning {{enumerator 'OPS_ENQ_LAST' is not relocated}}
  // Relocated, so not diagnosed.
  return x == BPF_MAP_TYPE_HASH;
}

// No diagnostic in a region where the warning is ignored.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbpf-preserve-access-index-enumerator"
struct {
  __uint(type, BPF_MAP_TYPE_HASH);
} m2;
#pragma clang diagnostic pop
