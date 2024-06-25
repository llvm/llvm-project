// RUN: %clang_cc1 -triple aarch64-linux-gnu  -fsyntax-only -verify %s

void __attribute__((target_clones("fp16+sve2-aes", "sb+sve2-sha3+rcpc3+mops", "rdma"))) no_def(void);

// expected-warning@+1 {{unsupported 'default' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void __attribute__((target_clones("default+sha3"))) warn1(void);
// expected-warning@+1 {{version list contains entries that don't impact code generation}}
void __attribute__((target_clones("ssbs+ls64"))) warn2(void);

// expected-error@+2 {{'target_clones' and 'target_version' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target_version("sve-bf16"), target_clones("sme+memtag"))) not_compat(void);

int redecl(void);
int __attribute__((target_clones("frintts", "simd+fp", "default"))) redecl(void) { return 1; }

int __attribute__((target_clones("jscvt+fcma", "rcpc", "default"))) redecl2(void);
int __attribute__((target_clones("jscvt+fcma", "rcpc"))) redecl2(void) { return 1; }

int __attribute__((target_clones("sve+dotprod"))) redecl3(void);
int redecl3(void);

int __attribute__((target_clones("rng", "fp16fml+fp", "default"))) redecl4(void);
// expected-error@+3 {{'target_clones' attribute does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
// expected-warning@+1 {{version list contains entries that don't impact code generation}}
int __attribute__((target_clones("dgh+memtag+rpres+ls64_v", "ebf16+dpb+sha1", "default"))) redecl4(void) { return 1; }

int __attribute__((target_version("flagm2"))) redef2(void) { return 1; }
// expected-error@+2 {{multiversioned function redeclarations require identical target attributes}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("flagm2", "default"))) redef2(void) { return 1; }

int __attribute__((target_clones("f32mm", "f64mm", "sha1+fp"))) redef3(void) { return 1; }
// expected-error@+2 {{'target_clones' attribute does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("f32mm", "sha1+fp", "f64mm"))) redef3(void) { return 1; }

int __attribute__((target_clones("rdm+lse+rdm", "lse+rdm"))) dup1(void) { return 1; }
// expected-warning@+1 {{version list contains duplicate entries}}
int __attribute__((target_clones("rdm+lse+rdm", "rdm+lse+rdm"))) dup2(void) { return 2; }
// expected-warning@+1 {{version list contains duplicate entries}}
int __attribute__((target_clones("rcpc2+sve2-pmull128", "rcpc2+sve2-pmull128"))) dup3(void) { return 3; }
// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("sha3", "default", "default"))) dup4(void);
// expected-warning@+2 {{version list contains duplicate entries}}
// expected-warning@+1 {{version list contains duplicate entries}}
int __attribute__((target_clones("fp", "fp", "crc+dotprod", "dotprod+crc"))) dup5(void) { return 5; }

// expected-warning@+1 {{version list contains duplicate entries}}
int __attribute__((target_clones("fp16+memtag", "memtag+fp16"))) dup6(void) { return 6; }
int __attribute__((target_clones("simd+ssbs2", "simd+dpb2"))) dup7(void) { return 7; }

// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones(""))) empty_target_1(void);
// expected-warning@+3 {{unsupported 'default' in the 'target_clones' attribute string;}}
// expected-warning@+2 {{unsupported 'default' in the 'target_clones' attribute string;}}
// expected-warning@+1 {{version list contains entries that don't impact code generation}}
void __attribute__((target_clones("default+default"))) empty_target_2(void);
// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("+sve2")))
empty_target_3(void);
// expected-warning@+1 {{unsupported 'bs' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("sb+bs")))
empty_target_4(void);

// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("default", "")))
empty_target_5(void);

// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("sve2-bitperm", "sve2-bitperm")))
dupe_normal(void);

void __attribute__((target_clones("default"), target_clones("memtag3+bti"))) dupe_normal2(void);

int mv_after_use(void);
int useage(void) {
  return mv_after_use();
}
// expected-error@+1 {{function declaration cannot become a multiversioned function after first usage}}
int __attribute__((target_clones("sve2-sha3+ssbs2", "sm4"))) mv_after_use(void) { return 1; }
// expected-error@+1 {{'main' cannot be a multiversioned function}}
int __attribute__((target_clones("sve-i8mm"))) main() { return 1; }
