// RUN: %clang_cc1 -triple x86_64-linux-gnu  -fsyntax-only -verify %s

// expected-error@+1 {{'target_clones' multiversioning requires a default target}}
void __attribute__((target_clones("sse4.2", "arch=sandybridge")))
no_default(void);

// expected-error@+2 {{'target_clones' and 'target' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target("sse4.2"), target_clones("arch=sandybridge")))
ignored_attr(void);
// expected-error@+2 {{'target' and 'target_clones' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target_clones("arch=sandybridge,default"), target("sse4.2")))
ignored_attr2(void);

int redecl(void);
int __attribute__((target_clones("sse4.2", "default"))) redecl(void) { return 1; }

int __attribute__((target_clones("sse4.2", "default"))) redecl2(void);
int __attribute__((target_clones("sse4.2", "default"))) redecl2(void) { return 1; }

int __attribute__((target_clones("sse4.2", "default"))) redecl3(void);
int redecl3(void);

int __attribute__((target_clones("sse4.2", "arch=atom", "default"))) redecl4(void);
// expected-error@+3 {{'target_clones' attribute does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("sse4.2", "arch=sandybridge", "default")))
redecl4(void) { return 1; }

int __attribute__((target("sse4.2"))) redef2(void) { return 1; }
// expected-error@+2 {{multiversioning attributes cannot be combined}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("sse4.2", "default"))) redef2(void) { return 1; }

int __attribute__((target_clones("sse4.2,default"))) redef3(void) { return 1; }
// expected-error@+2 {{redefinition of 'redef3'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((target_clones("sse4.2,default"))) redef3(void) { return 1; }

int __attribute__((target_clones("sse4.2,default"))) redef4(void) { return 1; }
// expected-error@+2 {{redefinition of 'redef4'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((target_clones("sse4.2,default"))) redef4(void) { return 1; }

// Duplicates are allowed, however they alter name mangling.
// expected-warning@+2 {{mixing 'target_clones' specifier mechanisms is permitted for GCC compatibility}}
// expected-warning@+1 2 {{version list contains duplicate entries}}
int __attribute__((target_clones("arch=atom,arch=atom", "arch=atom,default")))
dupes(void) { return 1; }

// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("")))
empty_target_1(void);
// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones(",default")))
empty_target_2(void);
// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("default,")))
empty_target_3(void);
// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("default, ,avx2")))
empty_target_4(void);

// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("default,avx2", "")))
empty_target_5(void);

// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("default", "default")))
dupe_default(void);

// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("avx2,avx2,default")))
dupe_normal(void);

// expected-error@+2 {{attribute 'target_clones' cannot appear more than once on a declaration}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target_clones("avx2,default"), target_clones("arch=atom,default")))
dupe_normal2(void);

int mv_after_use(void);
int useage(void) {
  return mv_after_use();
}
// expected-error@+1 {{function declaration cannot become a multiversioned function after first usage}}
int __attribute__((target_clones("sse4.2", "default"))) mv_after_use(void) { return 1; }

void bad_overload1(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
void bad_overload1(int p) {}

void bad_overload2(int p) {}
// expected-error@+2 {{conflicting types for 'bad_overload2'}}
// expected-note@-2 {{previous definition is here}}
void bad_overload2(void) __attribute__((target_clones("mmx", "sse4.2", "default")));

void bad_overload3(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
// expected-error@+2 {{conflicting types for 'bad_overload3'}}
// expected-note@-2 {{previous declaration is here}}
void bad_overload3(int) __attribute__((target_clones("mmx", "sse4.2", "default")));

void good_overload1(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
void __attribute__((__overloadable__)) good_overload1(int p) {}

// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload2(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
void good_overload2(int p) {}

// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload3(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload3(int) __attribute__((target_clones("mmx", "sse4.2", "default")));

void good_overload4(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload4(int) __attribute__((target_clones("mmx", "sse4.2", "default")));

// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload5(void) __attribute__((target_clones("mmx", "sse4.2", "default")));
void good_overload5(int) __attribute__((target_clones("mmx", "sse4.2", "default")));
