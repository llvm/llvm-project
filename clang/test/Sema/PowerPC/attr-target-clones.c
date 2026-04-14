// RUN: %clang_cc1 -triple powerpc-ibm-aix-xcoff  -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff  -fsyntax-only -verify %s

// expected-error@+1 {{'target_clones' multiversioning requires a default target}}
void __attribute__((target_clones("cpu=pwr7")))
no_default(void);

// expected-error@+2 {{'target_clones' and 'target' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target("cpu=pwr7"), target_clones("cpu=pwr8")))
ignored_attr(void);

// expected-error@+2 {{'target' and 'target_clones' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target_clones("default", "cpu=pwr8"), target("cpu=pwr7")))
ignored_attr2(void);

int __attribute__((target_clones("cpu=pwr9", "default"))) redecl4(void);
// expected-error@+3 {{'target_clones' attribute does not match previous declaration}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("cpu=pwr7", "default")))
redecl4(void) { return 1; }

int __attribute__((target_clones("cpu=pwr7", "default"))) redecl7(void);
// expected-error@+2 {{multiversioning attributes cannot be combined}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target("cpu=pwr8"))) redecl7(void) { return 1; }

int __attribute__((target("cpu=pwr9"))) redef2(void) { return 1; }
// expected-error@+2 {{multiversioning attributes cannot be combined}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target_clones("cpu=pwr7", "default"))) redef2(void) { return 1; }

int __attribute__((target_clones("cpu=pwr9,default"))) redef3(void) { return 1; }
// expected-error@+2 {{redefinition of 'redef3'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((target_clones("cpu=pwr9,default"))) redef3(void) { return 1; }

// Duplicates are allowed
// expected-warning@+2 {{mixing 'target_clones' specifier mechanisms is permitted for GCC compatibility}}
// expected-warning@+1 2 {{version list contains duplicate entries}}
int __attribute__((target_clones("cpu=pwr9,cpu=power9", "cpu=power9, default")))
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
void __attribute__((target_clones("default, ,cpu=pwr7")))
empty_target_4(void);

// expected-warning@+1 {{unsupported '' in the 'target_clones' attribute string;}}
void __attribute__((target_clones("default,cpu=pwr7", "")))
empty_target_5(void);

// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("default", "default")))
dupe_default(void);

// expected-warning@+1 {{version list contains duplicate entries}}
void __attribute__((target_clones("cpu=pwr9,cpu=power9,default")))
dupe_normal(void);

// expected-error@+2 {{attribute 'target_clones' cannot appear more than once on a declaration}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute__((target_clones("cpu=pwr7,default"), target_clones("cpu=pwr8,default")))
dupe_normal2(void);

int mv_after_use(void);
int useage(void) {
  return mv_after_use();
}
// expected-error@+1 {{function declaration cannot become a multiversioned function after first usage}}
int __attribute__((target_clones("cpu=pwr9", "default"))) mv_after_use(void) { return 1; }

void bad_overload1(void) __attribute__((target_clones("cpu=pwr8", "default")));
// expected-error@+2 {{conflicting types for 'bad_overload1'}}
// expected-note@-2 {{previous declaration is here}}
void bad_overload1(int p) {}

void bad_overload2(int p) {}
// expected-error@+2 {{conflicting types for 'bad_overload2'}}
// expected-note@-2 {{previous definition is here}}
void bad_overload2(void) __attribute__((target_clones("cpu=pwr8", "default")));

void bad_overload3(void) __attribute__((target_clones("cpu=pwr8", "default")));
// expected-error@+2 {{conflicting types for 'bad_overload3'}}
// expected-note@-2 {{previous declaration is here}}
void bad_overload3(int) __attribute__((target_clones("cpu=pwr8", "default")));


void good_overload1(void) __attribute__((target_clones("cpu=pwr7", "cpu=power10", "default")));
void __attribute__((__overloadable__)) good_overload1(int p) {}

// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload2(void) __attribute__((target_clones("cpu=pwr7", "default")));
void good_overload2(int p) {}
// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload3(void) __attribute__((target_clones("cpu=pwr7", "default")));
// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload3(int) __attribute__((target_clones("cpu=pwr7", "default")));

void good_overload4(void) __attribute__((target_clones("cpu=pwr7", "default")));
// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload4(int) __attribute__((target_clones("cpu=pwr7", "default")));

// expected-error@+1 {{attribute 'target_clones' multiversioning cannot be combined with attribute 'overloadable'}}
void __attribute__((__overloadable__)) good_overload5(void) __attribute__((target_clones("cpu=pwr7", "default")));
void good_overload5(int) __attribute__((target_clones("cpu=pwr7", "default")));


void good_isa_level(int) __attribute__((target_clones("default", "cpu=pwr7", "cpu=pwr8", "cpu=pwr9", "cpu=pwr10")));

// expected-warning@+1 {{unknown CPU 'bad-cpu' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void bad_cpu(int) __attribute__((target_clones("default", "cpu=bad-cpu")));

// expected-warning@+1 {{unsupported CPU 'pwr3' in the 'target_clones' attribute string; 'target_clones' attribute ignored}}
void bad_cpu(int) __attribute__((target_clones("default", "cpu=pwr3")));

// expected-error@+1 {{'target_clones' multiversioning requires a default target}}
void __attribute__((target_clones()))
gh173684_empty_attribute_args(void);

// expected-error@+1 {{'target_clones' multiversioning requires a default target}}
void __attribute__((target_clones))
gh173684_empty_attribute_args_2(void);
