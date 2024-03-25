// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wno-strict-prototypes -fsyntax-only -verify %s -Wnonnull

void __attribute__((cpu_specific(ivybridge))) no_default(void);
void __attribute__((cpu_specific(sandybridge)))  no_default(void);

void use1(void){
  // Should be OK, default not a problem.
  no_default();
}

int __attribute__((cpu_specific(atom))) addr_of(void);
int __attribute__((cpu_specific(ivybridge)))  addr_of(void);
int __attribute__((cpu_specific(ivybridge)))  addr_of2(void);

void use2(void){
  addr_of();
  addr_of2();
  // expected-error@+1{{reference to multiversioned function could not be resolved; did you mean to call it with no arguments?}}
  (void)+addr_of;
  // expected-error@+1{{reference to multiversioned function could not be resolved; did you mean to call it with no arguments?}}
  (void)+addr_of2;
  // expected-error@+1{{reference to multiversioned function could not be resolved; did you mean to call it with no arguments?}}
  (void)&addr_of;
  // expected-error@+1{{reference to multiversioned function could not be resolved; did you mean to call it with no arguments?}}
  (void)&addr_of2;
}

// expected-error@+1 {{multiversioned function must have a prototype}}
int __attribute__((cpu_specific(atom))) no_proto();

int __attribute__((cpu_specific(atom))) redecl1(void);
int __attribute__((cpu_specific(atom))) redecl1(void) { return 1; }

int __attribute__((cpu_dispatch(atom))) redecl2(void);
int __attribute__((cpu_dispatch(atom))) redecl2(void) { }
// expected-error@+2 {{redefinition of 'redecl2'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((cpu_dispatch(atom))) redecl2(void) { }

int allow_fwd_decl(void);
int __attribute__((cpu_dispatch(atom))) allow_fwd_decl(void) {}

int allow_fwd_decl2(void);
void use_fwd_decl(void) {
  allow_fwd_decl2();
}
// expected-error@+1 {{function declaration cannot become a multiversioned function after first usage}}
int __attribute__((cpu_dispatch(atom))) allow_fwd_decl2(void) {}


int __attribute__((cpu_specific(atom))) redecl4(void);
// expected-error@+1 {{function declaration is missing 'cpu_specific' or 'cpu_dispatch' attribute in a multiversioned function}}
int redecl4(void);

// expected-warning@+1 {{CPU list contains duplicate entries; attribute ignored}}
int __attribute__((cpu_specific(atom, atom))) dup_procs(void);

int __attribute__((cpu_specific(ivybridge, atom))) dup_procs2(void);
// expected-error@+2 {{multiple 'cpu_specific' functions cannot specify the same CPU: 'atom'}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((cpu_specific(atom))) dup_procs2(void);

int __attribute__((cpu_specific(ivybridge, atom))) dup_procs3(void);
// expected-error@+2 {{multiple 'cpu_specific' functions cannot specify the same CPU: 'ivybridge'}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((cpu_specific(atom, ivybridge))) dup_procs3(void);

int __attribute__((cpu_specific(atom))) redef(void) { return 1; }
// expected-error@+2 {{redefinition of 'redef'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((cpu_specific(atom))) redef(void) { return 2; }

int __attribute((cpu_dispatch(atom))) mult_dispatch(void) {}
// expected-error@+2 {{'cpu_dispatch' function redeclared with different CPUs}}
// expected-note@-2 {{previous declaration is here}}
int __attribute((cpu_dispatch(ivybridge))) mult_dispatch(void) {}

// expected-error@+1 {{'cpu_dispatch' attribute takes at least 1 argument}}
int __attribute((cpu_dispatch())) no_dispatch(void) {}
// expected-error@+1 {{'cpu_specific' attribute takes at least 1 argument}}
int __attribute((cpu_specific())) no_specific(void) {}

//expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'used'}}
void __attribute__((used,cpu_specific(sandybridge)))  addtl_attrs(void);

void __attribute__((target("default"))) addtl_attrs2(void);
// expected-error@+2 {{multiversioning attributes cannot be combined}}
// expected-note@-2 {{previous declaration is here}}
void __attribute__((cpu_specific(sandybridge))) addtl_attrs2(void);

// expected-error@+2 {{'cpu_dispatch' and 'cpu_specific' attributes are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void __attribute((cpu_specific(sandybridge), cpu_dispatch(atom, sandybridge)))
combine_attrs(void);

int __attribute__((cpu_dispatch(ivybridge))) diff_cc(void){}
// expected-error@+1 {{multiversioned function declaration has a different calling convention}}
__vectorcall int __attribute__((cpu_specific(sandybridge))) diff_cc(void);

// expected-warning@+2 {{body of cpu_dispatch function will be ignored}}
int __attribute__((cpu_dispatch(atom))) disp_with_body(void) {
  return 5;
}

// expected-error@+1 {{invalid option 'INVALID'}}
int __attribute__((cpu_specific(INVALID))) called_invalid_value(void){ return 1;}
// expected-warning@+3 {{attribute declaration must precede definition}}
// expected-note@-2 2 {{previous definition is here}}
// expected-error@+1 {{redefinition of}}
int __attribute__((cpu_specific(pentium_iii))) called_invalid_value(void){ return 2;}
int __attribute__((cpu_specific(pentium_4))) called_invalid_value(void){ return 3;}

int use3(void) {
  return called_invalid_value();
}

// expected-warning@+1 {{CPU list contains duplicate entries; attribute ignored}}
int __attribute__((cpu_dispatch(pentium_iii, pentium_iii_no_xmm_regs))) dupe_p3(void);

void __attribute__((cpu_specific(atom), nothrow, nonnull(1))) addtl_attrs(int*);

int __attribute__((cpu_specific(atom))) bad_overload1(void);
int __attribute__((cpu_specific(ivybridge))) bad_overload1(void);
// expected-error@+1 {{function declaration is missing 'cpu_specific' or 'cpu_dispatch' attribute in a multiversioned function}}
int bad_overload1(int);

int bad_overload2(int);
// expected-error@+2 {{conflicting types for 'bad_overload2'}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((cpu_specific(atom))) bad_overload2(void);
int __attribute__((cpu_specific(ivybridge))) bad_overload2(void);

int __attribute__((cpu_dispatch(generic))) bad_overload3(void);
int __attribute__((cpu_specific(ivybridge))) bad_overload3(void);
// expected-error@+1 {{function declaration is missing 'cpu_specific' or 'cpu_dispatch' attribute in a multiversioned function}}
int bad_overload3(int);

int bad_overload4(int);
// expected-error@+2 {{conflicting types for 'bad_overload4'}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((cpu_dispatch(generic))) bad_overload4(void);
int __attribute__((cpu_specific(ivybridge))) bad_overload4(void);

// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(generic))) bad_overload5(void);
int __attribute__((cpu_specific(ivybridge))) bad_overload5(void);

int __attribute__((cpu_specific(generic))) bad_overload6(void);
// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(ivybridge))) bad_overload6(void);

int __attribute__((cpu_specific(atom))) good_overload1(void);
int __attribute__((cpu_specific(ivybridge))) good_overload1(void);
int __attribute__((__overloadable__)) good_overload1(int);

int __attribute__((__overloadable__)) good_overload2(int);
int __attribute__((cpu_specific(atom))) good_overload2(void);
int __attribute__((cpu_specific(ivybridge))) good_overload2(void);

// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(atom))) good_overload3(void);
// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(ivybridge))) good_overload3(void);
int good_overload3(int);

int good_overload4(int);
// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(atom))) good_overload4(void);
// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(ivybridge))) good_overload4(void);

// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(atom))) good_overload5(void);
// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(ivybridge))) good_overload5(int);

int __attribute__((cpu_specific(atom))) good_overload6(void);
// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(ivybridge))) good_overload6(int);

// expected-error@+1 {{attribute 'cpu_specific' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((cpu_specific(atom))) good_overload7(void);
int __attribute__((cpu_specific(ivybridge))) good_overload7(int);

// expected-error@+1 {{invalid option 'x86_64'}}
int __attribute__((cpu_specific(x86_64))) baseline(void);
// expected-error@+1 {{invalid option 'x86_64_v2'}}
int __attribute__((cpu_specific(x86_64_v2))) baseline(void);
