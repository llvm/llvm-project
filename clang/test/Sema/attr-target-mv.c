// RUN: %clang_cc1 -triple x86_64-linux-gnu -Wno-strict-prototypes -fsyntax-only -verify %s

void __attribute__((target("sse4.2"))) no_default(void);
void __attribute__((target("arch=sandybridge")))  no_default(void);

void use1(void){
  // expected-error@+1 {{no matching function for call to 'no_default'}}
  no_default();
}

void __attribute__((target("sse4.2"))) has_def(void);
void __attribute__((target("default")))  has_def(void);

void use2(void){
  // expected-error@+2 {{reference to overloaded function could not be resolved; did you mean to call it?}}
  // expected-note@-4 {{possible target for call}}
  +has_def;
}

int __attribute__((target("sse4.2"))) no_proto();
// expected-error@-1 {{multiversioned function must have a prototype}}
// expected-note@+1 {{function multiversioning caused by this declaration}}
int __attribute__((target("arch=sandybridge"))) no_proto();

// The following should all be legal, since they are just redeclarations.
int __attribute__((target("sse4.2"))) redecl1(void);
int __attribute__((target("sse4.2"))) redecl1(void) { return 1; }
int __attribute__((target("arch=sandybridge")))  redecl1(void) { return 2; }

int __attribute__((target("sse4.2"))) redecl2(void) { return 1; }
int __attribute__((target("sse4.2"))) redecl2(void);
int __attribute__((target("arch=sandybridge")))  redecl2(void) { return 2; }

int __attribute__((target("sse4.2"))) redecl3(void) { return 0; }
int __attribute__((target("arch=ivybridge"))) redecl3(void) { return 1; }
int __attribute__((target("arch=sandybridge")))  redecl3(void);
int __attribute__((target("arch=sandybridge")))  redecl3(void) { return 2; }

int __attribute__((target("sse4.2"))) redecl4(void) { return 1; }
int __attribute__((target("arch=sandybridge")))  redecl4(void) { return 2; }
int __attribute__((target("arch=sandybridge")))  redecl4(void);

int __attribute__((target("sse4.2"))) redef(void) { return 1; }
int __attribute__((target("arch=ivybridge"))) redef(void) { return 1; }
int __attribute__((target("arch=sandybridge")))  redef(void) { return 2; }
// expected-error@+2 {{redefinition of 'redef'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((target("arch=sandybridge")))  redef(void) { return 2; }

int __attribute__((target("default"))) redef2(void) { return 1;}
// expected-error@+2 {{redefinition of 'redef2'}}
// expected-note@-2 {{previous definition is here}}
int __attribute__((target("default"))) redef2(void) { return 1;}

int __attribute__((target("sse4.2"))) mv_after_use(void) { return 1; }
int use3(void) {
  return mv_after_use();
}

// expected-error@+1 {{function declaration cannot become a multiversioned function after first usage}}
int __attribute__((target("arch=sandybridge")))  mv_after_use(void) { return 2; }

int __attribute__((target("sse4.2,arch=sandybridge"))) mangle(void) { return 1; }
//expected-error@+2 {{multiversioned function redeclarations require identical target attributes}}
//expected-note@-2 {{previous declaration is here}}
int __attribute__((target("arch=sandybridge,sse4.2")))  mangle(void) { return 2; }

// allow this, since we want to treat the 1st one as fwd-decl of the sandybridge version.
int prev_no_target(void);
int __attribute__((target("arch=sandybridge")))  prev_no_target(void) { return 2; }
int __attribute__((target("arch=ivybridge")))  prev_no_target(void) { return 2; }

int __attribute__((target("arch=sandybridge")))  prev_no_target2(void);
int prev_no_target2(void);
// expected-error@-1 {{function declaration is missing 'target' attribute in a multiversioned function}}
// expected-note@+1 {{function multiversioning caused by this declaration}}
int __attribute__((target("arch=ivybridge")))  prev_no_target2(void);

void __attribute__((target("sse4.2"))) addtl_attrs(void);
//expected-error@+2 {{attribute 'target' multiversioning cannot be combined with attribute 'no_caller_saved_registers'}}
void __attribute__((no_caller_saved_registers,target("arch=sandybridge")))
addtl_attrs(void);

//expected-error@+1 {{attribute 'target' multiversioning cannot be combined with attribute 'no_caller_saved_registers'}}
void __attribute__((target("default"), no_caller_saved_registers)) addtl_attrs2(void);

//expected-error@+2 {{attribute 'target' multiversioning cannot be combined with attribute 'no_caller_saved_registers'}}
//expected-note@+2 {{function multiversioning caused by this declaration}}
void __attribute__((no_caller_saved_registers,target("sse4.2"))) addtl_attrs3(void);
void __attribute__((target("arch=sandybridge")))  addtl_attrs3(void);

void __attribute__((target("sse4.2"))) addtl_attrs4(void);
void __attribute__((target("arch=sandybridge")))  addtl_attrs4(void);
//expected-error@+1 {{attribute 'target' multiversioning cannot be combined}}
void __attribute__((no_caller_saved_registers,target("arch=ivybridge")))  addtl_attrs4(void);

int __attribute__((target("sse4.2"))) diff_cc(void);
// expected-error@+1 {{multiversioned function declaration has a different calling convention}}
__vectorcall int __attribute__((target("arch=sandybridge")))  diff_cc(void);

int __attribute__((target("sse4.2"))) diff_ret(void);
// expected-error@+1 {{multiversioned function declaration has a different return type}}
short __attribute__((target("arch=sandybridge")))  diff_ret(void);

void __attribute__((target("sse4.2"), nothrow, used, nonnull(1))) addtl_attrs5(int*);
void __attribute__((target("arch=sandybridge"))) addtl_attrs5(int*);

void __attribute__((target("sse4.2"))) addtl_attrs6(int*);
void __attribute__((target("arch=sandybridge"), nothrow, used, nonnull)) addtl_attrs6(int*);

int __attribute__((target("sse4.2"))) bad_overload1(void);
int __attribute__((target("arch=sandybridge"))) bad_overload1(void);
// expected-error@+1 {{function declaration is missing 'target' attribute in a multiversioned function}}
int bad_overload1(int);

int bad_overload2(int);
// expected-error@+2 {{conflicting types for 'bad_overload2'}}
// expected-note@-2 {{previous declaration is here}}
int __attribute__((target("sse4.2"))) bad_overload2(void);
// expected-error@+2 {{conflicting types for 'bad_overload2'}}
// expected-note@-5 {{previous declaration is here}}
int __attribute__((target("arch=sandybridge"))) bad_overload2(void);

// expected-error@+2 {{attribute 'target' multiversioning cannot be combined with attribute 'overloadable'}}
// expected-note@+2 {{function multiversioning caused by this declaration}}
int __attribute__((__overloadable__)) __attribute__((target("sse4.2"))) bad_overload3(void);
int __attribute__((target("arch=sandybridge"))) bad_overload3(void);

int __attribute__((target("sse4.2"))) bad_overload4(void);
// expected-error@+1 {{attribute 'target' multiversioning cannot be combined with attribute 'overloadable'}}
int __attribute__((__overloadable__)) __attribute__((target("arch=sandybridge"))) bad_overload4(void);

int __attribute__((target("sse4.2"))) bad_overload5(void);
int __attribute__((target("arch=sandybridge"))) bad_overload5(int);

int __attribute__((target("sse4.2"))) good_overload1(void);
int __attribute__((target("arch=sandybridge"))) good_overload1(void);
int __attribute__((__overloadable__)) good_overload1(int);

int __attribute__((__overloadable__)) good_overload2(int);
int __attribute__((target("sse4.2"))) good_overload2(void);
int __attribute__((target("arch=sandybridge"))) good_overload2(void);

// expected-error@+2 {{attribute 'target' multiversioning cannot be combined with attribute 'overloadable'}}
// expected-note@+2 {{function multiversioning caused by this declaration}}
int __attribute__((__overloadable__)) __attribute__((target("sse4.2"))) good_overload3(void);
int __attribute__((__overloadable__)) __attribute__((target("arch=sandybridge"))) good_overload3(void);
int good_overload3(int);

int good_overload4(int);
// expected-error@+2 {{attribute 'target' multiversioning cannot be combined with attribute 'overloadable'}}
// expected-note@+2 {{function multiversioning caused by this declaration}}
int __attribute__((__overloadable__)) __attribute__((target("sse4.2"))) good_overload4(void);
int __attribute__((__overloadable__)) __attribute__((target("arch=sandybridge"))) good_overload4(void);

int __attribute__((__overloadable__)) __attribute__((target("sse4.2"))) good_overload5(void);
int __attribute__((__overloadable__)) __attribute__((target("arch=sandybridge"))) good_overload5(int);

int __attribute__((target("sse4.2"))) good_overload6(void);
int __attribute__((__overloadable__)) __attribute__((target("arch=sandybridge"))) good_overload6(int);

int __attribute__((__overloadable__)) __attribute__((target("sse4.2"))) good_overload7(void);
int __attribute__((target("arch=sandybridge"))) good_overload7(int);
