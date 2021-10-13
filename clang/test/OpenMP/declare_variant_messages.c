// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c -std=c99 -fms-extensions -Wno-pragma-pack %s


#pragma omp declare // expected-error {{expected an OpenMP directive}}

int foo(void);

#pragma omp declare variant // expected-error {{expected '(' after 'declare variant'}}
#pragma omp declare variant( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo // expected-error {{expected ')'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}} expected-note {{to match this '('}}
#pragma omp declare variant(x) // expected-error {{use of undeclared identifier 'x'}} expected-error {{expected 'match' clause on}}
#pragma omp declare variant(foo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) xxx // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match // expected-error {{expected '(' after 'match'}}
#pragma omp declare variant(foo) match( // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match() // expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx=) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx=yyy) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx=yyy}) // expected-error {{expected ')'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={) // expected-error {{expected ')'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx={vvv, vvv}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx={vvv} xxx) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foo) match(xxx={vvv}) xxx // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(implementation={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(implementation={vendor}) // expected-warning {{the context selector 'vendor' in context set 'implementation' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(implementation={vendor(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}}
#pragma omp declare variant(foo) match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}}
#pragma omp declare variant(foo) match(implementation={vendor(score( ibm)}) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor(score(2 ibm)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor(score(foo()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{score expressions in the OpenMP context selector need to be constant; foo() is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm), vendor(llvm)}) // expected-warning {{the context selector 'vendor' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'vendor' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm), kind(cpu)}) // expected-warning {{the context selector 'kind' is not valid for the context set 'implementation'; selector ignored}} expected-note {{the context selector 'kind' can be nested in the context set 'device'; try 'match(device={kind(property)})'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(device={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'device'; selector ignored}} expected-note {{context selector options are: 'kind' 'arch' 'isa'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(device={kind}) // expected-warning {{the context selector 'kind' in context set 'device' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(device={kind(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(device={kind()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}}
#pragma omp declare variant(foo) match(device={kind(score cpu)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<invalid>'); score ignored}}
#pragma omp declare variant(foo) match(device = {kind(score(ibm) }) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<recovery-expr>()'); score ignored}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(device={kind(score(2 gpu)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('2'); score ignored}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(device={kind(score(foo()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo()'); score ignored}} expected-warning {{'ibm' is not a valid context property for the context selector 'kind' and the context set 'device'; property ignored}} expected-note {{try 'match(implementation={vendor(ibm)})'}} expected-note {{the ignored property spans until here}}
#pragma omp declare variant(foo) match(device={kind(score(5): host), kind(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('5'); score ignored}} expected-warning {{the context selector 'kind' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'kind' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(device={kind(score(5): nohost), vendor(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('5'); score ignored}} expected-warning {{the context selector 'vendor' is not valid for the context set 'device'; selector ignored}} expected-note {{the context selector 'vendor' can be nested in the context set 'implementation'; try 'match(implementation={vendor(property)})'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foo) match(implementation={extension("aaa")}) // expected-warning {{'aaa' is not a valid context property for the context selector 'extension' and the context set 'implementation'; property ignored}} expected-note {{context property options are: 'match_all' 'match_any' 'match_none'}} expected-note {{the ignored property spans until here}}
int bar(void);

#pragma omp declare variant(foo) match(implementation = {vendor(score(foo) :llvm)}) // expected-warning {{score expressions in the OpenMP context selector need to be constant; foo is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation = {vendor(score(foo()) :llvm)}) // expected-warning {{score expressions in the OpenMP context selector need to be constant; foo() is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation = {vendor(score(<expr>) :llvm)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}}
#pragma omp declare variant(foo) match(user = {condition(foo)}) // expected-error {{the user condition in the OpenMP context selector needs to be constant; foo is not}}
#pragma omp declare variant(foo) match(user = {condition(foo())}) // expected-error {{the user condition in the OpenMP context selector needs to be constant; foo() is not}}
#pragma omp declare variant(foo) match(user = {condition(<expr>)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
int score_and_cond_non_const();

#pragma omp declare variant(foo) match(construct={teams,parallel,for,simd})
#pragma omp declare variant(foo) match(construct={target teams}) // expected-error {{expected ')'}} expected-warning {{expected '}' after the context selectors for the context set "construct"; '}' assumed}} expected-note {{to match this '('}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(construct={parallel for}) // expected-error {{expected ')'}} expected-warning {{expected '}' after the context selectors for the context set "construct"; '}' assumed}} expected-note {{to match this '('}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(construct={for simd}) // expected-error {{expected ')'}} expected-warning {{expected '}' after the context selectors for the context set "construct"; '}' assumed}} expected-note {{to match this '('}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
int construct(void);

#pragma omp declare variant(foo) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int a; // expected-error {{'#pragma omp declare variant' can only be applied to functions}}

#pragma omp declare variant(foo) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp threadprivate(a) // expected-error {{'#pragma omp declare variant' can only be applied to functions}}
int var;
#pragma omp threadprivate(var)


#pragma omp declare variant(foo) match(xxx={}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare // expected-error {{expected an OpenMP directive}}



#pragma omp declare variant(foo) match(xxx={}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma options align=packed
int main();



#pragma omp declare variant(foo) match(implementation={vendor(llvm)}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(foo) match(implementation={vendor(llvm)}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma init_seg(compiler)
int main();


#pragma omp declare variant(foo) match(xxx={}) // expected-error {{single declaration is expected after 'declare variant' directive}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int b, c;

int no_proto();
#pragma omp declare variant(no_proto) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int no_proto_too();

int proto1(int);

#pragma omp declare variant(proto1) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int diff_proto(); // expected-note {{previous declaration is here}}

int diff_proto(double); // expected-error {{conflicting types for 'diff_proto'}}

#pragma omp declare variant(no_proto) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int diff_proto1(double);

int after_use_variant(void);
int after_use();
int bar() {
  return after_use();
}


#pragma omp declare variant(after_use_variant) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-warning {{'#pragma omp declare variant' cannot be applied for function after first usage; the original function might be used}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int after_use(void);
#pragma omp declare variant(after_use_variant) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int defined(void) { return 0; }
int defined1(void) { return 0; }

#pragma omp declare variant(after_use_variant) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-warning {{'#pragma omp declare variant' cannot be applied to the function that was defined already; the original function might be used}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int defined1(void);


int diff_cc_variant(void);

#pragma omp declare variant(diff_cc_variant) match(xxx={}) // expected-error {{variant in '#pragma omp declare variant' with type 'int (void)' is incompatible with type 'int (void) __attribute__((vectorcall))'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
__vectorcall int diff_cc(void);

int diff_ret_variant(void);

#pragma omp declare variant(diff_ret_variant) match(xxx={}) // expected-error {{variant in '#pragma omp declare variant' with type 'int (void)' is incompatible with type 'void (void)'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
void diff_ret(void);

void marked(void);
void not_marked(void);

#pragma omp declare variant(not_marked) match(implementation={vendor(unknown)}, device={kind(cpu)}) // expected-note {{marked as 'declare variant' here}}
void marked_variant(void);

#pragma omp declare variant(marked_variant) match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-warning {{variant function in '#pragma omp declare variant' is itself marked as '#pragma omp declare variant'}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
void marked(void);

#pragma omp declare variant(foo) match(device = {isa("foo")})
int unknown_isa_trait(void);
#pragma omp declare variant(foo) match(device = {isa(foo)})
int unknown_isa_trait2(void);
#pragma omp declare variant(foo) match(device = {kind(fpga), isa(bar)})
int ignored_isa_trait(void);

void caller() {
  unknown_isa_trait();  // expected-warning {{isa trait 'foo' is not known to the current target; verify the spelling or consider restricting the context selector with the 'arch' selector further}}
  unknown_isa_trait2(); // expected-warning {{isa trait 'foo' is not known to the current target; verify the spelling or consider restricting the context selector with the 'arch' selector further}}
  ignored_isa_trait();
}

// Unknown arch
#pragma omp begin declare variant match(device={isa(sse2020)}) // expected-warning {{isa trait 'sse2020' is not known to the current target; verify the spelling or consider restricting the context selector with the 'arch' selector further}}
#pragma omp end declare variant

// Unknown arch guarded by arch.
#pragma omp begin declare variant match(device={isa(sse2020), arch(ppc)})
#pragma omp end declare variant

#pragma omp declare variant // expected-error {{function declaration is expected after 'declare variant' directive}}

#pragma omp declare variant // expected-error {{function declaration is expected after 'declare variant' directive}}

// FIXME: If the scores are equivalent we should detect that and allow it.
#pragma omp begin declare variant match(implementation = {vendor(score(2) \
                                                                 : llvm)})
#pragma omp declare variant(foo) match(implementation = {vendor(score(2) \
                                                                : llvm)}) // expected-error@-1 {{nested OpenMP context selector contains duplicated trait 'llvm' in selector 'vendor' and set 'implementation' with different score}}
int conflicting_nested_score(void);
#pragma omp end declare variant

// FIXME: We should build the conjuction of different conditions, see also the score fixme above.
#pragma omp begin declare variant match(user = {condition(1)})
#pragma omp declare variant(foo) match(user = {condition(1)}) // expected-error {{nested user conditions in OpenMP context selector not supported (yet)}}
int conflicting_nested_condition(void);
#pragma omp end declare variant
