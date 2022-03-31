// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c -std=c99 -fms-extensions -Wno-pragma-pack %s


#pragma omp begin // expected-error {{expected an OpenMP directive}}
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}
#pragma omp begin declare // expected-error {{expected an OpenMP directive}}
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}
#pragma omp begin variant // expected-error {{expected an OpenMP directive}}
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}
#pragma omp variant begin // expected-error {{expected an OpenMP directive}}
#pragma omp declare variant end // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp begin declare variant // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp end declare variant
// TODO: Issue an error message
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}
#pragma omp end declare variant // expected-error {{'#pragma omp end declare variant' with no matching '#pragma omp begin declare variant'}}

int foo(void);
const int var;

#pragma omp begin declare variant // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp end declare variant
#pragma omp begin declare variant xxx // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp end declare variant
#pragma omp begin declare variant match // expected-error {{expected '(' after 'match'}}
#pragma omp end declare variant
#pragma omp begin declare variant match( // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match() // expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx=) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx=yyy) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx=yyy}) // expected-error {{expected ')'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-warning {{extra tokens at the end of '#pragma omp begin declare variant' are ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx={) // expected-error {{expected ')'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx={}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx={vvv, vvv}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx={vvv} xxx) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(xxx={vvv}) xxx // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-warning {{extra tokens at the end of '#pragma omp begin declare variant' are ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor}) // expected-warning {{the context selector 'vendor' in context set 'implementation' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'nec' 'nvidia' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'nec' 'nvidia' 'pgi' 'ti' 'unknown'}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score( ibm)}) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'nec' 'nvidia' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(2 ibm)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'nec' 'nvidia' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(foo()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(5): ibm), vendor(llvm)}) // expected-warning {{the context selector 'vendor' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'vendor' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation={vendor(score(5): ibm), kind(cpu)}) // expected-warning {{the context selector 'kind' is not valid for the context set 'implementation'; selector ignored}} expected-note {{the context selector 'kind' can be nested in the context set 'device'; try 'match(device={kind(property)})'}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'device'; selector ignored}} expected-note {{context selector options are: 'kind' 'arch' 'isa'}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind}) // expected-warning {{the context selector 'kind' in context set 'device' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(score cpu)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<invalid>'); score ignored}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(ibm) }) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<recovery-expr>()'); score ignored}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(score(2 gpu)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('2'); score ignored}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(score(foo()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo()'); score ignored}} expected-warning {{'ibm' is not a valid context property for the context selector 'kind' and the context set 'device'; property ignored}} expected-note {{try 'match(implementation={vendor(ibm)})'}} expected-note {{the ignored property spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(score(5): host), kind(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('5'); score ignored}} expected-warning {{the context selector 'kind' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'kind' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(score(5): nohost), vendor(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('5'); score ignored}} expected-warning {{the context selector 'vendor' is not valid for the context set 'device'; selector ignored}} expected-note {{the context selector 'vendor' can be nested in the context set 'implementation'; try 'match(implementation={vendor(property)})'}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(foo()): cpu}) // expected-error {{expected ')'}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo()'); score ignored}} expected-note {{to match this '('}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(foo()): cpu)) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo()'); score ignored}} expected-warning {{expected '}' after the context selectors for the context set "device"; '}' assumed}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(foo()): cpu)} // expected-error {{expected ')'}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo()'); score ignored}} expected-note {{to match this '('}}
#pragma omp end declare variant

#pragma omp begin declare variant match(implementation = {vendor(score(foo) :llvm)})
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation = {vendor(score(foo()) :llvm)})
#pragma omp end declare variant
#pragma omp begin declare variant match(implementation = {vendor(score(<expr>) :llvm)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}}
#pragma omp end declare variant
#pragma omp begin declare variant match(user = {condition(foo)})
#pragma omp end declare variant
#pragma omp begin declare variant match(user = {condition(foo())})
#pragma omp end declare variant
#pragma omp begin declare variant match(user = {condition(<expr>)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
#pragma omp end declare variant

#pragma omp begin declare variant match(device = {kind(score(&var): cpu)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('&var'); score ignored}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(var): cpu)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('var'); score ignored}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(foo): cpu)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo'); score ignored}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(foo()): cpu)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foo()'); score ignored}}
#pragma omp end declare variant
#pragma omp begin declare variant match(device = {kind(score(<expr>): cpu)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<invalid>'); score ignored}}
#pragma omp end declare variant

#pragma omp begin declare variant match(device={kind(cpu)})
static int defined_twice_a(void) { // expected-note {{previous definition is here}}
  return 0;
}
int defined_twice_b(void) { // expected-note {{previous definition is here}}
  return 0;
}
inline int defined_twice_c(void) { // expected-note {{previous definition is here}}
  return 0;
}
#pragma omp end declare variant
#pragma omp begin declare variant match(device={kind(cpu)})
static int defined_twice_a(void) { // expected-error {{redefinition of 'defined_twice_a[device={kind(cpu)}]'}}
  return 1;
}
int defined_twice_b(void) { // expected-error {{redefinition of 'defined_twice_b[device={kind(cpu)}]'}}
  return 1;
}
inline int defined_twice_c(void) { // expected-error {{redefinition of 'defined_twice_c[device={kind(cpu)}]'}}
  return 1;
}
#pragma omp end declare variant


// TODO: Issue an error message
#pragma omp begin declare variant match(device={kind(cpu)})
// The matching end is missing. Since the device clause is matching we will
// emit and error.
int also_before(void) {
  return 0;
}


#pragma omp begin declare variant match(device={kind(gpu)}) // expected-note {{to match this '#pragma omp begin declare variant'}}
// The matching end is missing. Since the device clause is not matching we will
// cause us to elide the rest of the file and emit and error.
int also_after(void) {
  return 2;
}
int also_before(void) {
  return 2;
}

#pragma omp begin declare variant match(device={kind(fpga)})

This text is never parsed!

#pragma omp end declare variant

This text is also not parsed! // expected-error {{expected '#pragma omp end declare variant'}}
