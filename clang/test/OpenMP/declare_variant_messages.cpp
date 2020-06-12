// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c++ -std=c++14 -fms-extensions -Wno-pragma-pack -fexceptions -fcxx-exceptions %s

// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c++ -std=c++14 -fms-extensions -Wno-pragma-pack -fexceptions -fcxx-exceptions %s


#pragma omp declare // expected-error {{expected an OpenMP directive}}

int foo();

template <typename T>
T foofoo();

#pragma omp declare variant // expected-error {{expected '(' after 'declare variant'}}
#pragma omp declare variant( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo // expected-error {{expected ')'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}} expected-note {{to match this '('}}
#pragma omp declare variant(x) // expected-error {{use of undeclared identifier 'x'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) xxx // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <int>) match // expected-error {{expected '(' after 'match'}}
#pragma omp declare variant(foofoo <int>) match( // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match() // expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation) // expected-warning {{expected '=' after the context set name "implementation"; '=' assumed}} expected-warning {{expected '{' after the '=' that follows the context set name "implementation"; '{' assumed}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation =) // expected-warning {{expected '{' after the '=' that follows the context set name "implementation"; '{' assumed}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation = yyy) // expected-warning {{expected '{' after the '=' that follows the context set name "implementation"; '{' assumed}} expected-warning {{'yyy' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation = yyy }) // expected-warning {{expected '{' after the '=' that follows the context set name "implementation"; '{' assumed}} expected-warning {{'yyy' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation = {) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation = {vvv, vvv}) // expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation = {vvv} implementation) // expected-error {{expected ')'}} expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation = {vvv}) implementation // expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor}) // expected-warning {{the context selector 'vendor' in context set 'implementation' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score( ibm)}) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(2 ibm)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(foofoo <int>()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{score expressions in the OpenMP context selector need to be constant; foofoo<int>() is not and will be ignored}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm), vendor(llvm)}) // expected-warning {{the context selector 'vendor' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'vendor' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm), kind(cpu)}) // expected-warning {{the context selector 'kind' is not valid for the context set 'implementation'; selector ignored}} expected-note {{the context selector 'kind' can be nested in the context set 'device'; try 'match(device={kind(property)})'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'device'; selector ignored}} expected-note {{context selector options are: 'kind' 'isa' 'arch'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind}) // expected-warning {{the context selector 'kind' in context set 'device' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score cpu)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<invalid>'); score ignored}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score( ibm)}) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(2 gpu)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('2'); score ignored}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(foofoo <int>()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foofoo<int>()'); score ignored}} expected-warning {{'ibm' is not a valid context property for the context selector 'kind' and the context set 'device'; property ignored}} expected-note {{try 'match(implementation={vendor(ibm)})'}} expected-note {{the ignored property spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(5): host), kind(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('5'); score ignored}} expected-warning {{the context selector 'kind' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'kind' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(5): nohost), vendor(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('5'); score ignored}} expected-warning {{the context selector 'vendor' is not valid for the context set 'device'; selector ignored}} expected-note {{the context selector 'vendor' can be nested in the context set 'implementation'; try 'match(implementation={vendor(property)})'}} expected-note {{the ignored selector spans until here}}
int bar();

#pragma omp declare variant // expected-error {{expected '(' after 'declare variant'}}
#pragma omp declare variant( // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <T> // expected-error {{expected ')'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}} expected-note {{to match this '('}}
#pragma omp declare variant(x) // expected-error {{use of undeclared identifier 'x'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) xxx // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foofoo <T>) match // expected-error {{expected '(' after 'match'}}
#pragma omp declare variant(foofoo <T>) match( // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <T>) match() // expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
#pragma omp declare variant(foofoo <T>) match(implementation) // expected-warning {{expected '=' after the context set name "implementation"; '=' assumed}} expected-warning {{expected '{' after the '=' that follows the context set name "implementation"; '{' assumed}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(implementation =) // expected-warning {{expected '{' after the '=' that follows the context set name "implementation"; '{' assumed}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(implementation = {) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{expected '}' after the context selectors for the context set "implementation"; '}' assumed}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(implementation = {vvv, vvv}) // expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(user = {score(<expr>) : condition(<expr>)}) // expected-warning {{'score' is not a valid context selector for the context set 'user'; selector ignored}} expected-note {{context selector options are: 'condition'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(user = {score(<expr>) : condition(<expr>)}) // expected-warning {{'score' is not a valid context selector for the context set 'user'; selector ignored}} expected-note {{context selector options are: 'condition'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(user = {condition(<expr>)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(user = {condition(<expr>)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <T>) match(implementation = {vvv} implementation) // expected-error {{expected ')'}} expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <T>) match(implementation = {vvv}) xxx // expected-warning {{'vvv' is not a valid context selector for the context set 'implementation'; selector ignored}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score( ibm)}) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(C ibm)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'amd' 'arm' 'bsc' 'cray' 'fujitsu' 'gnu' 'ibm' 'intel' 'llvm' 'pgi' 'ti' 'unknown'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(foofoo <int>()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(C+5): ibm), vendor(llvm)}) // expected-warning {{the context selector 'vendor' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'vendor' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm), kind(cpu)}) // expected-warning {{the context selector 'kind' is not valid for the context set 'implementation'; selector ignored}} expected-note {{the context selector 'kind' can be nested in the context set 'device'; try 'match(device={kind(property)})'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={xxx}) // expected-warning {{'xxx' is not a valid context selector for the context set 'device'; selector ignored}} expected-note {{context selector options are: 'kind' 'isa' 'arch'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind}) // expected-warning {{the context selector 'kind' in context set 'device' requires a context property defined in parentheses; selector ignored}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind(}) // expected-error {{expected ')'}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind()}) // expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score cpu)}) // expected-error {{expected '(' after 'score'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('<invalid>'); score ignored}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score( ibm)}) // expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(C gpu)}) // expected-error {{expected ')'}} expected-error {{expected ')'}} expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('C'); score ignored}} expected-warning {{expected identifier or string literal describing a context property; property skipped}} expected-note {{to match this '('}} expected-note {{context property options are: 'host' 'nohost' 'cpu' 'gpu' 'fpga' 'any'}} expected-note {{to match this '('}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(foofoo <int>()) ibm)}) // expected-warning {{expected '':'' after the score expression; '':'' assumed}} expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('foofoo<int>()'); score ignored}} expected-warning {{'ibm' is not a valid context property for the context selector 'kind' and the context set 'device'; property ignored}} expected-note {{try 'match(implementation={vendor(ibm)})'}} expected-note {{the ignored property spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(C+5): host), kind(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('C + 5'); score ignored}} expected-warning {{the context selector 'kind' was used already in the same 'omp declare variant' directive; selector ignored}} expected-note {{the previous context selector 'kind' used here}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(foofoo <int>) match(device={kind(score(C+5): nohost), vendor(llvm)}) // expected-warning {{the context selector 'kind' in the context set 'device' cannot have a score ('C + 5'); score ignored}} expected-warning {{the context selector 'vendor' is not valid for the context set 'device'; selector ignored}} expected-note {{the context selector 'vendor' can be nested in the context set 'implementation'; try 'match(implementation={vendor(property)})'}} expected-note {{the ignored selector spans until here}}
template <typename T, int C>
T barbar();

#pragma omp declare variant(foo) match(implementation = {vendor(score(foo) :llvm)}) // expected-warning {{score expressions in the OpenMP context selector need to be constant; foo is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation = {vendor(score(foo()) :llvm)}) // expected-warning {{score expressions in the OpenMP context selector need to be constant; foo() is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation = {vendor(score(<expr>) :llvm)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}}
#pragma omp declare variant(foo) match(user = {condition(foo)}) // expected-error {{the user condition in the OpenMP context selector needs to be constant; foo is not}}
#pragma omp declare variant(foo) match(user = {condition(foo())}) // expected-error {{the user condition in the OpenMP context selector needs to be constant; foo() is not}}
#pragma omp declare variant(foo) match(user = {condition(<expr>)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
int score_and_cond_non_const();

#pragma omp declare variant(foo) match(implementation = {vendor(score(foo) :llvm)})
#pragma omp declare variant(foo) match(implementation = {vendor(score(foo()) :llvm)})
#pragma omp declare variant(foo) match(implementation = {vendor(score(<expr>) :llvm)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}}
#pragma omp declare variant(foo) match(user = {condition(foo)})
#pragma omp declare variant(foo) match(user = {condition(foo())})
#pragma omp declare variant(foo) match(user = {condition(<expr>)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
template<int C>
int score_and_cond_non_const_no_inst();

#pragma omp declare variant(foo) match(implementation = {vendor(score(foo) :llvm)}) // expected-warning {{score expressions in the OpenMP context selector need to be constant; foo is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation = {vendor(score(foo()) :llvm)}) // expected-warning {{score expressions in the OpenMP context selector need to be constant; foo() is not and will be ignored}}
#pragma omp declare variant(foo) match(implementation = {vendor(score(<expr>) :llvm)}) // expected-error {{expected expression}} expected-error {{use of undeclared identifier 'expr'}} expected-error {{expected expression}}
#pragma omp declare variant(foo) match(user = {condition(foo)}) // expected-error {{the user condition in the OpenMP context selector needs to be constant; foo is not}}
#pragma omp declare variant(foo) match(user = {condition(foo())}) // expected-error {{the user condition in the OpenMP context selector needs to be constant; foo() is not}}
#pragma omp declare variant(foo) match(user = {condition(<C>)}) // expected-error {{expected expression}} expected-error {{expected expression}} expected-note {{the ignored selector spans until here}}
template<int C>
int score_and_cond_non_const_inst();

constexpr int constexpr_fn(int i) { return 7 * i; }
#pragma omp declare variant(foo) match(implementation = {vendor(score(constexpr_fn(3)) : llvm)})
#pragma omp declare variant(foo) match(user = {condition(constexpr_fn(1))})
int score_and_cond_const();

#pragma omp declare variant(foo) match(implementation = {vendor(score(constexpr_fn(3)) : llvm)})
#pragma omp declare variant(foo) match(implementation = {vendor(score(constexpr_fn(C)) : llvm)})
#pragma omp declare variant(foo) match(user = {condition(constexpr_fn(1))})
#pragma omp declare variant(foo) match(user = {condition(constexpr_fn(C))})
template <int C>
int score_and_cond_const_inst();

__attribute__((pure)) int pure() { return 0; }

#pragma omp declare variant(pure) match(user = {condition(1)}) // expected-warning {{ignoring return value of function declared with pure attribute}}
int unused_warning_after_specialization() { return foo(); }

void score_and_cond_inst() {
  score_and_cond_non_const();
  score_and_cond_non_const_inst<8>(); // expected-note {{in instantiation of function template specialization 'score_and_cond_non_const_inst<8>' requested here}}
  score_and_cond_const_inst<9>();
  unused_warning_after_specialization();
}

#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int a; // expected-error {{'#pragma omp declare variant' can only be applied to functions}}

#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp threadprivate(a) // expected-error {{'#pragma omp declare variant' can only be applied to functions}}
int var;
#pragma omp threadprivate(var)


#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare // expected-error {{expected an OpenMP directive}}



#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(barbar <int>) match(xxx = {}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma options align = packed
int main();



#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(barbar <int>) match(xxx = {}) // expected-error {{function declaration is expected after 'declare variant' directive}}
#pragma init_seg(compiler)
int main();


#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-error {{single declaration is expected after 'declare variant' directive}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int b, c;


#pragma omp declare variant(C) match(implementation = {}) // expected-error {{'C' does not refer to a value}}

template <class C> // expected-note {{declared here}}
void h(C *hp, C *hp2, C *hq, C *lin) {
  b = 0;
}


#pragma omp declare variant(barbar <int>) match(implementation = {}) // expected-error {{variant in '#pragma omp declare variant' with type '<overloaded function type>' is incompatible with type 'void (int *, int *, int *, int *)'}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
template <>
void h(int *hp, int *hp2, int *hq, int *lin);

int after_use_variant(void);
int after_use();
int bar() {
  return after_use();
}


#pragma omp declare variant(after_use_variant) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{'#pragma omp declare variant' cannot be applied for function after first usage; the original function might be used}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int after_use(void);

int fn();
int fn(int);
#pragma omp declare variant(fn) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int overload(void);

int fn1();
int fn1(int);

#pragma omp declare variant(fn1) match(implementation = {}) // expected-error {{variant in '#pragma omp declare variant' with type '<overloaded function type>' is incompatible with type 'int (float)'}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int overload1(float);

int fn_constexpr_variant();

#pragma omp declare variant(fn_constexpr_variant) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
constexpr int fn_constexpr(); // expected-error {{'#pragma omp declare variant' does not support constexpr functions}}

constexpr int fn_constexpr_variant1();

#pragma omp declare variant(fn_constexpr_variant1) match(implementation = {}) // expected-error {{'#pragma omp declare variant' does not support constexpr functions}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int fn_constexpr1();

int fn_sc_variant();

#pragma omp declare variant(fn_sc_variant) match(xxx = {}) // expected-error {{function with '#pragma omp declare variant' has a different storage class}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
static int fn_sc();

static int fn_sc_variant1();

#pragma omp declare variant(fn_sc_variant1) match(implementation = {}) // expected-error {{function with '#pragma omp declare variant' has a different storage class}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int fn_sc1();

int fn_inline_variant();

#pragma omp declare variant(fn_inline_variant) match(xxx = {}) // expected-error {{function with '#pragma omp declare variant' has a different inline specification}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
inline int fn_inline();

inline int fn_inline_variant1();

#pragma omp declare variant(fn_inline_variant1) match(implementation = {}) // expected-error {{function with '#pragma omp declare variant' has a different inline specification}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int fn_inline1();

auto fn_deduced_variant() { return 0; }
#pragma omp declare variant(fn_deduced_variant) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int fn_deduced();

int fn_deduced_variant1();
#pragma omp declare variant(fn_deduced_variant1) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
auto fn_deduced1() { return 0; }

auto fn_deduced3() { return 0; }

#pragma omp declare variant(fn_deduced_variant1) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-warning {{'#pragma omp declare variant' cannot be applied to the function that was defined already; the original function might be used}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
auto fn_deduced3();

auto fn_deduced_variant2() { return 0; }

#pragma omp declare variant(fn_deduced_variant2) match(xxx = {}) // expected-error {{variant in '#pragma omp declare variant' with type 'int ()' is incompatible with type 'float ()'}} expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
float fn_deduced2();


int fn_except_variant() noexcept(true); // expected-error {{exception specification in declaration does not match previous declaration}}

#pragma omp declare variant(fn_except_variant) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
int fn_except() noexcept(false); // expected-note {{previous declaration is here}}


int fn_except_variant1() noexcept(false); // expected-error {{exception specification in declaration does not match previous declaration}}

#pragma omp declare variant(fn_except_variant1) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
int fn_except1() noexcept(true); // expected-note {{previous declaration is here}}

struct SpecialFuncs {
  void vd();

#pragma omp declare variant(SpecialFuncs::vd) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
  SpecialFuncs(); // expected-error {{'#pragma omp declare variant' does not support constructors}}

#pragma omp declare variant(SpecialFuncs::vd) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
  ~SpecialFuncs(); // expected-error {{'#pragma omp declare variant' does not support destructors}}

  void baz();
  void bar();
  void bar(int);
#pragma omp declare variant(SpecialFuncs::baz) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
#pragma omp declare variant(SpecialFuncs::bar) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}

#pragma omp declare variant(fn_sc_variant1) match(implementation = {}) // expected-error {{variant in '#pragma omp declare variant' with type 'int (*)()' is incompatible with type 'void (SpecialFuncs::*)()'}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
  void foo1();
  SpecialFuncs& foo(const SpecialFuncs&);
  SpecialFuncs& bar(SpecialFuncs&&);

#pragma omp declare variant(SpecialFuncs::foo) match(xxx = {}) // expected-warning {{'xxx' is not a valid context set in a `declare variant`; set ignored}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
  SpecialFuncs& operator=(const SpecialFuncs&) = default; // expected-error {{'#pragma omp declare variant' does not support defaulted functions}}

#pragma omp declare variant(SpecialFuncs::bar) match(implementation = {}) // expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'vendor' 'extension' 'unified_address' 'unified_shared_memory' 'reverse_offload' 'dynamic_allocators' 'atomic_default_mem_order'}} expected-note {{the ignored selector spans until here}}
  SpecialFuncs& operator=(SpecialFuncs&&) = delete; // expected-error {{'#pragma omp declare variant' does not support deleted functions}}
};

namespace N {

#pragma omp declare variant // expected-error {{function declaration is expected after 'declare variant' directive}}
} // namespace N

#pragma omp declare variant // expected-error {{function declaration is expected after 'declare variant' directive}}

#pragma omp declare variant // expected-error {{function declaration is expected after 'declare variant' directive}}
