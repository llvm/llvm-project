```{raw} html
<style type="text/css">
  .none { background-color: #FFCCCC }
  .part { background-color: #FFFF99 }
  .good { background-color: #CCFF99 }
</style>
```

```{role} none
```

```{role} part
```

```{role} good
```

```{contents}
:local:
```

# OpenMP Support

Clang fully supports OpenMP 4.5, almost all of 5.0 and most of 5.1/2.
Clang supports offloading to X86_64, AArch64, PPC64[LE], NVIDIA GPUs (all models) and AMD GPUs (all models).

In addition, the LLVM OpenMP runtime `libomp` supports the OpenMP Tools
Interface (OMPT) on x86, x86_64, AArch64, and PPC64 on Linux, Windows, and macOS.
OMPT is also supported for NVIDIA and AMD GPUs.

For the list of supported features from OpenMP 5.0 and 5.1
see {ref}`OpenMP implementation details <openmp-implementation-details>` and
{ref}`OpenMP 51 implementation details <openmp-51-implementation-details>`.

## General improvements

- New collapse clause scheme to avoid expensive remainder operations.
  Compute loop index variables after collapsing a loop nest via the
  collapse clause by replacing the expensive remainder operation with
  multiplications and additions.
- When using the collapse clause on a loop nest the default behavior
  is to automatically extend the representation of the loop counter to
  64 bits for the cases where the sizes of the collapsed loops are not
  known at compile time. To prevent this conservative choice and use
  at most 32 bits, compile your program with the
  `-fopenmp-optimistic-collapse`.

## GPU devices support

### Data-sharing modes

Clang supports two data-sharing models for Cuda devices: `Generic` and `Cuda`
modes. The default mode is `Generic`. `Cuda` mode can give an additional
performance and can be activated using the `-fopenmp-cuda-mode` flag. In
`Generic` mode all local variables that can be shared in the parallel regions
are stored in the global memory. In `Cuda` mode local variables are not shared
between the threads and it is user responsibility to share the required data
between the threads in the parallel regions. Often, the optimizer is able to
reduce the cost of `Generic` mode to the level of `Cuda` mode, but the flag,
as well as other assumption flags, can be used for tuning.

### Features not supported or with limited support for Cuda devices

- Cancellation constructs are not supported.
- Doacross loop nest is not supported.
- User-defined reductions are supported only for trivial types.
- Nested parallelism: inner parallel regions are executed sequentially.
- Debug information for OpenMP target regions is supported, but sometimes it may
  be required to manually specify the address class of the inspected variables.
  In some cases the local variables are actually allocated in the global memory,
  but the debug info may be not aware of it.

(openmp-implementation-details)=

## OpenMP 5.0 Implementation Details

The following table provides a quick overview over various OpenMP 5.0 features
and their implementation status. Please post on the
[Discourse forums (Runtimes - OpenMP category)] for more
information or if you want to help with the
implementation.

| Category          | Feature                                                      | Status                  | Reviews                                                                                               |
| ----------------- | ------------------------------------------------------------ | ----------------------- | ----------------------------------------------------------------------------------------------------- |
| loop              | support != in the canonical loop form                        | {good}`done`            | [D54441][D54441]                                                                                      |
| loop              | #pragma omp loop (directive)                                 | {part}`partial`         | [D145823][D145823] (combined forms)                                                                   |
| loop              | #pragma omp loop bind                                        | {part}`worked on`       | [D144634][D144634] (needs review)                                                                     |
| loop              | collapse imperfectly nested loop                             | {good}`done`            |                                                                                                       |
| loop              | collapse non-rectangular nested loop                         | {good}`done`            |                                                                                                       |
| loop              | C++ range-base for loop                                      | {good}`done`            |                                                                                                       |
| loop              | clause: if for SIMD directives                               | {good}`done`            |                                                                                                       |
| loop              | inclusive scan (matching C++17 PSTL)                         | {good}`done`            |                                                                                                       |
| memory management | memory allocators                                            | {good}`done`            | r341687,r357929                                                                                       |
| memory management | allocate directive and allocate clause                       | {good}`done`            | r355614,r335952                                                                                       |
| OMPD              | OMPD interfaces                                              | {good}`done`            | [D99914][D99914] (Supports only HOST(CPU) and Linux                                                   |
| OMPT              | OMPT interfaces (callback support)                           | {good}`done`            |                                                                                                       |
| thread affinity   | thread affinity                                              | {good}`done`            |                                                                                                       |
| task              | taskloop reduction                                           | {good}`done`            |                                                                                                       |
| task              | task affinity                                                | {part}`not upstream`    | <https://github.com/jklinkenberg/openmp/tree/task-affinity>                                           |
| task              | clause: depend on the taskwait construct                     | {good}`done`            | [D113540][D113540] (regular codegen only)                                                             |
| task              | depend objects and detachable tasks                          | {good}`done`            |                                                                                                       |
| task              | mutexinoutset dependence-type for tasks                      | {good}`done`            | [D53380][D53380],[D57576][D57576]                                                                     |
| task              | combined taskloop constructs                                 | {good}`done`            |                                                                                                       |
| task              | master taskloop                                              | {good}`done`            |                                                                                                       |
| task              | parallel master taskloop                                     | {good}`done`            |                                                                                                       |
| task              | master taskloop simd                                         | {good}`done`            |                                                                                                       |
| task              | parallel master taskloop simd                                | {good}`done`            |                                                                                                       |
| SIMD              | atomic and simd constructs inside SIMD code                  | {good}`done`            |                                                                                                       |
| SIMD              | SIMD nontemporal                                             | {good}`done`            |                                                                                                       |
| device            | infer target functions from initializers                     | {part}`worked on`       |                                                                                                       |
| device            | infer target variables from initializers                     | {good}`done`            | [D146418][D146418]                                                                                    |
| device            | OMP_TARGET_OFFLOAD environment variable                      | {good}`done`            | [D50522][D50522]                                                                                      |
| device            | support full 'defaultmap' functionality                      | {good}`done`            | [D69204][D69204]                                                                                      |
| device            | device specific functions                                    | {good}`done`            |                                                                                                       |
| device            | clause: device_type                                          | {good}`done`            |                                                                                                       |
| device            | clause: extended device                                      | {good}`done`            |                                                                                                       |
| device            | clause: uses_allocators clause                               | {good}`done`            | [PR157025][PR157025]                                                                                  |
| device            | clause: in_reduction                                         | {part}`worked on`       | r308768                                                                                               |
| device            | omp_get_device_num()                                         | {good}`done`            | [D54342][D54342],[D128347][D128347]                                                                   |
| device            | structure mapping of references                              | {none}`unclaimed`       |                                                                                                       |
| device            | nested target declare                                        | {good}`done`            | [D51378][D51378]                                                                                      |
| device            | implicitly map 'this' (this[:1])                             | {good}`done`            | [D55982][D55982]                                                                                      |
| device            | allow access to the reference count (omp_target_is_present)  | {good}`done`            |                                                                                                       |
| device            | requires directive                                           | {good}`done`            |                                                                                                       |
| device            | clause: unified_shared_memory                                | {good}`done`            | [D52625][D52625],[D52359][D52359]                                                                     |
| device            | clause: unified_address                                      | {part}`partial`         |                                                                                                       |
| device            | clause: reverse_offload                                      | {part}`partial`         | [D52780][D52780],[D155003][D155003]                                                                   |
| device            | clause: atomic_default_mem_order                             | {good}`done`            | [D53513][D53513]                                                                                      |
| device            | clause: dynamic_allocators                                   | {part}`unclaimed parts` | [D53079][D53079]                                                                                      |
| device            | user-defined mappers                                         | {good}`done`            | [D56326][D56326],[D58638][D58638],[D58523][D58523],[D58074][D58074],[D60972][D60972],[D59474][D59474] |
| device            | map array-section with implicit mapper                       | {good}`done`            | [PR101101][PR101101]                                                                                  |
| device            | mapping lambda expression                                    | {good}`done`            | [D51107][D51107]                                                                                      |
| device            | clause: use_device_addr for target data                      | {good}`done`            |                                                                                                       |
| device            | support close modifier on map clause                         | {good}`done`            | [D55719][D55719],[D55892][D55892]                                                                     |
| device            | teams construct on the host device                           | {good}`done`            | r371553                                                                                               |
| device            | support non-contiguous array sections for target update      | {good}`done`            | [PR144635][PR144635]                                                                                  |
| device            | pointer attachment                                           | {part}`being repaired`  | @abhinavgaba ([PR153683][PR153683])                                                                   |
| atomic            | hints for the atomic construct                               | {good}`done`            | [D51233][D51233]                                                                                      |
| base language     | C11 support                                                  | {good}`done`            |                                                                                                       |
| base language     | C++11/14/17 support                                          | {good}`done`            |                                                                                                       |
| base language     | lambda support                                               | {good}`done`            |                                                                                                       |
| misc              | array shaping                                                | {good}`done`            | [D74144][D74144]                                                                                      |
| misc              | library shutdown (omp_pause_resource[\_all])                 | {good}`done`            | [D55078][D55078]                                                                                      |
| misc              | metadirectives                                               | {part}`mostly done`     | [D91944][D91944], [PR128640][PR128640]                                                                |
| misc              | conditional modifier for lastprivate clause                  | {good}`done`            |                                                                                                       |
| misc              | iterator and multidependences                                | {good}`done`            |                                                                                                       |
| misc              | depobj directive and depobj dependency kind                  | {good}`done`            |                                                                                                       |
| misc              | user-defined function variants                               | {good}`done`.           | [D67294][D67294], [D64095][D64095], [D71847][D71847], [D71830][D71830], [D109635][D109635]            |
| misc              | pointer/reference to pointer based array reductions          | {good}`done`            |                                                                                                       |
| misc              | prevent new type definitions in clauses                      | {good}`done`            |                                                                                                       |
| memory model      | memory model update (seq_cst, acq_rel, release, acquire,...) | {good}`done`            |                                                                                                       |

(openmp-51-implementation-details)=

## OpenMP 5.1 Implementation Details

The following table provides a quick overview over various OpenMP 5.1 features
and their implementation status.
Please post on the
[Discourse forums (Runtimes - OpenMP category)] for more
information or if you want to help with the
implementation.

| Category          | Feature                                                     | Status              | Reviews                                                                                                                                                                                                                         |
| ----------------- | ----------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| atomic            | 'compare' clause on atomic construct                        | {good}`done`        | [D120290][D120290], [D120007][D120007], [D118632][D118632], [D120200][D120200], [D116261][D116261], [D118547][D118547], [D116637][D116637]                                                                                      |
| atomic            | 'fail' clause on atomic construct                           | {part}`worked on`   | [D123235][D123235] (in progress)                                                                                                                                                                                                |
| base language     | C++ attribute specifier syntax                              | {good}`done`        | [D105648][D105648]                                                                                                                                                                                                              |
| device            | 'present' map type modifier                                 | {good}`done`        | [D83061][D83061], [D83062][D83062], [D84422][D84422]                                                                                                                                                                            |
| device            | 'present' motion modifier                                   | {good}`done`        | [D84711][D84711], [D84712][D84712]                                                                                                                                                                                              |
| device            | 'present' in defaultmap clause                              | {good}`done`        | [D92427][D92427]                                                                                                                                                                                                                |
| device            | map clause reordering based on 'present' modifier           | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| device            | device-specific environment variables                       | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| device            | omp_target_is_accessible routine                            | {good}`done`        | [PR138294][PR138294]                                                                                                                                                                                                            |
| device            | omp_get_mapped_ptr routine                                  | {good}`done`        | [D141545][D141545]                                                                                                                                                                                                              |
| device            | new async target memory copy routines                       | {good}`done`        | [D136103][D136103]                                                                                                                                                                                                              |
| device            | thread_limit clause on target construct                     | {part}`partial`     | [D141540][D141540] (offload), [D152054][D152054] (host, in progress)                                                                                                                                                            |
| device            | has_device_addr clause on target construct                  | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| device            | use_device_ptr/addr preserve host address when lookup fails | {good}`done`        | [PR174659][PR174659]                                                                                                                                                                                                            |
| device            | iterators in map clause or motion clauses                   | {good}`done`        | [PR159112][PR159112]                                                                                                                                                                                                            |
| device            | indirect clause on declare target directive                 | {part}`In Progress` |                                                                                                                                                                                                                                 |
| device            | allow virtual functions calls for mapped object on device   | {part}`partial`     |                                                                                                                                                                                                                                 |
| device            | interop construct                                           | {part}`partial`     | parsing/sema done: [D98558][D98558], [D98834][D98834], [D98815][D98815]                                                                                                                                                         |
| device            | assorted routines for querying interoperable properties     | {part}`partial`     | [D106674][D106674]                                                                                                                                                                                                              |
| loop              | Loop tiling transformation                                  | {good}`done`        | [D76342][D76342]                                                                                                                                                                                                                |
| loop              | Loop unrolling transformation                               | {good}`done`        | [D99459][D99459]                                                                                                                                                                                                                |
| loop              | 'reproducible'/'unconstrained' modifiers in 'order' clause  | {part}`partial`     | [D127855][D127855]                                                                                                                                                                                                              |
| memory management | alignment for allocate directive and clause                 | {good}`done`        | [D115683][D115683]                                                                                                                                                                                                              |
| memory management | 'allocator' modifier for allocate clause                    | {good}`done`        | [PR114883][PR114883]                                                                                                                                                                                                            |
| memory management | 'align' modifier for allocate clause                        | {good}`done`        | [PR121814][PR121814]                                                                                                                                                                                                            |
| memory management | new memory management routines                              | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| memory management | changes to omp_alloctrait_key enum                          | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| memory model      | seq_cst clause on flush construct                           | {good}`done`        | [PR114072][PR114072]                                                                                                                                                                                                            |
| misc              | 'omp_all_memory' keyword and use in 'depend' clause         | {good}`done`        | [D125828][D125828], [D126321][D126321]                                                                                                                                                                                          |
| misc              | error directive                                             | {good}`done`        | [D139166][D139166]                                                                                                                                                                                                              |
| misc              | scope construct                                             | {good}`done`        | [D157933][D157933], [PR109197][PR109197]                                                                                                                                                                                        |
| misc              | routines for controlling and querying team regions          | {part}`partial`     | [D95003][D95003] (libomp only)                                                                                                                                                                                                  |
| misc              | omp_display_env routine                                     | {good}`done`        | [D74956][D74956]                                                                                                                                                                                                                |
| misc              | extended OMP_PLACES syntax                                  | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| misc              | OMP_NUM_TEAMS and OMP_TEAMS_THREAD_LIMIT env vars           | {good}`done`        | [D138769][D138769]                                                                                                                                                                                                              |
| misc              | 'target_device' selector in context specifier               | {part}`worked on`   |                                                                                                                                                                                                                                 |
| misc              | begin/end declare variant                                   | {good}`done`        | [D71179][D71179]                                                                                                                                                                                                                |
| misc              | dispatch construct and function variant argument adjustment | {part}`worked on`   | [D99537][D99537], [D99679][D99679]                                                                                                                                                                                              |
| misc              | assumes directives                                          | {part}`worked on`   |                                                                                                                                                                                                                                 |
| misc              | assume directive                                            | {good}`done`        |                                                                                                                                                                                                                                 |
| misc              | nothing directive                                           | {good}`done`        | [D123286][D123286]                                                                                                                                                                                                              |
| misc              | masked construct and related combined constructs            | {good}`done`        | [D99995][D99995], [D100514][D100514], [PR121741][PR121741] (parallel_masked_taskloop) [PR121746][PR121746] (parallel_masked_task_loop_simd), [PR121914][PR121914] (masked_taskloop) [PR121916][PR121916] (masked_taskloop_simd) |
| misc              | default(firstprivate) & default(private)                    | {good}`done`        | [D75591][D75591] (firstprivate), [D125912][D125912] (private)                                                                                                                                                                   |
| other             | deprecating master construct                                | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| OMPT              | changes to ompt_scope_endpoint_t enum                       | {good}`done`        | [D90752][D90752]                                                                                                                                                                                                                |
| OMPT              | new barrier types added to ompt_sync_region_t enum          | {good}`done`        | [D90752][D90752]                                                                                                                                                                                                                |
| OMPT              | async data transfers added to ompt_target_data_op_t enum    | {good}`done`        | [D90752][D90752]                                                                                                                                                                                                                |
| OMPT              | new barrier state values added to ompt_state_t enum         | {good}`done`        | [D90752][D90752]                                                                                                                                                                                                                |
| OMPT              | new 'emi' callbacks for external monitoring interfaces      | {good}`done`        |                                                                                                                                                                                                                                 |
| OMPT              | device tracing interface                                    | {part}`in progress` | jplehr                                                                                                                                                                                                                          |
| task              | 'strict' modifier for taskloop construct                    | {none}`unclaimed`   |                                                                                                                                                                                                                                 |
| task              | inoutset in depend clause                                   | {good}`done`        | [D97085][D97085], [D118383][D118383]                                                                                                                                                                                            |
| task              | nowait clause on taskwait                                   | {part}`partial`     | parsing/sema done: [D131830][D131830], [D141531][D141531]                                                                                                                                                                       |

(openmp-5-2-implementation-details)=

## OpenMP 5.2 Implementation Details

The following table provides a quick overview of various OpenMP 5.2 features
and their implementation status. Please post on the
[Discourse forums (Runtimes - OpenMP category)] for more
information or if you want to help with the
implementation.

| Feature                                                 | C/C++ Status      | Fortran Status    | Reviews                                  |
| ------------------------------------------------------- | ----------------- | ----------------- | ---------------------------------------- |
| omp_in_explicit_task()                                  | {none}`unclaimed` | {none}`unclaimed` |                                          |
| semantics of explicit_task_var and implicit_task_var    | {none}`unclaimed` | {none}`unclaimed` |                                          |
| ompx sentinel for C/C++ directive extensions            | {none}`unclaimed` | {none}`unclaimed` |                                          |
| ompx prefix for clause extensions                       | {none}`unclaimed` | {none}`unclaimed` |                                          |
| if clause on teams construct                            | {none}`unclaimed` | {none}`unclaimed` |                                          |
| step modifier added                                     | {none}`unclaimed` | {none}`unclaimed` |                                          |
| declare mapper: Add iterator modifier on map clause     | {none}`unclaimed` | {none}`unclaimed` |                                          |
| declare mapper: Add iterator modifier on map clause     | {none}`unclaimed` | {none}`unclaimed` |                                          |
| memspace and traits modifiers to uses allocator i       | {none}`unclaimed` | {none}`unclaimed` |                                          |
| Add otherwise clause to metadirectives                  | {none}`unclaimed` | {none}`unclaimed` |                                          |
| doacross clause with support for omp_cur_iteration      | {none}`unclaimed` | {none}`unclaimed` |                                          |
| position of interop_type in init clause on iterop       | {none}`unclaimed` | {none}`unclaimed` |                                          |
| implicit map type for target enter/exit data            | {none}`unclaimed` | {none}`unclaimed` |                                          |
| work OMPT type for work-sharing loop constructs         | {good}`done`      | {good}`done`      | [PR189347][PR189347], [PR97429][PR97429] |
| extend ompt_dispatch_t for new enum values              | {good}`done`      | {good}`done`      | [D122107][D122107]                       |
| allocate and firstprivate on scope directive            | {none}`unclaimed` | {none}`unclaimed` |                                          |
| Change loop consistency for order clause                | {none}`unclaimed` | {none}`unclaimed` |                                          |
| Add memspace and traits modifiers to uses_allocators    | {none}`unclaimed` | {none}`unclaimed` |                                          |
| Keep original base pointer on map w/o matched candidate | {none}`unclaimed` | {none}`unclaimed` |                                          |
| Pure procedure support for certain directives           | {none}`N/A`       | {none}`unclaimed` |                                          |
| ALLOCATE statement support for allocators               | {none}`N/A`       | {none}`unclaimed` |                                          |
| dispatch construct extension to support end directive   | {none}`N/A`       | {none}`unclaimed` |                                          |

(openmp-5-2-deprecations)=

## OpenMP 5.2 Deprecations

|                                                          | C/C++ Status      | Fortran Status    | Reviews |
| -------------------------------------------------------- | ----------------- | ----------------- | ------- |
| Linear clause syntax                                     | {none}`unclaimed` | {none}`unclaimed` |         |
| The minus operator                                       | {none}`unclaimed` | {none}`unclaimed` |         |
| Map clause modifiers without commas                      | {none}`unclaimed` | {none}`unclaimed` |         |
| The use of allocate directives with ALLOCATE statement   | {good}`N/A`       | {none}`unclaimed` |         |
| uses_allocators list syntax                              | {none}`unclaimed` | {none}`unclaimed` |         |
| The default clause on metadirectives                     | {none}`unclaimed` | {none}`unclaimed` |         |
| The delimited form of the declare target directive       | {none}`unclaimed` | {good}`N/A`       |         |
| The use of the to clause on the declare target directive | {none}`unclaimed` | {none}`unclaimed` |         |
| The syntax of the destroy clause on the depobj construct | {none}`unclaimed` | {none}`unclaimed` |         |
| keyword source and sink as task-dependence modifiers     | {none}`unclaimed` | {none}`unclaimed` |         |
| interop types in any position on init clause of interop  | {none}`unclaimed` | {none}`unclaimed` |         |
| ompd prefix usage for some ICVs                          | {none}`unclaimed` | {none}`unclaimed` |         |

(openmp-6-0-implementation-details)=

## OpenMP 6.0 Implementation Details

The following table provides a quick overview of various OpenMP 6.0 features
and their implementation status. Please post on the
[Discourse forums (Runtimes - OpenMP category)] for more
information or if you want to help with the
implementation.

| Feature                                                   | C/C++ Status        | Fortran Status      | Reviews                                                                                                                             |
| --------------------------------------------------------- | ------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| free-agent threads                                        | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| threadset clause                                          | {part}`partial`     | {none}`unclaimed`   | Parse/Sema/Codegen: [PR13580][PR13580]                                                                                              |
| Recording of task graphs                                  | {part}`in progress` | {part}`in progress` | clang: jtb20, flang: kparzysz                                                                                                       |
| Parallel inductions                                       | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| init_complete for scan directive                          | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| loop interchange transformation                           | {good}`done`        | {none}`unclaimed`   | Clang (interchange): [PR93022][PR93022] Clang (permutation): [PR92030][PR92030]                                                     |
| loop reverse transformation                               | {good}`done`        | {none}`unclaimed`   | [PR92916][PR92916]                                                                                                                  |
| loop stripe transformation                                | {good}`done`        | {none}`unclaimed`   | [PR119891][PR119891]                                                                                                                |
| loop fusion transformation                                | {part}`in progress` | {good}`done`        | [PR139293][PR139293] [PR161213][PR161213] [PR168898][PR168898]                                                                      |
| loop index set splitting transformation with count clause | {part}`in progress` | {none}`unclaimed`   | @amitamd7                                                                                                                           |
| loop transformation apply clause                          | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| loop fuse transformation                                  | {good}`done`        | {none}`unclaimed`   |                                                                                                                                     |
| workdistribute construct                                  |                     | {none}`in progress` | @skc7, @mjklemm                                                                                                                     |
| task_iteration                                            | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| memscope clause for atomic and flush                      | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| transparent clause (hull tasks)                           | {part}`partial`     | {none}`unclaimed`   | Clang parsing/sema [PR174646][PR174646]                                                                                             |
| rule-based compound directives                            | {part}`In Progress` | {part}`In Progress` | kparzysz Testing for Fortran missing                                                                                                |
| C23, C++23                                                | {none}`unclaimed`   |                     |                                                                                                                                     |
| Fortran 2023                                              |                     | {none}`unclaimed`   |                                                                                                                                     |
| decl attribute for declarative directives                 | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| C attribute syntax                                        | {none}`unclaimed`   |                     |                                                                                                                                     |
| pure directives in DO CONCURRENT                          |                     | {none}`unclaimed`   |                                                                                                                                     |
| Optional argument for all clauses                         | {part}`partial`     | {part}`In Progress` | Parse/Sema (nowait): [PR159628][PR159628]                                                                                           |
| Function references for locator list items                | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| All clauses accept directive name modifier                | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Extensions to depobj construct                            | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Extensions to atomic construct                            | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Private reductions                                        | {part}`mostly`      | {none}`unclaimed`   | Parse/Sema: [PR129938][PR129938] Codegen: [PR134709][PR134709]                                                                      |
| Self maps                                                 | {part}`partial`     | {none}`unclaimed`   | parsing/sema done: [PR129888][PR129888]                                                                                             |
| Release map type for declare mapper                       | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Extensions to interop construct                           | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| no_openmp_constructs                                      | {good}`done`        | {none}`unclaimed`   | [PR125933][PR125933]                                                                                                                |
| safe_sync and progress with identifier and API            | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| OpenMP directives in concurrent loop regions              | {good}`done`        | {none}`unclaimed`   | [PR125621][PR125621]                                                                                                                |
| atomics constructs on concurrent loop regions             | {good}`done`        | {none}`unclaimed`   | [PR125621][PR125621]                                                                                                                |
| Loop construct with DO CONCURRENT                         |                     | {part}`In Progress` |                                                                                                                                     |
| device_type clause for target construct                   | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| nowait for ancestor target directives                     | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| New API for devices' num_teams/thread_limit               | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Host and device environment variables                     | {part}`in progress` | {none}`unclaimed`   | @amitamd7                                                                                                                           |
| num_threads ICV and clause accepts list                   | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Numeric names for environment variables                   | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Increment between places for OMP_PLACES                   | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| OMP_AVAILABLE_DEVICES envirable                           | {none}`unclaimed`   | {none}`unclaimed`   | (should wait for "Traits for default device envirable" being done)                                                                  |
| Traits for default device envirable                       | {part}`in progress` | {none}`unclaimed`   | ro-i                                                                                                                                |
| Optionally omit array length expression                   | {good}`done`        | {none}`unclaimed`   | (Parse) [PR148048][PR148048], (Sema) [PR152786][PR152786]                                                                           |
| Canonical loop sequences                                  | {part}`in progress` | {part}`in progress` | Clang: [PR139293][PR139293]                                                                                                         |
| Clarifications to Fortran map semantics                   | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| default clause at target construct                        | {good}`done`        | {none}`unclaimed`   | [PR162910][PR162910]                                                                                                                |
| ref count update use_device_{ptr, addr}                   | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Clarifications to implicit reductions                     | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| ref modifier for map clauses                              | {part}`In Progress` | {none}`unclaimed`   |                                                                                                                                     |
| map-type modifiers in arbitrary position                  | {good}`done`        | {none}`unclaimed`   | [PR90499][PR90499]                                                                                                                  |
| Lift nesting restriction on concurrent loop               | {good}`done`        | {none}`unclaimed`   | [PR125621][PR125621]                                                                                                                |
| priority clause for target constructs                     | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| changes to target_data construct                          | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Non-const do_not_sync for nowait/nogroup                  | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| need_device_addr modifier for adjust_args clause          | {part}`partial`     | {none}`unclaimed`   | Parsing/Sema: [PR143442][PR143442] [PR149586][PR149586]                                                                             |
| need_device_ptr modifier for adjust_args clause           | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                                                     |
| Prescriptive num_threads                                  | {good}`done`        | {none}`unclaimed`   | [PR160659][PR160659] [PR146403][PR146403] [PR146404][PR146404] [PR146405][PR146405]                                                 |
| Message and severity clauses                              | {good}`done`        | {none}`unclaimed`   | [PR146093][PR146093]                                                                                                                |
| Local clause on declare target                            | {good}`done`        | {none}`unclaimed`   | clang Parse/Sema: [PR186281][PR186281] clang Codegen: [PR196431][PR196431]                                                          |
| groupprivate directive                                    | {part}`In Progress` | {part}`partial`     | Flang: kparzysz, mjklemm Flang parser: [PR153807][PR153807] Flang sema: [PR154779][PR154779] Clang parse/sema: [PR158134][PR158134] |
| variable-category on default clause                       | {good}`done`        | {none}`unclaimed`   |                                                                                                                                     |
| Changes to omp_target_is_accessible                       | {part}`In Progress` | {part}`In Progress` |                                                                                                                                     |
| defaultmap implicit-behavior 'storage'                    | {good}`done`        | {none}`unclaimed`   | [PR158336][PR158336]                                                                                                                |
| defaultmap implicit-behavior 'private'                    | {good}`done`        | {none}`unclaimed`   | [PR158712][PR158712]                                                                                                                |
| OMPT: ompt_get_buffer_limits entry point                  | {part}`partial`     | {good}`N/A`         | Definition: [PR195829][PR195829]                                                                                                    |
| OMPT: ompt_any_record_ompt_t for device tracing           | {good}`done`        | {good}`N/A`         | [PR195829][PR195829]                                                                                                                |
| OMPT: ompt_target_data_transfer_rect(_async) & subvolume  | {part}`partial`     | {good}`N/A`         | Enum: [PR195829][PR195829]                                                                                                          |
| OMPT: ompt_target_data_transfer(_async)                   | {part}`partial`     | {good}`N/A`         | Enum: [PR195829][PR195829]                                                                                                          |
| OMPT: ompt_target_data_memset(_async)                     | {part}`partial`     | {good}`N/A`         | Enum: [PR195829][PR195829] Callbacks: [PR194168][PR194168]                                                                          |
| OMPT: workdistribute work callback enum                   | {part}`partial`     | {good}`N/A`         | Enum: [PR195829][PR195829]                                                                                                          |
| OMPT: transparent task flag enum (importing/exporting)    | {part}`partial`     | {good}`N/A`         | Enum: [PR195829][PR195829]                                                                                                          |
| OMPT: dependence type {out, inout}_all_memory             | {part}`partial`     | {good}`N/A`         | Enum: [PR195829][PR195829]                                                                                                          |
| OMPT: removed master callback                             | {none}`unclaimed`   | {good}`N/A`         |                                                                                                                                     |
| OMPT: removed sync_region_barrier(_implicit) enum value   | {none}`unclaimed`   | {good}`N/A`         |                                                                                                                                     |

(openmp-6-0-deprecations)=

## OpenMP 6.0 Deprecations

|                                                       | C/C++ Status      | Fortran Status | Reviews |
| ----------------------------------------------------- | ----------------- | -------------- | ------- |
| OMPT: non-emi target callbacks                        | {none}`unclaimed` | {good}`N/A`    |         |
| OMPT: transfer_to_device / transfer_from_device enums | {none}`unclaimed` | {good}`N/A`    |         |

(openmp-6-1-implementation-details)=

## OpenMP 6.1 Implementation Details (Experimental)

The following table provides a quick overview over various OpenMP 6.1 features
and their implementation status. Since OpenMP 6.1 has not yet been released, the
following features are experimental and are subject to change at any time.
Please post on the [Discourse forums (Runtimes - OpenMP category)] for more
information or if you want to help with the
implementation.

| Feature                                                                                         | C/C++ Status        | Fortran Status      | Reviews                                                                                                  |
| ----------------------------------------------------------------------------------------------- | ------------------- | ------------------- | -------------------------------------------------------------------------------------------------------- |
| dyn_groupprivate clause                                                                         | {part}`partial`     | {part}`In Progress` | C/C++: Host device support missing                                                                       |
| loop flatten transformation                                                                     | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                          |
| loop grid/tile modifiers for sizes clause                                                       | {none}`unclaimed`   | {none}`unclaimed`   |                                                                                                          |
| attach map-type modifier                                                                        | {part}`In Progress` | {none}`unclaimed`   | C/C++: @abhinavgaba; RT: @abhinavgaba ([PR149036][PR149036], [PR158370][PR158370])                       |
| need_device_ptr modifier for adjust_args clause                                                 | {part}`partial`     | {none}`unclaimed`   | Clang Parsing/Sema: [PR168905][PR168905] [PR169558][PR169558]                                            |
| fallback modifier for use_device_ptr clause                                                     | {good}`done`        | {none}`unclaimed`   | Clang: @abhinavgaba ([PR170578][PR170578], [PR173931][PR173931]) RT: @abhinavgaba ([PR169603][PR169603]) |
| dims modifier for num_teams, thread_limit, and num_threads (multidimensional teams and leagues) | {part}`partial`     | {part}`In Progress` | C/C++: @kevinsala ([PR206412]); Fortran: @skc7, @kparzysz, @mjklemm                                      |

## OpenMP Extensions

The following table provides a quick overview over various OpenMP
extensions and their implementation status. These extensions are not
currently defined by any standard, so links to associated LLVM
documentation are provided. As these extensions mature, they will be
considered for standardization. Please post on the
[Discourse forums (Runtimes - OpenMP category)] to provide feedback.

| Category         | Feature                                                                                                                 | Status             | Reviews                                                        |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------------------------------------------------------- |
| atomic extension | ['atomic' strictly nested within 'teams'](https://openmp.llvm.org/docs/openacc/OpenMPExtensions.html#atomicWithinTeams) | {part}`prototyped` | [D126323][D126323]                                             |
| device extension | ['ompx_hold' map type modifier](https://openmp.llvm.org/docs/openacc/OpenMPExtensions.html#ompx-hold)                   | {part}`prototyped` | [D106509][D106509], [D106510][D106510]                         |
| device extension | ['ompx_bare' clause on 'target teams' construct](https://www.osti.gov/servlets/purl/2205717)                            | {part}`prototyped` | [PR66844][PR66844], [PR70612][PR70612]                         |
| device extension | Multi-dim 'num_teams' and 'thread_limit' clause on 'target teams ompx_bare' construct                                   | {part}`partial`    | [PR99732][PR99732], [PR101407][PR101407], [PR102715][PR102715] |


[D50522]: https://reviews.llvm.org/D50522
[D51107]: https://reviews.llvm.org/D51107
[D51233]: https://reviews.llvm.org/D51233
[D51378]: https://reviews.llvm.org/D51378
[D52359]: https://reviews.llvm.org/D52359
[D52625]: https://reviews.llvm.org/D52625
[D52780]: https://reviews.llvm.org/D52780
[D53079]: https://reviews.llvm.org/D53079
[D53380]: https://reviews.llvm.org/D53380
[D53513]: https://reviews.llvm.org/D53513
[D54342]: https://reviews.llvm.org/D54342
[D54441]: https://reviews.llvm.org/D54441
[D55078]: https://reviews.llvm.org/D55078
[D55719]: https://reviews.llvm.org/D55719
[D55892]: https://reviews.llvm.org/D55892
[D55982]: https://reviews.llvm.org/D55982
[D56326]: https://reviews.llvm.org/D56326
[D57576]: https://reviews.llvm.org/D57576
[D58074]: https://reviews.llvm.org/D58074
[D58523]: https://reviews.llvm.org/D58523
[D58638]: https://reviews.llvm.org/D58638
[D59474]: https://reviews.llvm.org/D59474
[D60972]: https://reviews.llvm.org/D60972
[D64095]: https://reviews.llvm.org/D64095
[D67294]: https://reviews.llvm.org/D67294
[D69204]: https://reviews.llvm.org/D69204
[D71179]: https://reviews.llvm.org/D71179
[D71830]: https://reviews.llvm.org/D71830
[D71847]: https://reviews.llvm.org/D71847
[D74144]: https://reviews.llvm.org/D74144
[D74956]: https://reviews.llvm.org/D74956
[D75591]: https://reviews.llvm.org/D75591
[D76342]: https://reviews.llvm.org/D76342
[D83061]: https://reviews.llvm.org/D83061
[D83062]: https://reviews.llvm.org/D83062
[D84422]: https://reviews.llvm.org/D84422
[D84711]: https://reviews.llvm.org/D84711
[D84712]: https://reviews.llvm.org/D84712
[D90752]: https://reviews.llvm.org/D90752
[D91944]: https://reviews.llvm.org/D91944
[D92427]: https://reviews.llvm.org/D92427
[D95003]: https://reviews.llvm.org/D95003
[D97085]: https://reviews.llvm.org/D97085
[D98558]: https://reviews.llvm.org/D98558
[D98815]: https://reviews.llvm.org/D98815
[D98834]: https://reviews.llvm.org/D98834
[D99459]: https://reviews.llvm.org/D99459
[D99537]: https://reviews.llvm.org/D99537
[D99679]: https://reviews.llvm.org/D99679
[D99914]: https://reviews.llvm.org/D99914
[D99995]: https://reviews.llvm.org/D99995
[D100514]: https://reviews.llvm.org/D100514
[D105648]: https://reviews.llvm.org/D105648
[D106509]: https://reviews.llvm.org/D106509
[D106510]: https://reviews.llvm.org/D106510
[D106674]: https://reviews.llvm.org/D106674
[D109635]: https://reviews.llvm.org/D109635
[D113540]: https://reviews.llvm.org/D113540
[D115683]: https://reviews.llvm.org/D115683
[D116261]: https://reviews.llvm.org/D116261
[D116637]: https://reviews.llvm.org/D116637
[D118383]: https://reviews.llvm.org/D118383
[D118547]: https://reviews.llvm.org/D118547
[D118632]: https://reviews.llvm.org/D118632
[D120007]: https://reviews.llvm.org/D120007
[D120200]: https://reviews.llvm.org/D120200
[D120290]: https://reviews.llvm.org/D120290
[D122107]: https://reviews.llvm.org/D122107
[D123235]: https://reviews.llvm.org/D123235
[D123286]: https://reviews.llvm.org/D123286
[D125828]: https://reviews.llvm.org/D125828
[D125912]: https://reviews.llvm.org/D125912
[D126321]: https://reviews.llvm.org/D126321
[D126323]: https://reviews.llvm.org/D126323
[D127855]: https://reviews.llvm.org/D127855
[D128347]: https://reviews.llvm.org/D128347
[D131830]: https://reviews.llvm.org/D131830
[D136103]: https://reviews.llvm.org/D136103
[D138769]: https://reviews.llvm.org/D138769
[D139166]: https://reviews.llvm.org/D139166
[D141531]: https://reviews.llvm.org/D141531
[D141540]: https://reviews.llvm.org/D141540
[D141545]: https://reviews.llvm.org/D141545
[D144634]: https://reviews.llvm.org/D144634
[D145823]: https://reviews.llvm.org/D145823
[D146418]: https://reviews.llvm.org/D146418
[D152054]: https://reviews.llvm.org/D152054
[D155003]: https://reviews.llvm.org/D155003
[D157933]: https://reviews.llvm.org/D157933
[PR66844]: https://github.com/llvm/llvm-project/pull/66844
[PR70612]: https://github.com/llvm/llvm-project/pull/70612
[PR13580]: https://github.com/llvm/llvm-project/pull/13580
[PR90499]: https://github.com/llvm/llvm-project/pull/90499
[PR92030]: https://github.com/llvm/llvm-project/pull/92030
[PR92916]: https://github.com/llvm/llvm-project/pull/92916
[PR93022]: https://github.com/llvm/llvm-project/pull/93022
[PR97429]: https://github.com/llvm/llvm-project/pull/97429
[PR99732]: https://github.com/llvm/llvm-project/pull/99732
[PR101101]: https://github.com/llvm/llvm-project/pull/101101
[PR101407]: https://github.com/llvm/llvm-project/pull/101407
[PR102715]: https://github.com/llvm/llvm-project/pull/102715
[PR109197]: https://github.com/llvm/llvm-project/pull/109197
[PR114072]: https://github.com/llvm/llvm-project/pull/114072
[PR114883]: https://github.com/llvm/llvm-project/pull/114883
[PR119891]: https://github.com/llvm/llvm-project/pull/119891
[PR121741]: https://github.com/llvm/llvm-project/pull/121741
[PR121746]: https://github.com/llvm/llvm-project/pull/121746
[PR121814]: https://github.com/llvm/llvm-project/pull/121814
[PR121914]: https://github.com/llvm/llvm-project/pull/121914
[PR121916]: https://github.com/llvm/llvm-project/pull/121916
[PR125621]: https://github.com/llvm/llvm-project/pull/125621
[PR125933]: https://github.com/llvm/llvm-project/pull/125933
[PR128640]: https://github.com/llvm/llvm-project/pull/128640
[PR129888]: https://github.com/llvm/llvm-project/pull/129888
[PR129938]: https://github.com/llvm/llvm-project/pull/129938
[PR134709]: https://github.com/llvm/llvm-project/pull/134709
[PR138294]: https://github.com/llvm/llvm-project/pull/138294
[PR139293]: https://github.com/llvm/llvm-project/pull/139293
[PR143442]: https://github.com/llvm/llvm-project/pull/143442
[PR144635]: https://github.com/llvm/llvm-project/pull/144635
[PR146093]: https://github.com/llvm/llvm-project/pull/146093
[PR146403]: https://github.com/llvm/llvm-project/pull/146403
[PR146404]: https://github.com/llvm/llvm-project/pull/146404
[PR146405]: https://github.com/llvm/llvm-project/pull/146405
[PR148048]: https://github.com/llvm/llvm-project/pull/148048
[PR149036]: https://github.com/llvm/llvm-project/pull/149036
[PR149586]: https://github.com/llvm/llvm-project/pull/149586
[PR152786]: https://github.com/llvm/llvm-project/pull/152786
[PR153683]: https://github.com/llvm/llvm-project/pull/153683
[PR153807]: https://github.com/llvm/llvm-project/pull/153807
[PR154779]: https://github.com/llvm/llvm-project/pull/154779
[PR157025]: https://github.com/llvm/llvm-project/pull/157025
[PR158134]: https://github.com/llvm/llvm-project/pull/158134
[PR158336]: https://github.com/llvm/llvm-project/pull/158336
[PR158370]: https://github.com/llvm/llvm-project/pull/158370
[PR158712]: https://github.com/llvm/llvm-project/pull/158712
[PR159112]: https://github.com/llvm/llvm-project/pull/159112
[PR159628]: https://github.com/llvm/llvm-project/pull/159628
[PR160659]: https://github.com/llvm/llvm-project/pull/160659
[PR161213]: https://github.com/llvm/llvm-project/pull/161213
[PR162910]: https://github.com/llvm/llvm-project/pull/162910
[PR168898]: https://github.com/llvm/llvm-project/pull/168898
[PR168905]: https://github.com/llvm/llvm-project/pull/168905
[PR169558]: https://github.com/llvm/llvm-project/pull/169558
[PR169603]: https://github.com/llvm/llvm-project/pull/169603
[PR170578]: https://github.com/llvm/llvm-project/pull/170578
[PR173931]: https://github.com/llvm/llvm-project/pull/173931
[PR174646]: https://github.com/llvm/llvm-project/pull/174646
[PR174659]: https://github.com/llvm/llvm-project/pull/174659
[PR186281]: https://github.com/llvm/llvm-project/pull/186281
[PR189347]: https://github.com/llvm/llvm-project/pull/189347
[PR194168]: https://github.com/llvm/llvm-project/pull/194168
[PR195829]: https://github.com/llvm/llvm-project/pull/195829
[PR196431]: https://github.com/llvm/llvm-project/pull/196431

[discourse forums (runtimes - openmp category)]: https://discourse.llvm.org/c/runtimes/openmp/35
