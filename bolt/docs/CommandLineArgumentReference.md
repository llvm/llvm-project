# BOLT - a post-link optimizer developed to speed up large applications

## SYNOPSIS

`llvm-bolt <executable> [-o outputfile] <executable>.bolt [-data=perf.fdata] [options]`

## OPTIONS

### Generic options:

- `-h`

  Alias for --help

- `--help`

  Display available options (--help-hidden for more)

- `--help-hidden`

  Display all available options

- `--help-list`

  Display list of available options (--help-list-hidden for more)

- `--help-list-hidden`

  Display list of all available options

- `--version`

  Display the version of this program

### Output options:

- `--bolt-info`

  Write bolt info section in the output binary

- `-o <string>`

  output file

- `-w <string>`

  Save recorded profile to a file

### BOLT generic options:

- `--align-text=<uint>`

  Alignment of .text section

- `--allow-stripped`

  Allow processing of stripped binaries

- `--alt-inst-feature-size=<uint>`

  Size of feature field in .altinstructions

- `--alt-inst-has-padlen`

  Specify that .altinstructions has padlen field

- `--asm-dump[=<dump folder>]`

  Dump function into assembly

- `-b`

  Alias for -data

- `--bolt-id=<string>`

  Add any string to tag this execution in the output binary via bolt info section

- `--break-funcs=<func1,func2,func3,...>`

  List of functions to core dump on (debugging)

- `--check-encoding`

  Perform verification of LLVM instruction encoding/decoding. Every instruction
  in the input is decoded and re-encoded. If the resulting bytes do not match
  the input, a warning message is printed.

- `--comp-dir-override=<string>`

  Overrides DW_AT_comp_dir, and provides an alternative base location, which is
  used with DW_AT_dwo_name to construct a path to *.dwo files.

- `--create-debug-names-section`

  Creates .debug_names section, if the input binary doesn't have it already, for
  DWARF5 CU/TUs.

- `--cu-processing-batch-size=<uint>`

  Specifies the size of batches for processing CUs. Higher number has better
  performance, but more memory usage. Default value is 1.

- `--data=<string>`

  data file

- `--data2=<string>`

  data file

- `--debug-skeleton-cu`

  Prints out offsets for abbrev and debug_info of Skeleton CUs that get patched.

- `--debug-thread-count=<uint>`

  Specifies the number of threads to be used when processing DWO debug information.

- `--dot-tooltip-code`

  Add basic block instructions as tool tips on nodes

- `--dump-alt-instructions`

  Dump Linux alternative instructions info

- `--dump-cg=<string>`

  Dump callgraph to the given file

- `--dump-data`

  Dump parsed bolt data for debugging

- `--dump-dot-all`

  Dump function CFGs to graphviz format after each stage;enable '-print-loops'
  for color-coded blocks

- `--dump-linux-exceptions`

  Dump Linux kernel exception table

- `--dump-orc`

  Dump raw ORC unwind information (sorted)

- `--dump-para-sites`

  Dump Linux kernel paravitual patch sites

- `--dump-pci-fixups`

  Dump Linux kernel PCI fixup table

- `--dump-smp-locks`

  Dump Linux kernel SMP locks

- `--dump-static-calls`

  Dump Linux kernel static calls

- `--dump-static-keys`

  Dump Linux kernel static keys jump table

- `--dwarf-output-path=<string>`

  Path to where .dwo files or dwp file will be written out to.

- `--dwp=<string>`

  Path and name to DWP file.

- `--dyno-stats`

  Print execution info based on profile

- `--dyno-stats-all`

  Print dyno stats after each stage

- `--dyno-stats-scale=<uint>`

  Scale to be applied while reporting dyno stats

- `--enable-bat`

  Write BOLT Address Translation tables

- `--force-data-relocations`

  Force relocations to data sections to always be processed

- `--force-patch`

  Force patching of original entry points

- `--funcs=<func1,func2,func3,...>`

  Limit optimizations to functions from the list

- `--funcs-file=<string>`

  File with list of functions to optimize

- `--funcs-file-no-regex=<string>`

  File with list of functions to optimize (non-regex)

- `--funcs-no-regex=<func1,func2,func3,...>`

  Limit optimizations to functions from the list (non-regex)

- `--hot-data`

  Hot data symbols support (relocation mode)

- `--hot-functions-at-end`

  If reorder-functions is used, order functions putting hottest last

- `--hot-text`

  Generate hot text symbols. Apply this option to a precompiled binary that
  manually calls into hugify, such that at runtime hugify call will put hot code
  into 2M pages. This requires relocation.

- `--hot-text-move-sections=<sec1,sec2,sec3,...>`

  List of sections containing functions used for hugifying hot text. BOLT makes
  sure these functions are not placed on the same page as the hot text.
  (default='.stub,.mover').

- `--insert-retpolines`

  Run retpoline insertion pass

- `--keep-aranges`

  Keep or generate .debug_aranges section if .gdb_index is written

- `--keep-tmp`

  Preserve intermediate .o file

- `--lite`

  Skip processing of cold functions

- `--log-file=<string>`

  Redirect journaling to a file instead of stdout/stderr

- `--long-jump-labels`

  Always use long jumps/nops for Linux kernel static keys

- `--match-profile-with-function-hash`

  Match profile with function hash

- `--max-data-relocations=<uint>`

  Maximum number of data relocations to process

- `--max-funcs=<uint>`

  Maximum number of functions to process

- `--no-huge-pages`

  Use regular size pages for code alignment

- `--no-threads`

  Disable multithreading

- `--pad-funcs=<func1:pad1,func2:pad2,func3:pad3,...>`

  List of functions to pad with amount of bytes

- `--print-mappings`

  Print mappings in the legend, between characters/blocks and text sections
  (default false).


- `--profile-format=<value>`

  Format to dump profile output in aggregation mode, default is fdata
  - `fdata`: offset-based plaintext format
  - `yaml`: dense YAML representation

- `--r11-availability=<value>`

  Determine the availability of r11 before indirect branches
  - `never`: r11 not available
  - `always`: r11 available before calls and jumps
  - `abi`: r11 available before calls but not before jumps

- `--relocs`

  Use relocations in the binary (default=autodetect)

- `--remove-symtab`

  Remove .symtab section

- `--reorder-skip-symbols=<symbol1,symbol2,symbol3,...>`

  List of symbol names that cannot be reordered

- `--reorder-symbols=<symbol1,symbol2,symbol3,...>`

  List of symbol names that can be reordered

- `--retpoline-lfence`

  Determine if lfence instruction should exist in the retpoline

- `--skip-funcs=<func1,func2,func3,...>`

  List of functions to skip

- `--skip-funcs-file=<string>`

  File with list of functions to skip

- `--strict`

  Trust the input to be from a well-formed source

- `--tasks-per-thread=<uint>`

  Number of tasks to be created per thread

- `--terminal-trap`

  Assume that execution stops at trap instruction

- `--thread-count=<uint>`

  Number of threads

- `--top-called-limit=<uint>`

  Maximum number of functions to print in top called functions section

- `--trap-avx512`

  In relocation mode trap upon entry to any function that uses AVX-512
  instructions

- `--trap-old-code`

  Insert traps in old function bodies (relocation mode)

- `--update-debug-sections`

  Update DWARF debug sections of the executable

- `--use-gnu-stack`

  Use GNU_STACK program header for new segment (workaround for issues with
  strip/objcopy)

- `--use-old-text`

  Re-use space in old .text if possible (relocation mode)

- `-v <uint>`

  Set verbosity level for diagnostic output

- `--write-dwp`

  Output a single dwarf package file (dwp) instead of multiple non-relocatable
  dwarf object files (dwo).

### BOLT optimization options:

- `--align-blocks`

  Align basic blocks

- `--align-blocks-min-size=<uint>`

  Minimal size of the basic block that should be aligned

- `--align-blocks-threshold=<uint>`

  Align only blocks with frequency larger than containing function execution
  frequency specified in percent. E.g. 1000 means aligning blocks that are 10
  times more frequently executed than the containing function.

- `--align-functions=<uint>`

  Align functions at a given value (relocation mode)

- `--align-functions-max-bytes=<uint>`

  Maximum number of bytes to use to align functions

- `--assume-abi`

  Assume the ABI is never violated

- `--block-alignment=<uint>`

  Boundary to use for alignment of basic blocks

- `--bolt-seed=<uint>`

  Seed for randomization

- `--cg-from-perf-data`

  Use perf data directly when constructing the call graph for stale functions

- `--cg-ignore-recursive-calls`

  Ignore recursive calls when constructing the call graph

- `--cg-use-split-hot-size`

  Use hot/cold data on basic blocks to determine hot sizes for call graph
  functions

- `--cold-threshold=<uint>`

  Tenths of percents of main entry frequency to use as a threshold when
  evaluating whether a basic block is cold (0 means it is only considered cold
  if the block has zero samples). Default: 0

- `--elim-link-veneers`

  Run veneer elimination pass

- `--eliminate-unreachable`

  Eliminate unreachable code

- `--equalize-bb-counts`

  Use same count for BBs that should have equivalent count (used in non-LBR and
  shrink wrapping)

- `--execution-count-threshold=<uint>`

  Perform profiling accuracy-sensitive optimizations only if function execution
  count >= the threshold (default: 0)

- `--fix-block-counts`

  Adjust block counts based on outgoing branch counts

- `--fix-func-counts`

  Adjust function counts based on basic blocks execution count

- `--force-inline=<func1,func2,func3,...>`

  List of functions to always consider for inlining

- `--frame-opt=<value>`

  Optimize stack frame accesses
  - `none`: do not perform frame optimization
  - `hot`: perform FOP on hot functions
  - `all`: perform FOP on all functions

- `--frame-opt-rm-stores`

  Apply additional analysis to remove stores (experimental)

- `--function-order=<string>`

  File containing an ordered list of functions to use for function reordering

- `--generate-function-order=<string>`

  File to dump the ordered list of functions to use for function reordering

- `--generate-link-sections=<string>`

  Generate a list of function sections in a format suitable for inclusion in a
  linker script

- `--group-stubs`

  Share stubs across functions

- `--hugify`

  Automatically put hot code on 2MB page(s) (hugify) at runtime. No manual call
  to hugify is needed in the binary (which is what --hot-text relies on).

- `--icf`

  Fold functions with identical code

- `--icp`

  Alias for --indirect-call-promotion

- `--icp-calls-remaining-percent-threshold=<uint>`

  The percentage threshold against remaining unpromoted indirect call count for
  the promotion for calls

- `--icp-calls-topn`

  Alias for --indirect-call-promotion-calls-topn

- `--icp-calls-total-percent-threshold=<uint>`

  The percentage threshold against total count for the promotion for calls

- `--icp-eliminate-loads`

  Enable load elimination using memory profiling data when performing ICP

- `--icp-funcs=<func1,func2,func3,...>`

  List of functions to enable ICP for

- `--icp-inline`

  Only promote call targets eligible for inlining

- `--icp-jt-remaining-percent-threshold=<uint>`

  The percentage threshold against remaining unpromoted indirect call count for
  the promotion for jump tables

- `--icp-jt-targets`

  Alias for --icp-jump-tables-targets

- `--icp-jt-topn`

  Alias for --indirect-call-promotion-jump-tables-topn

- `--icp-jt-total-percent-threshold=<uint>`

  The percentage threshold against total count for the promotion for jump tables

- `--icp-jump-tables-targets`

  For jump tables, optimize indirect jmp targets instead of indices

- `--icp-mp-threshold`

  Alias for --indirect-call-promotion-mispredict-threshold

- `--icp-old-code-sequence`

  Use old code sequence for promoted calls

- `--icp-top-callsites=<uint>`

  Optimize hottest calls until at least this percentage of all indirect calls
  frequency is covered. 0 = all callsites

- `--icp-topn`

  Alias for --indirect-call-promotion-topn

- `--icp-use-mp`

  Alias for --indirect-call-promotion-use-mispredicts

- `--indirect-call-promotion=<value>`

  Indirect call promotion
  - `none`: do not perform indirect call promotion
  - `calls`: perform ICP on indirect calls
  - `jump-tables`: perform ICP on jump tables
  - `all`: perform ICP on calls and jump tables

- `--indirect-call-promotion-calls-topn=<uint>`

  Limit number of targets to consider when doing indirect call promotion on
  calls. 0 = no limit

- `--indirect-call-promotion-jump-tables-topn=<uint>`

  Limit number of targets to consider when doing indirect call promotion on jump
  tables. 0 = no limit

- `--indirect-call-promotion-topn=<uint>`

  Limit number of targets to consider when doing indirect call promotion. 0 = no
  limit

- `--indirect-call-promotion-use-mispredicts`

  Use misprediction frequency for determining whether or not ICP should be
  applied at a callsite.  The -indirect-call-promotion-mispredict-threshold
  value will be used by this heuristic

- `--infer-fall-throughs`

  Infer execution count for fall-through blocks

- `--infer-stale-profile`

  Infer counts from stale profile data.

- `--inline-all`

  Inline all functions

- `--inline-ap`

  Adjust function profile after inlining

- `--inline-limit=<uint>`

  Maximum number of call sites to inline

- `--inline-max-iters=<uint>`

  Maximum number of inline iterations

- `--inline-memcpy`

  Inline memcpy using 'rep movsb' instruction (X86-only)

- `--inline-small-functions`

  Inline functions if increase in size is less than defined by -inline-small-
  functions-bytes

- `--inline-small-functions-bytes=<uint>`

  Max number of bytes for the function to be considered small for inlining
  purposes

- `--instrument`

  Instrument code to generate accurate profile data

- `--iterative-guess`

  In non-LBR mode, guess edge counts using iterative technique

- `--jt-footprint-optimize-for-icache`

  With jt-footprint-reduction, only process PIC jumptables and turn off other
  transformations that increase code size

- `--jt-footprint-reduction`

  Make jump tables size smaller at the cost of using more instructions at jump
  sites

- `--jump-tables=<value>`

  Jump tables support (default=basic)
  - `none`: do not optimize functions with jump tables
  - `basic`: optimize functions with jump tables
  - `move`: move jump tables to a separate section
  - `split`: split jump tables section into hot and cold based on function
  execution frequency
  - `aggressive`: aggressively split jump tables section based on usage of the
  tables

- `--keep-nops`

  Keep no-op instructions. By default they are removed.

- `--lite-threshold-count=<uint>`

  Similar to '-lite-threshold-pct' but specify threshold using absolute function
  call count. I.e. limit processing to functions executed at least the specified
  number of times.

- `--lite-threshold-pct=<uint>`

  Threshold (in percent) for selecting functions to process in lite mode. Higher
  threshold means fewer functions to process. E.g threshold of 90 means only top
  10 percent of functions with profile will be processed.

- `--match-with-call-graph`

  Match functions with call graph

- `--memcpy1-spec=<func1,func2:cs1:cs2,func3:cs1,...>`

  List of functions with call sites for which to specialize memcpy() for size 1

- `--min-branch-clusters`

  Use a modified clustering algorithm geared towards minimizing branches

- `--name-similarity-function-matching-threshold=<uint>`

  Match functions using namespace and edit distance.

- `--no-inline`

  Disable all inlining (overrides other inlining options)

- `--no-scan`

  Do not scan cold functions for external references (may result in slower binary)

- `--peepholes=<value>`

  Enable peephole optimizations
  - `none`: disable peepholes
  - `double-jumps`: remove double jumps when able
  - `tailcall-traps`: insert tail call traps
  - `useless-branches`: remove useless conditional branches
  - `all`: enable all peephole optimizations

- `--plt=<value>`

  Optimize PLT calls (requires linking with -znow)
  - `none`: do not optimize PLT calls
  - `hot`: optimize executed (hot) PLT calls
  - `all`: optimize all PLT calls

- `--preserve-blocks-alignment`

  Try to preserve basic block alignment

- `--profile-ignore-hash`

  Ignore hash while reading function profile

- `--profile-use-dfs`

  Use DFS order for YAML profile

- `--reg-reassign`

  Reassign registers so as to avoid using REX prefixes in hot code

- `--reorder-blocks=<value>`

  Change layout of basic blocks in a function
  - `none`: do not reorder basic blocks
  - `reverse`: layout blocks in reverse order
  - `normal`: perform optimal layout based on profile
  - `branch-predictor`: perform optimal layout prioritizing branch predictions
  - `cache`: perform optimal layout prioritizing I-cache behavior
  - `cache+`: perform layout optimizing I-cache behavior
  - `ext-tsp`: perform layout optimizing I-cache behavior
  - `cluster-shuffle`: perform random layout of clusters

- `--reorder-data=<section1,section2,section3,...>`

  List of sections to reorder

- `--reorder-data-algo=<value>`

  Algorithm used to reorder data sections
  - `count`: sort hot data by read counts
  - `funcs`: sort hot data by hot function usage and count

- `--reorder-data-inplace`

  Reorder data sections in place

- `--reorder-data-max-bytes=<uint>`

  Maximum number of bytes to reorder

- `--reorder-data-max-symbols=<uint>`

  Maximum number of symbols to reorder

- `--reorder-functions=<value>`

  Reorder and cluster functions (works only with relocations)
  - `none`: do not reorder functions
  - `exec-count`: order by execution count
  - `hfsort`: use hfsort algorithm
  - `hfsort+`: use cache-directed sort
  - `cdsort`: use cache-directed sort
  - `pettis-hansen`: use Pettis-Hansen algorithm
  - `random`: reorder functions randomly
  - `user`: use function order specified by -function-order

- `--reorder-functions-use-hot-size`

  Use a function's hot size when doing clustering

- `--report-bad-layout=<uint>`

  Print top <uint> functions with suboptimal code layout on input

- `--report-stale`

  Print the list of functions with stale profile

- `--runtime-hugify-lib=<string>`

  Specify file name of the runtime hugify library

- `--runtime-instrumentation-lib=<string>`

  Specify file name of the runtime instrumentation library

- `--sctc-mode=<value>`

  Mode for simplify conditional tail calls
  - `always`: always perform sctc
  - `preserve`: only perform sctc when branch direction is preserved
  - `heuristic`: use branch prediction data to control sctc

- `--sequential-disassembly`

  Performs disassembly sequentially

- `--shrink-wrapping-threshold=<uint>`

  Percentage of prologue execution count to use as threshold when evaluating
  whether a block is cold enough to be profitable to move eligible spills there

- `--simplify-conditional-tail-calls`

  Simplify conditional tail calls by removing unnecessary jumps

- `--simplify-rodata-loads`

  Simplify loads from read-only sections by replacing the memory operand with
  the constant found in the corresponding section

- `--split-align-threshold=<uint>`

  When deciding to split a function, apply this alignment while doing the size
  comparison (see -split-threshold). Default value: 2.

- `--split-all-cold`

  Outline as many cold basic blocks as possible

- `--split-eh`

  Split C++ exception handling code

- `--split-functions`

  Split functions into fragments

- `--split-strategy=<value>`

  Strategy used to partition blocks into fragments
  - `profile2`: split each function into a hot and cold fragment using profiling
  information
  - `cdsplit`: split each function into a hot, warm, and cold fragment using
  profiling information
  - `random2`: split each function into a hot and cold fragment at a randomly
  chosen split point (ignoring any available profiling information)
  - `randomN`: split each function into N fragments at a randomly chosen split
  points (ignoring any available profiling information)
  - `all`: split all basic blocks of each function into fragments such that each
  fragment contains exactly a single basic block

- `--split-threshold=<uint>`

  Split function only if its main size is reduced by more than given amount of
  bytes. Default value: 0, i.e. split iff the size is reduced. Note that on some
  architectures the size can increase after splitting.

- `--stale-matching-max-func-size=<uint>`

  The maximum size of a function to consider for inference.

- `--stale-matching-min-matched-block=<uint>`

  Percentage threshold of matched basic blocks at which stale profile inference
  is executed.

- `--stale-threshold=<uint>`

  Maximum percentage of stale functions to tolerate (default: 100)

- `--stoke`

  Turn on the stoke analysis

- `--strip-rep-ret`

  Strip 'repz' prefix from 'repz retq' sequence (on by default)

- `--tail-duplication=<value>`

  Duplicate unconditional branches that cross a cache line
  - `none`: do not apply
  - `aggressive`: aggressive strategy
  - `moderate`: moderate strategy
  - `cache`: cache-aware duplication strategy

- `--tsp-threshold=<uint>`

  Maximum number of hot basic blocks in a function for which to use a precise
  TSP solution while re-ordering basic blocks

- `--use-aggr-reg-reassign`

  Use register liveness analysis to try to find more opportunities for -reg-
  reassign optimization

- `--use-compact-aligner`

  Use compact approach for aligning functions

- `--use-edge-counts`

  Use edge count data when doing clustering

- `--verify-cfg`

  Verify the CFG after every pass

- `--x86-align-branch-boundary-hot-only`

  Only apply branch boundary alignment in hot code

- `--x86-strip-redundant-address-size`

  Remove redundant Address-Size override prefix

### BOLT options in relocation mode:

- `--align-macro-fusion=<value>`

  Fix instruction alignment for macro-fusion (x86 relocation mode)
  - `none`: do not insert alignment no-ops for macro-fusion
  - `hot`: only insert alignment no-ops on hot execution paths (default)
  - `all`: always align instructions to allow macro-fusion

### BOLT instrumentation options:

`llvm-bolt <executable> -instrument [-o outputfile] <instrumented-executable>`

- `--conservative-instrumentation`

  Disable instrumentation optimizations that sacrifice profile accuracy (for
  debugging, default: false)

- `--instrument-calls`

  Record profile for inter-function control flow activity (default: true)

- `--instrument-hot-only`

  Only insert instrumentation on hot functions (needs profile, default: false)

- `--instrumentation-binpath=<string>`

  Path to instrumented binary in case if /proc/self/map_files is not accessible
  due to access restriction issues

- `--instrumentation-file=<string>`

  File name where instrumented profile will be saved (default: /tmp/prof.fdata)

- `--instrumentation-file-append-pid`

  Append PID to saved profile file name (default: false)

- `--instrumentation-no-counters-clear`

  Don't clear counters across dumps (use with instrumentation-sleep-time option)

- `--instrumentation-sleep-time=<uint>`

  Interval between profile writes (default: 0 = write only at program end).
  This is useful for service workloads when you want to dump profile every X
  minutes or if you are killing the program and the profile is not being dumped
  at the end.

- `--instrumentation-wait-forks`

  Wait until all forks of instrumented process will finish (use with
  instrumentation-sleep-time option)

### BOLT printing options:

- `--print-aliases`

  Print aliases when printing objects

- `--print-all`

  Print functions after each stage

- `--print-cfg`

  Print functions after CFG construction

- `--print-debug-info`

  Print debug info when printing functions

- `--print-disasm`

  Print function after disassembly

- `--print-dyno-opcode-stats=<uint>`

  Print per instruction opcode dyno stats and the functionnames:BB offsets of
  the nth highest execution counts

- `--print-dyno-stats-only`

  While printing functions output dyno-stats and skip instructions

- `--print-exceptions`

  Print exception handling data

- `--print-globals`

  Print global symbols after disassembly

- `--print-jump-tables`

  Print jump tables

- `--print-loops`

  Print loop related information

- `--print-mem-data`

  Print memory data annotations when printing functions

- `--print-normalized`

  Print functions after CFG is normalized

- `--print-only=<func1,func2,func3,...>`

  List of functions to print

- `--print-orc`

  Print ORC unwind information for instructions

- `--print-profile`

  Print functions after attaching profile

- `--print-profile-stats`

  Print profile quality/bias analysis

- `--print-pseudo-probes=<value>`

  Print pseudo probe info
  - `decode`: decode probes section from binary
  - `address_conversion`: update address2ProbesMap with output block address
  - `encoded_probes`: display the encoded probes in binary section
  - `all`: enable all debugging printout

- `--print-relocations`

  Print relocations when printing functions/objects

- `--print-reordered-data`

  Print section contents after reordering

- `--print-retpoline-insertion`

  Print functions after retpoline insertion pass

- `--print-sdt`

  Print all SDT markers

- `--print-sections`

  Print all registered sections

- `--print-unknown`

  Print names of functions with unknown control flow

- `--time-build`

  Print time spent constructing binary functions

- `--time-rewrite`

  Print time spent in rewriting passes

- `--print-after-branch-fixup`

  Print function after fixing local branches

- `--print-after-jt-footprint-reduction`

  Print function after jt-footprint-reduction pass

- `--print-after-lowering`

  Print function after instruction lowering

- `--print-cache-metrics`

  Calculate and print various metrics for instruction cache

- `--print-clusters`

  Print clusters

- `--print-estimate-edge-counts`

  Print function after edge counts are set for no-LBR profile

- `--print-finalized`

  Print function after CFG is finalized

- `--print-fix-relaxations`

  Print functions after fix relaxations pass

- `--print-fix-riscv-calls`

  Print functions after fix RISCV calls pass

- `--print-fop`

  Print functions after frame optimizer pass

- `--print-function-statistics=<uint>`

  Print statistics about basic block ordering

- `--print-icf`

  Print functions after ICF optimization

- `--print-icp`

  Print functions after indirect call promotion

- `--print-inline`

  Print functions after inlining optimization

- `--print-large-functions`

  Print functions that could not be overwritten due to excessive size

- `--print-longjmp`

  Print functions after longjmp pass

- `--print-optimize-bodyless`

  Print functions after bodyless optimization

- `--print-output-address-range`

  Print output address range for each basic block in the function
  whenBinaryFunction::print is called

- `--print-peepholes`

  Print functions after peephole optimization

- `--print-plt`

  Print functions after PLT optimization

- `--print-regreassign`

  Print functions after regreassign pass

- `--print-reordered`

  Print functions after layout optimization

- `--print-reordered-functions`

  Print functions after clustering

- `--print-sctc`

  Print functions after conditional tail call simplification

- `--print-simplify-rodata-loads`

  Print functions after simplification of RO data loads

- `--print-sorted-by=<value>`

  Print functions sorted by order of dyno stats
  - `executed-forward-branches`: executed forward branches
  - `taken-forward-branches`: taken forward branches
  - `executed-backward-branches`: executed backward branches
  - `taken-backward-branches`: taken backward branches
  - `executed-unconditional-branches`: executed unconditional branches
  - `all-function-calls`: all function calls
  - `indirect-calls`: indirect calls
  - `PLT-calls`: PLT calls
  - `executed-instructions`: executed instructions
  - `executed-load-instructions`: executed load instructions
  - `executed-store-instructions`: executed store instructions
  - `taken-jump-table-branches`: taken jump table branches
  - `taken-unknown-indirect-branches`: taken unknown indirect branches
  - `total-branches`: total branches
  - `taken-branches`: taken branches
  - `non-taken-conditional-branches`: non-taken conditional branches
  - `taken-conditional-branches`: taken conditional branches
  - `all-conditional-branches`: all conditional branches
  - `linker-inserted-veneer-calls`: linker-inserted veneer calls
  - `all`: sorted by all names

- `--print-sorted-by-order=<value>`

  Use ascending or descending order when printing functions ordered by dyno stats

- `--print-split`

  Print functions after code splitting

- `--print-stoke`

  Print functions after stoke analysis

- `--print-uce`

  Print functions after unreachable code elimination

- `--print-veneer-elimination`

  Print functions after veneer elimination pass

- `--time-opts`

  Print time spent in each optimization

- `--print-all-options`

  Print all option values after command line parsing

- `--print-options`

  Print non-default options after command line parsing
