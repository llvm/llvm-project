# Optimizing Clang : A Practical Example of Applying BOLT

## Preface

*BOLT* (Binary Optimization and Layout Tool) is designed to improve the application
performance by laying out code in a manner that helps CPU better utilize its caching and
branch predicting resources.

The most obvious candidates for BOLT optimizations
are programs that suffer from many instruction cache and iTLB misses, such as
large applications measuring over hundreds of megabytes in size. However, medium-sized
programs can benefit too. Clang, one of the most popular open-source C/C++ compilers,
is a good example of the latter. Its code size could easily be in the order of tens of megabytes.
As we will see, the Clang binary suffers from many instruction cache
misses and can be significantly improved with BOLT, even on top of profile-guided and
link-time optimizations.

In this tutorial we will first build Clang with PGO and LTO, and then will show steps on how to
apply BOLT optimizations to make Clang up to 15% faster. We will also analyze where
the compile-time performance gains are coming from, and verify that the speed-ups are
sustainable while building other applications.

## Building Clang

The process of getting Clang sources and performing the build is very similar to the
one described at http://clang.llvm.org/get_started.html. For completeness, we provide the detailed steps
on how to obtain and build Clang in [Bootstrapping Clang-7 with PGO and LTO](#bootstrapping-clang-7-with-pgo-and-lto) section.

The only difference from the standard Clang build is that we require the `-Wl,-q` flag to be present during
the final link. This option saves relocation metadata in the executable file, but does not affect
the generated code in any way.

## Optimizing Clang with BOLT

We will use the setup described in [Bootstrapping Clang-7 with PGO and LTO](#bootstrapping-clang-7-with-pgo-and-lto).
Adjust the steps accordingly if you skipped that section. We will also assume that `llvm-bolt` is present in your `$PATH`.

Before we can run BOLT optimizations, we need to collect the profile for Clang, and we will use
Clang/LLVM sources for that.
Collecting accurate profile requires running `perf` on a hardware that
implements taken branch sampling (`-b/-j` flag). For that reason, it may not be possible to
collect the accurate profile in a virtualized environment, e.g. in the cloud.
We do support regular sampling profiles, but the performance
improvements are expected to be more modest.

```bash
$ mkdir ${TOPLEV}/stage3
$ cd ${TOPLEV}/stage3
$ CPATH=${TOPLEV}/stage2-prof-use-lto/install/bin/
$ cmake -G Ninja ${TOPLEV}/llvm -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CPATH/clang -DCMAKE_CXX_COMPILER=$CPATH/clang++ \
    -DLLVM_USE_LINKER=lld -DCMAKE_INSTALL_PREFIX=${TOPLEV}/stage3/install
$ perf record -e cycles:u -j any,u -- ninja clang
```

Once the last command is finished, it will create a `perf.data` file larger than 10GiB.
We will first convert this profile into a more compact aggregated
form suitable to be consumed by BOLT:
```bash
  $ perf2bolt $CPATH/clang-7 -p perf.data -o clang-7.fdata -w clang-7.yaml
```
Notice that we are passing `clang-7` to `perf2bolt` which is the real binary that
`clang` and `clang++` are symlinking to. The next step will optimize Clang using
the generated profile:
```bash
$ llvm-bolt $CPATH/clang-7 -o $CPATH/clang-7.bolt -b clang-7.yaml \
    -reorder-blocks=cache+ -reorder-functions=hfsort+ -split-functions=3 \
    -split-all-cold -dyno-stats -icf=1 -use-gnu-stack
```
The output will look similar to the one below:
```t
...
BOLT-INFO: enabling relocation mode
BOLT-INFO: 11415 functions out of 104526 simple functions (10.9%) have non-empty execution profile.
...
BOLT-INFO: ICF folded 29144 out of 105177 functions in 8 passes. 82 functions had jump tables.
BOLT-INFO: Removing all identical functions will save 5466.69 KB of code space. Folded functions were called 2131985 times based on profile.
BOLT-INFO: basic block reordering modified layout of 7848 (10.32%) functions
...
           660155947 : executed forward branches (-2.3%)
            48252553 : taken forward branches (-57.2%)
           129897961 : executed backward branches (+13.8%)
            52389551 : taken backward branches (-19.5%)
            35650038 : executed unconditional branches (-33.2%)
           128338874 : all function calls (=)
            19010563 : indirect calls (=)
             9918250 : PLT calls (=)
          6113398840 : executed instructions (-0.6%)
          1519537463 : executed load instructions (=)
           943321306 : executed store instructions (=)
            20467109 : taken jump table branches (=)
           825703946 : total branches (-2.1%)
           136292142 : taken branches (-41.1%)
           689411804 : non-taken conditional branches (+12.6%)
           100642104 : taken conditional branches (-43.4%)
           790053908 : all conditional branches (=)
...
```
The statistics in the output is based on the LBR profile collected with `perf`, and since we were using
the `cycles` counter, its accuracy is affected. However, the relative improvement in `taken conditional
 branches` is a good indication that BOLT was able to straighten out the code even after PGO.

## Measuring Compile-time Improvement

`clang-7.bolt` can be used as a replacement for *PGO+LTO* Clang:
```bash
$ mv $CPATH/clang-7 $CPATH/clang-7.org
$ ln -fs $CPATH/clang-7.bolt $CPATH/clang-7
```
Doing a new build of Clang using the new binary shows a significant overall
build time reduction on a 48-core Haswell system:
```bash
$ ln -fs $CPATH/clang-7.org $CPATH/clang-7
$ ninja clean && /bin/time -f %e ninja clang -j48
202.72
$ ln -fs $CPATH/clang-7.bolt $CPATH/clang-7
$ ninja clean && /bin/time -f %e ninja clang -j48
180.11
```
That's 22.61 seconds (or 12%) faster compared to the *PGO+LTO* build.
Notice that we are measuring an improvement of the total build time, which includes the time spent in the linker.
Compilation time improvements for individual files differ, and speedups over 15% are not uncommon.
If we run BOLT on a Clang binary compiled without *PGO+LTO* (in which case the build is finished in 253.32 seconds),
the gains we see are over 50 seconds (25%),
but, as expected, the result is still slower than *PGO+LTO+BOLT* build.

## Source of the Wins

We mentioned that Clang suffers from considerable instruction cache misses. This can be measured with `perf`:
```bash
$ ln -fs $CPATH/clang-7.org $CPATH/clang-7
$ ninja clean && perf stat -e instructions,L1-icache-misses -- ninja clang -j48
  ...
   16,366,101,626,647      instructions
      359,996,216,537      L1-icache-misses
```
That's about 22 instruction cache misses per thousand instructions. As a rule of thumb, if the application
has over 10 misses per thousand instructions, it is a good indication that it will be improved by BOLT.
Now let's see how many misses are in the BOLTed binary:
```bash
$ ln -fs $CPATH/clang-7.bolt $CPATH/clang-7
$ ninja clean && perf stat -e instructions,L1-icache-misses -- ninja clang -j48
  ...
  16,319,818,488,769      instructions
     244,888,677,972      L1-icache-misses
```
The number of misses per thousand instructions went down from 22 to 15, significantly reducing
the number of stalls in the CPU front-end.
Notice how the number of executed instructions stayed roughly the same. That's because we didn't
run any optimizations beyond the ones affecting the code layout. Other than instruction cache misses,
BOLT also improves branch mispredictions, iTLB misses, and misses in L2 and L3.

## Using Clang for Other Applications

We have collected profile for Clang using its own source code. Would it be enough to speed up
the compilation of other projects? We picked `mysqld`, an open-source database, to do the test.

On our 48-core Haswell system using the *PGO+LTO* Clang, the build finished in 136.06 seconds, while using the *PGO+LTO+BOLT* Clang, 126.10 seconds.
That's a noticeable improvement, but not as significant as the one we saw on Clang itself.
This is partially because the number of instruction cache misses is slightly lower on this scenario : 19 vs 22.
Another reason is that Clang is run with a different set of options while building `mysqld` compared
to the training run.

Different options exercise different code paths, and
if we trained without a specific option, we may have misplaced parts of the code responsible for handling it.
To test this theory, we have collected another `perf` profile while building `mysqld`, and merged it with an existing profile
using the `merge-fdata` utility that comes with BOLT. Optimized with that profile, the *PGO+LTO+BOLT* Clang was able
to perform the `mysqld` build in 124.74 seconds, i.e. 11 seconds or 9% faster compared to *PGO+LGO* Clang.
The merged profile didn't make the original Clang compilation slower either, while the number of profiled functions in Clang increased from 11,415 to 14,025.

Ideally, the profile run has to be done with a superset of all commonly used options. However, the main improvement is expected with just the basic set.

## Summary

In this tutorial we demonstrated how to use BOLT to improve the
performance of the Clang compiler. Similarly, BOLT could be used to improve the performance
of GCC, or any other application suffering from a high number of instruction
cache misses.

----
# Appendix

## Bootstrapping Clang-7 with PGO and LTO

Below we describe detailed steps to build Clang, and make it ready for BOLT
optimizations. If you already have the build setup, you can skip this section,
except for the last step that adds `-Wl,-q` linker flag to the final build.

### Getting Clang-7 Sources

Set `$TOPLEV` to the directory of your preference where you would like to do
builds. E.g. `TOPLEV=~/clang-7/`. Follow with commands to clone the `release_70`
branch of LLVM monorepo:
```bash
$ mkdir ${TOPLEV}
$ cd ${TOPLEV}
$ git clone --branch=release/7.x https://github.com/llvm/llvm-project.git
```

### Building Stage 1 Compiler

Stage 1 will be the first build we are going to do, and we will be using the
default system compiler to build Clang. If your system lacks a compiler, use
your distribution package manager to install one that supports C++11. In this
example we are going to use GCC. In addition to the compiler, you will need the
`cmake` and `ninja` packages. Note that we disable the build of certain
compiler-rt components that are known to cause build issues at release/7.x.
```bash
$ mkdir ${TOPLEV}/stage1
$ cd ${TOPLEV}/stage1
$ cmake -G Ninja ${TOPLEV}/llvm-project/llvm -DLLVM_TARGETS_TO_BUILD=X86 \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_ASM_COMPILER=gcc \
      -DLLVM_ENABLE_PROJECTS="clang;lld" \
      -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
      -DCOMPILER_RT_BUILD_SANITIZERS=OFF -DCOMPILER_RT_BUILD_XRAY=OFF \
      -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
      -DCMAKE_INSTALL_PREFIX=${TOPLEV}/stage1/install
$ ninja install
```

### Building Stage 2 Compiler With Instrumentation

Using the freshly-baked stage 1 Clang compiler, we are going to build Clang with
profile generation capabilities:
```bash
$ mkdir ${TOPLEV}/stage2-prof-gen
$ cd ${TOPLEV}/stage2-prof-gen
$ CPATH=${TOPLEV}/stage1/install/bin/
$ cmake -G Ninja ${TOPLEV}/llvm-project/llvm -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CPATH/clang -DCMAKE_CXX_COMPILER=$CPATH/clang++ \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_USE_LINKER=lld -DLLVM_BUILD_INSTRUMENTED=ON \
    -DCMAKE_INSTALL_PREFIX=${TOPLEV}/stage2-prof-gen/install
$ ninja install
```

### Generating Profile for PGO

While there are many ways to obtain the profile data, we are going to use the
source code already at our disposal, i.e. we are going to collect the profile
while building Clang itself:
```bash
$ mkdir ${TOPLEV}/stage3-train
$ cd ${TOPLEV}/stage3-train
$ CPATH=${TOPLEV}/stage2-prof-gen/install/bin
$ cmake -G Ninja ${TOPLEV}/llvm-project/llvm -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CPATH/clang -DCMAKE_CXX_COMPILER=$CPATH/clang++ \
    -DLLVM_ENABLE_PROJECTS="clang" \
    -DLLVM_USE_LINKER=lld -DCMAKE_INSTALL_PREFIX=${TOPLEV}/stage3-train/install
$ ninja clang
```
Once the build is completed, the profile files will be saved under
`${TOPLEV}/stage2-prof-gen/profiles`. We will merge them before they can be
passed back into Clang:
```bash
$ cd ${TOPLEV}/stage2-prof-gen/profiles
$ ${TOPLEV}/stage1/install/bin/llvm-profdata merge -output=clang.profdata *
```

### Building Clang with PGO and LTO

Now the profile can be used to guide optimizations to produce better code for
our scenario, i.e. building Clang. We will also enable link-time optimizations
to allow cross-module inlining and other optimizations. Finally, we are going to
add one extra step that is useful for BOLT: a linker flag instructing it to
preserve relocations in the output binary. Note that this flag does not affect
the generated code or data used at runtime, it only writes metadata to the file
on disk:
```bash
$ mkdir ${TOPLEV}/stage2-prof-use-lto
$ cd ${TOPLEV}/stage2-prof-use-lto
$ CPATH=${TOPLEV}/stage1/install/bin/
$ export LDFLAGS="-Wl,-q"
$ cmake -G Ninja ${TOPLEV}/llvm-project/llvm -DLLVM_TARGETS_TO_BUILD=X86 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=$CPATH/clang -DCMAKE_CXX_COMPILER=$CPATH/clang++ \
    -DLLVM_ENABLE_PROJECTS="clang;lld" \
    -DLLVM_ENABLE_LTO=Full \
    -DLLVM_PROFDATA_FILE=${TOPLEV}/stage2-prof-gen/profiles/clang.profdata \
    -DLLVM_USE_LINKER=lld \
    -DCMAKE_INSTALL_PREFIX=${TOPLEV}/stage2-prof-use-lto/install
$ ninja install
```
Now we have a Clang compiler that can build itself much faster. As we will see,
it builds other applications faster as well, and, with BOLT, the compile time
can be improved even further.
