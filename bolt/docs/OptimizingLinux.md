# Optimizing Linux Kernel with BOLT


## Introduction

Many Linux applications spend a significant amount of their execution time in the kernel. Thus, when we consider code optimization for system performance, it is essential to improve the CPU utilization not only in the user-space applications and libraries but also in the kernel. BOLT has demonstrated double-digit gains while being applied to user-space programs. This guide shows how to apply BOLT to the x86-64 Linux kernel and enhance your system's performance. In our experiments, BOLT boosted database TPS by 2 percent when applied to the kernel compiled with the highest level optimizations, including PGO and LTO. The database spent ~40% of the time in the kernel and was quite sensitive to kernel performance.

BOLT optimizes code layout based on a low-level execution profile collected with the Linux `perf` tool. The best quality profile should include branch history, such as Intel's last branch records (LBR). BOLT runs on a linked binary and reorders the code while combining frequently executed blocks of instructions in a manner best suited for the hardware. Other than branch instructions, most of the code is left unchanged. Additionally, BOLT updates all metadata associated with the modified code, including DWARF debug information and Linux ORC unwind information.

While BOLT optimizations are not specific to the Linux kernel, certain quirks distinguish the kernel from user-level applications.

BOLT has been successfully applied to and tested with several flavors of the x86-64 Linux kernel.


## QuickStart Guide

BOLT operates on a statically-linked kernel executable, a.k.a. `vmlinux` binary. However, most Linux distributions use a `vmlinuz` compressed image for system booting. To use BOLT on the kernel, you must either repackage `vmlinuz` after BOLT optimizations or add steps for running BOLT into the kernel build and rebuild `vmlinuz`. Uncompressing `vmlinuz` and repackaging it with a new `vmlinux` binary falls beyond the scope of this guide, and at some point, we may add the capability to run BOLT directly on `vmlinuz`. Meanwhile, this guide focuses on steps for integrating BOLT into the kernel build process.


### Building the Kernel

After downloading the kernel sources and configuration for your distribution, you should be able to build `vmlinuz` using the `make bzImage` command. Ideally, the kernel should binary match the kernel on the system you are about to optimize (the target system). The binary matching part is critical as BOLT performs profile matching and optimizations at the binary level. We recommend installing a freshly built kernel on the target system to avoid any discrepancies.

Note that the kernel build will produce several artifacts besides bzImage. The most important of them is the uncompressed `vmlinux` binary, which will be used in the next steps. Make sure to save this file.

Build and target systems should have a `perf` tool installed for collecting and processing profiles. If your build system differs from the target, make sure `perf` versions are compatible. The build system should also have the latest BOLT binary and tools (`llvm-bolt`, `perf2bolt`).

Once the target system boots with the freshly-built kernel, start your workload, such as a database benchmark. While the system is under load, collect the kernel profile using perf:


```bash
$ sudo perf record -a -e cycles -j any,k -F 5000 -- sleep 600
```


Convert `perf` profile into a format suitable for BOLT passing the `vmlinux` binary to `perf2bolt`:


```bash
$ sudo chown $USER perf.data
$ perf2bolt -p perf.data -o perf.fdata vmlinux
```


Under a high load, `perf.data` should be several gigabytes in size and you should expect the converted `perf.fdata` not to exceed 100 MB.

Profiles collected from multiple workloads could be joined into a single profile using `merge-fdata` utility:
```bash
$ merge-fdata perf.1.fdata perf.2.fdata ... perf.<N>.fdata > perf.merged.fdata
```

Two changes are required for the kernel build. The first one is optional but highly recommended. It introduces a BOLT-reserved space into `vmlinux` code section:


```diff
--- a/arch/x86/kernel/vmlinux.lds.S
+++ b/arch/x86/kernel/vmlinux.lds.S
@@ -139,6 +139,11 @@ SECTIONS
                STATIC_CALL_TEXT
                *(.gnu.warning)

+    /* Allocate space for BOLT */
+    __bolt_reserved_start = .;
+               . += 2048 * 1024;
+    __bolt_reserved_end = .;
+
 #ifdef CONFIG_RETPOLINE
                __indirect_thunk_start = .;
                *(.text.__x86.*)
```


The second patch adds a step that runs BOLT on `vmlinux` binary:


```diff
--- a/scripts/link-vmlinux.sh
+++ b/scripts/link-vmlinux.sh
@@ -340,5 +340,13 @@ if is_enabled CONFIG_KALLSYMS; then
        fi
 fi

+# Apply BOLT
+BOLT=llvm-bolt
+BOLT_PROFILE=perf.fdata
+BOLT_OPTS="--dyno-stats --eliminate-unreachable=0 --reorder-blocks=ext-tsp --simplify-conditional-tail-calls=0 --skip-funcs=__entry_text_start,irq_entries_start --split-functions"
+mv vmlinux vmlinux.pre-bolt
+echo BOLTing vmlinux
+${BOLT} vmlinux.pre-bolt -o vmlinux --data ${BOLT_PROFILE} ${BOLT_OPTS}
+
 # For fixdep
 echo "vmlinux: $0" > .vmlinux.d
```


If you skipped the first step or are running BOLT on a pre-built `vmlinux` binary, drop the `--split-functions` option.


## Performance Expectations

By improving the code layout, BOLT can boost the kernel's performance by up to 5% by reducing instruction cache misses and branch mispredictions. When measuring total system performance, you should scale this number accordingly based on the time your application spends in the kernel (excluding I/O time).


## Profile Quality

The timing and duration of the profiling may have a significant effect on the performance of the BOLTed kernel. If you don't know your workload well, it's recommended that you profile for the whole duration of the benchmark run. As longer times will result in larger `perf.data` files, you can lower the profiling frequency by providing a smaller value of `-F` flag. E.g., to record the kernel profile for half an hour, use the following command:


```bash
$ sudo perf record -a -e cycles -j any,k -F 1000 -- sleep 1800
```



## BOLT Disassembly

BOLT annotates the disassembly with control-flow information and attaches Linux-specific metadata to the code. To view annotated disassembly, run:


```bash
$ llvm-bolt vmlinux -o /dev/null --print-cfg
```


If you want to limit the disassembly to a set of functions, add `--print-only=<func1regex>,<func2regex>,...`, where a function name is specified using regular expressions.
