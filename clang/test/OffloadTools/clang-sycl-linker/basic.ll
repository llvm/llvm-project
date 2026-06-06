; Tests the clang-sycl-linker tool.
;
; REQUIRES: spirv-registered-target
;
; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-as %t/input1.ll -o %t/input1.bc
; RUN: llvm-as %t/input2.ll -o %t/input2.bc
;
; Test --help
; RUN: clang-sycl-linker --help | FileCheck %s --check-prefix=HELP
; HELP: OVERVIEW: A utility that wraps around the SYCL device code linking process.
; HELP: USAGE: clang-sycl-linker [options] <input bitcode files>
; HELP: OPTIONS:
;
; Test --version
; RUN: clang-sycl-linker --version | FileCheck %s --check-prefix=VERSION
; VERSION: clang-sycl-linker version
;
; Test missing input files
; RUN: not clang-sycl-linker -o %t.out 2>&1 | FileCheck %s --check-prefix=NO-INPUT
; NO-INPUT: No input files provided
;
; Test non-existent input file
; RUN: not clang-sycl-linker %t-missing.bc -o %t.out 2>&1 | FileCheck %s --check-prefix=MISSING
; MISSING: input file not found: '{{.*}}-missing.bc'
;
; Test the dry run of a simple case to link two input files.
; Test that IMG_SPIRV image kind is set for non-AOT compilation.
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none %t/input1.bc %t/input2.bc -o %t/spirv.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=SIMPLE-FO
; SIMPLE-FO:      link: inputs: {{.*}}.bc, {{.*}}.bc output: [[LLVMLINKOUT:.*]].bc
; SIMPLE-FO-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: {{.*}}_0.spv
; SIMPLE-FO-NEXT: sycl-bundle: image kind: spv, triple: spirv64, arch: {{$}}
; SIMPLE-FO-NOT:  {{.+}}
;
; Test the dry run of a simple case with device library archive specified using --whole-archive.
; RUN: mkdir -p %t/libs
; RUN: llvm-as %t/lib1.ll -o %t/libs/lib1.bc
; RUN: llvm-as %t/lib2.ll -o %t/libs/lib2.bc
; RUN: rm -f %t/libs/libdevice.a
; RUN: llvm-ar rc %t/libs/libdevice.a %t/libs/lib1.bc %t/libs/lib2.bc
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none %t/input1.bc %t/input2.bc --library-path=%t/libs --whole-archive -l device -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBS
; DEVLIBS:      link: inputs: {{.*}}.bc, {{.*}}.bc, {{.*}}libdevice.a(lib1.bc), {{.*}}libdevice.a(lib2.bc) output: [[LLVMLINKOUT:.*]].bc
; DEVLIBS-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: a_0.spv
; DEVLIBS-NEXT: sycl-bundle: image kind: spv, triple: spirv64, arch: {{$}}
; DEVLIBS-NOT:  {{.+}}
;
; Test -L short form (joined) and -l with archive using --whole-archive.
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none %t/input1.bc -L%t/libs --whole-archive -l device -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBS-SHORT
; DEVLIBS-SHORT: link: inputs: {{.*}}.bc, {{.*}}libdevice.a(lib1.bc), {{.*}}libdevice.a(lib2.bc) output: {{.*}}.bc
;
; Test that search continues past the first -L when the library is not found there. libdevice.a exists only in %t/libs (the second -L).
; RUN: mkdir -p %t/empty
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none %t/input1.bc -L %t/empty -L %t/libs --whole-archive -l device -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBS-FALLTHROUGH
; DEVLIBS-FALLTHROUGH: link: inputs: {{.*}}.bc, {{.*}}libdevice.a(lib1.bc), {{.*}}libdevice.a(lib2.bc) output: {{.*}}.bc
;
; Test a simple case with a random file (not bitcode) as input.
; RUN: touch %t/dummy.o
; RUN: not clang-sycl-linker %t/dummy.o -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=FILETYPEERROR
; FILETYPEERROR: Unsupported file type: '{{.*}}dummy.o'
;
; Test that unsupported file type error includes buffer identifier when found inside an archive.
; Create an archive containing an unsupported file (text file instead of bitcode).
; RUN: echo "not bitcode" > %t/invalid.txt
; RUN: rm -f %t/libinvalid.a
; RUN: llvm-ar rc %t/libinvalid.a %t/invalid.txt
; RUN: not clang-sycl-linker --dry-run %t/input1.bc -L %t --whole-archive -l invalid -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ARCHIVE-INVALID-MEMBER
; ARCHIVE-INVALID-MEMBER: Unsupported file type: '{{.*}}libinvalid.a(invalid.txt)'
;
; Test mixed archive: valid bitcode member + invalid member.
; The error should clearly identify which member is invalid.
; RUN: rm -f %t/libmixed.a
; RUN: llvm-ar rc %t/libmixed.a %t/libs/lib1.bc %t/invalid.txt
; RUN: not clang-sycl-linker --dry-run %t/input1.bc -L %t --whole-archive -l mixed -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=ARCHIVE-MIXED-INVALID
; ARCHIVE-MIXED-INVALID: Unsupported file type: '{{.*}}libmixed.a(invalid.txt)'
;
; Test to see if device library related errors are emitted.
; RUN: not clang-sycl-linker --dry-run %t/input1.bc %t/input2.bc --library-path=%t/libs -l device -l nonexistent -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=DEVLIBSERR
; DEVLIBSERR: unable to find library -lnonexistent
;
; Test that there is no implicit CWD search: a bare library name without any -L
; must fail to resolve, even if a same-named file exists in the CWD.
; RUN: cd %t && not clang-sycl-linker --dry-run input1.bc -l mixed -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-CWD-SEARCH
; NO-CWD-SEARCH: unable to find library -lmixed
;
; Test that a directory matching the requested name is not accepted as a library:
; %t/libs is a directory created above; resolving -l:libs against -L %t
; would detect it's a directory and error with the filename in the message.
; RUN: not clang-sycl-linker --dry-run %t/input1.bc -L %t -l :libs -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-DIR-AS-LIB
; NO-DIR-AS-LIB: '{{.*}}libs': Is a directory
;
; Test that providing only an empty archive results in "No input files could be resolved" error
; RUN: rm -f %t/empty.a
; RUN: llvm-ar rc %t/empty.a
; RUN: not clang-sycl-linker --dry-run --whole-archive %t/empty.a -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-RESOLVED-INPUT
; NO-RESOLVED-INPUT: No input files could be resolved
;
; Test that providing only a lazy archive with no extracted members results in "No input files could be resolved" error
; RUN: not clang-sycl-linker --dry-run %t/libs/libdevice.a -o a.spv 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NO-RESOLVED-LAZY
; NO-RESOLVED-LAZY: No input files could be resolved
;
; Test AOT compilation for an Intel GPU.
; Test that IMG_Object image kind is set for AOT compilation (Intel GPU).
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none -arch=bmg_g21 %t/input1.bc %t/input2.bc -o %t/aot-gpu.out 2>&1 \
; RUN:     --ocloc-options="-a -b" \
; RUN:   | FileCheck %s --check-prefix=AOT-INTEL-GPU
; AOT-INTEL-GPU:      link: inputs: {{.*}}.bc, {{.*}}.bc output: [[LLVMLINKOUT:.*]].bc
; AOT-INTEL-GPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
; AOT-INTEL-GPU-NEXT: "{{.*}}ocloc{{.*}}" {{.*}}-device bmg_g21 -a -b {{.*}}-output [[SPIRVTRANSLATIONOUT]]_0.out -file [[SPIRVTRANSLATIONOUT]]_0.spv
; AOT-INTEL-GPU-NEXT: sycl-bundle: image kind: o, triple: spirv64, arch: bmg_g21
; AOT-INTEL-GPU-NOT:  {{.+}}
;
; Test AOT compilation for an Intel CPU.
; Test that IMG_Object image kind is set for AOT compilation (Intel CPU).
; RUN: clang-sycl-linker --dry-run -v --module-split-mode=none -arch=graniterapids %t/input1.bc %t/input2.bc -o %t/aot-cpu.out 2>&1 \
; RUN:     --opencl-aot-options="-a -b" \
; RUN:   | FileCheck %s --check-prefix=AOT-INTEL-CPU
; AOT-INTEL-CPU:      link: inputs: {{.*}}.bc, {{.*}}.bc output: [[LLVMLINKOUT:.*]].bc
; AOT-INTEL-CPU-NEXT: LLVM backend: input: [[LLVMLINKOUT]].bc, output: [[SPIRVTRANSLATIONOUT:.*]]_0.spv
; AOT-INTEL-CPU-NEXT: "{{.*}}opencl-aot{{.*}}" {{.*}}--device=cpu -a -b {{.*}}-o [[SPIRVTRANSLATIONOUT]]_0.out [[SPIRVTRANSLATIONOUT]]_0.spv
; AOT-INTEL-CPU-NEXT: sycl-bundle: image kind: o, triple: spirv64, arch: graniterapids
; AOT-INTEL-CPU-NOT:  {{.+}}
;
; Check that the output file must be specified.
; RUN: not clang-sycl-linker --dry-run %t/input1.bc %t/input2.bc 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOOUTPUT
; NOOUTPUT: Output file must be specified
;
; Check parser error reporting for unknown options.
; RUN: not clang-sycl-linker --dry-run --not-a-real-flag -triple=spirv64 %t/input1.bc -o a.out 2>&1 \
; RUN:   | FileCheck %s --check-prefix=BADOPT
; BADOPT: unknown argument '--not-a-real-flag'
;
; Input with no entry points still produces an offload image.
; RUN: llvm-as %t/no-entry-points.ll -o %t/no-entry-points.bc
; RUN: clang-sycl-linker %t/no-entry-points.bc -o %t/no-entry-points.out
; RUN: llvm-objdump --offloading %t/no-entry-points.out | FileCheck %s --check-prefix=NO-ENTRY-POINTS
; NO-ENTRY-POINTS: OFFLOADING IMAGE [0]:
; NO-ENTRY-POINTS: producer        sycl
;
; Real (non-dry-run) run: input1/input2 have different
; sycl-module-id values, so two images are produced and packaged on disk.
; Covers write on disk logic not reachable under --dry-run.
; RUN: clang-sycl-linker %t/input1.bc %t/input2.bc -o %t/srcsplit.out
; RUN: llvm-objdump --offloading %t/srcsplit.out | FileCheck %s --check-prefix=SRCSPLIT
; SRCSPLIT:      OFFLOADING IMAGE [0]:
; SRCSPLIT:      kind            spir-v
; SRCSPLIT:      triple          spirv64
; SRCSPLIT:      OFFLOADING IMAGE [1]:
; SRCSPLIT:      kind            spir-v
; SRCSPLIT:      triple          spirv64

;--- input1.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @kernel_a() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="TU1.cpp" }

;--- input2.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @kernel_b() #0 {
  ret void
}

attributes #0 = { "sycl-module-id"="TU2.cpp" }

;--- no-entry-points.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @helper() {
  ret i32 0
}

;--- lib1.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @lib1_func() {
  ret i32 1
}

;--- lib2.ll
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_func i32 @lib2_func() {
  ret i32 2
}
