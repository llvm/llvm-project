/*===- DataFlow.cpp - a standalone DataFlow tracer                  -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// An experimental data-flow tracer for fuzz targets.
// It is based on DFSan and SanitizerCoverage.
// https://clang.llvm.org/docs/DataFlowSanitizer.html
// https://clang.llvm.org/docs/SanitizerCoverage.html#tracing-data-flow
//
// It executes the fuzz target on the given input while monitoring the
// data flow for every instrumented comparison instruction.
//
// The output shows which functions depend on which bytes of the input.
//
// Build:
//   1. Compile this file with -fsanitize=dataflow
//   2. Build the fuzz target with -g -fsanitize=dataflow
//       -fsanitize-coverage=trace-pc-guard,pc-table,func,trace-cmp
//   3. Link those together with -fsanitize=dataflow
//
//  -fsanitize-coverage=trace-cmp inserts callbacks around every comparison
//  instruction, DFSan modifies the calls to pass the data flow labels.
//  The callbacks update the data flow label for the current function.
//  See e.g. __dfsw___sanitizer_cov_trace_cmp1 below.
//
//  -fsanitize-coverage=trace-pc-guard,pc-table,func instruments function
//  entries so that the comparison callback knows that current function.
//
//
// Run:
//   # Collect data flow for INPUT_FILE, write to OUTPUT_FILE (default: stdout)
//   ./a.out INPUT_FILE [OUTPUT_FILE]
//
//   # Print all instrumented functions. llvm-symbolizer must be present in PATH
//   ./a.out
//
// Example output:
// ===============
//  F0 11111111111111
//  F1 10000000000000
//  ===============
// "FN xxxxxxxxxx": tells what bytes of the input does the function N depend on.
//    The byte string is LEN+1 bytes. The last byte is set if the function
//    depends on the input length.
//===----------------------------------------------------------------------===*/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <execinfo.h>  // backtrace_symbols_fd

#include <sanitizer/dfsan_interface.h>

extern "C" {
extern int LLVMFuzzerTestOneInput(const unsigned char *Data, size_t Size);
__attribute__((weak)) extern int LLVMFuzzerInitialize(int *argc, char ***argv);
} // extern "C"

static size_t InputLen;
static size_t NumFuncs;
static const uintptr_t *FuncsBeg;
static __thread size_t CurrentFunc;
static dfsan_label *FuncLabels;  // Array of NumFuncs elements.
static char *PrintableStringForLabel;  // InputLen + 2 bytes.

// Prints all instrumented functions.
static int PrintFunctions() {
  // We don't have the symbolizer integrated with dfsan yet.
  // So use backtrace_symbols_fd and pipe it through llvm-symbolizer.
  // TODO(kcc): this is pretty ugly and may break in lots of ways.
  //      We'll need to make a proper in-process symbolizer work with DFSan.
  FILE *Pipe = popen("sed 's/(+/ /g; s/).*//g' "
                     "| llvm-symbolizer "
                     "| grep 'dfs\\$' "
                     "| sed 's/dfs\\$//g'", "w");
  for (size_t I = 0; I < NumFuncs; I++) {
    uintptr_t PC = FuncsBeg[I * 2];
    void *const Buf[1] = {(void*)PC};
    backtrace_symbols_fd(Buf, 1, fileno(Pipe));
  }
  pclose(Pipe);
  return 0;
}

static void SetBytesForLabel(dfsan_label L, char *Bytes) {
  if (L <= InputLen) {
    Bytes[L] = '1';
  } else {
    auto *DLI = dfsan_get_label_info(L);
    SetBytesForLabel(DLI->l1, Bytes);
    SetBytesForLabel(DLI->l2, Bytes);
  }
}

static char *GetPrintableStringForLabel(dfsan_label L) {
  memset(PrintableStringForLabel, '0', InputLen + 1);
  PrintableStringForLabel[InputLen + 1] = 0;
  SetBytesForLabel(L, PrintableStringForLabel);
  return PrintableStringForLabel;
}

static void PrintDataFlow(FILE *Out) {
  for (size_t I = 0; I < NumFuncs; I++)
    if (FuncLabels[I])
      fprintf(Out, "F%zd %s\n", I, GetPrintableStringForLabel(FuncLabels[I]));
}

int main(int argc, char **argv) {
  if (LLVMFuzzerInitialize)
    LLVMFuzzerInitialize(&argc, &argv);
  if (argc == 1)
    return PrintFunctions();
  assert(argc == 2 || argc == 3);

  const char *Input = argv[1];
  fprintf(stderr, "INFO: reading '%s'\n", Input);
  FILE *In = fopen(Input, "r");
  assert(In);
  fseek(In, 0, SEEK_END);
  InputLen = ftell(In);
  fseek(In, 0, SEEK_SET);
  unsigned char *Buf = (unsigned char*)malloc(InputLen);
  size_t NumBytesRead = fread(Buf, 1, InputLen, In);
  assert(NumBytesRead == InputLen);
  PrintableStringForLabel = (char*)malloc(InputLen + 2);
  fclose(In);

  fprintf(stderr, "INFO: running '%s'\n", Input);
  for (size_t I = 1; I <= InputLen; I++) {
    dfsan_label L = dfsan_create_label("", nullptr);
    assert(L == I);
    dfsan_set_label(L, Buf + I - 1, 1);
  }
  dfsan_label SizeL = dfsan_create_label("", nullptr);
  assert(SizeL == InputLen + 1);
  dfsan_set_label(SizeL, &InputLen, sizeof(InputLen));

  LLVMFuzzerTestOneInput(Buf, InputLen);
  free(Buf);

  bool OutIsStdout = argc == 2;
  fprintf(stderr, "INFO: writing dataflow to %s\n",
          OutIsStdout ? "<stdout>" : argv[2]);
  FILE *Out = OutIsStdout ? stdout : fopen(argv[2], "w");
  PrintDataFlow(Out);
  if (!OutIsStdout) fclose(Out);
}

extern "C" {

void __sanitizer_cov_trace_pc_guard_init(uint32_t *start,
                                         uint32_t *stop) {
  assert(NumFuncs == 0 && "This tool does not support DSOs");
  assert(start < stop && "The code is not instrumented for coverage");
  if (start == stop || *start) return;  // Initialize only once.
  for (uint32_t *x = start; x < stop; x++)
    *x = ++NumFuncs;  // The first index is 1.
  FuncLabels = (dfsan_label*)calloc(NumFuncs, sizeof(dfsan_label));
  fprintf(stderr, "INFO: %zd instrumented function(s) observed\n", NumFuncs);
}

void __sanitizer_cov_pcs_init(const uintptr_t *pcs_beg,
                              const uintptr_t *pcs_end) {
  assert(NumFuncs == (pcs_end - pcs_beg) / 2);
  FuncsBeg = pcs_beg;
}

void __sanitizer_cov_trace_pc_indir(uint64_t x){}  // unused.

void __sanitizer_cov_trace_pc_guard(uint32_t *guard){
  uint32_t FuncNum = *guard - 1;  // Guards start from 1.
  assert(FuncNum < NumFuncs);
  CurrentFunc = FuncNum;
}

void __dfsw___sanitizer_cov_trace_switch(uint64_t Val, uint64_t *Cases,
                                         dfsan_label L1, dfsan_label UnusedL) {
  assert(CurrentFunc < NumFuncs);
  FuncLabels[CurrentFunc] = dfsan_union(FuncLabels[CurrentFunc], L1);
}

#define HOOK(Name, Type)                                                       \
  void Name(Type Arg1, Type Arg2, dfsan_label L1, dfsan_label L2) {            \
    assert(CurrentFunc < NumFuncs);                                            \
    FuncLabels[CurrentFunc] =                                                  \
        dfsan_union(FuncLabels[CurrentFunc], dfsan_union(L1, L2));             \
  }

HOOK(__dfsw___sanitizer_cov_trace_const_cmp1, uint8_t)
HOOK(__dfsw___sanitizer_cov_trace_const_cmp2, uint16_t)
HOOK(__dfsw___sanitizer_cov_trace_const_cmp4, uint32_t)
HOOK(__dfsw___sanitizer_cov_trace_const_cmp8, uint64_t)
HOOK(__dfsw___sanitizer_cov_trace_cmp1, uint8_t)
HOOK(__dfsw___sanitizer_cov_trace_cmp2, uint16_t)
HOOK(__dfsw___sanitizer_cov_trace_cmp4, uint32_t)
HOOK(__dfsw___sanitizer_cov_trace_cmp8, uint64_t)

} // extern "C"
