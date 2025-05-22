// This test works on both MSVC and MinGW targets, but they require different
// build commands. By default we use DWARF debug info on MinGW and dbghelp does
// not support it, so we need to specify extra flags to get the compiler to
// generate PDB debug info.

// RUN: %clangxx_asan %if target={{.*-windows-gnu}} %{ -gcodeview -gcolumn-info -Wl,--pdb= -ldbghelp %} -O0 %s -o %t
// RUN: %env_asan_opts=external_symbolizer_path=non-existent\\\\asdf not %run %t 2>&1 | FileCheck %s

#include <windows.h>
#include <dbghelp.h>

#pragma comment(lib, "dbghelp")

int main() {
  // Make sure the RTL recovers from "no options enabled" dbghelp setup.
  SymSetOptions(0);

  // Make sure the RTL recovers from "fInvadeProcess=FALSE".
  if (!SymInitialize(GetCurrentProcess(), 0, FALSE))
    return 42;

  *(volatile int*)0 = 42;
  // CHECK: ERROR: AddressSanitizer: access-violation on unknown address
  // CHECK: The signal is caused by a WRITE memory access.
  // CHECK: Hint: address points to the zero page.
  // CHECK: {{WARNING: .*DbgHelp}}
  // CHECK: {{WARNING: Failed to use and restart external symbolizer}}
  // CHECK: {{#0 0x.* in main.*report_after_syminitialize.cpp:}}[[@LINE-6]]
  // CHECK: AddressSanitizer can not provide additional info.
}
