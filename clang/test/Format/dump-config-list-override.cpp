/// Check that the ForEachMacros, etc. config entries replace default values instead of appending
/// FIXME: clang-format currently start overriding at index 0 (keeping the remaining
/// values) instead of either appending or completely replacing the values.
/// This behaviour is highly confusing. For now this test documents the current state.
// RUN: clang-format -style="{BasedOnStyle: LLVM}" -dump-config %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: clang-format -style="{BasedOnStyle: LLVM, ForEachMacros: ['OVERRIDE_FOREACH']}" -dump-config %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK,OVERRIDE,FIXME-SHOULD-NOT-BE
// RUN: clang-format -style="{BasedOnStyle: LLVM, ForEachMacros: ['M1', 'M2', 'M3', 'M4']}" -dump-config %s | \
// RUN:   FileCheck %s --check-prefixes=CHECK,MORE-ENTRIES-THAN-DEFAULT


// CHECK-LABEL:   ForEachMacros:
// DEFAULT-NEXT:  {{^  }}- foreach
// DEFAULT-NEXT:  {{^  }}- Q_FOREACH
// DEFAULT-NEXT:  {{^  }}- BOOST_FOREACH
// OVERRIDE-NEXT: {{^  }}- OVERRIDE_FOREACH
// FIXME-SHOULD-NOT-BE-NEXT:  {{^  }}- Q_FOREACH
// FIXME-SHOULD-NOT-BE-NEXT:  {{^  }}- BOOST_FOREACH
// MORE-ENTRIES-THAN-DEFAULT-NEXT: {{^  }}- M1
// MORE-ENTRIES-THAN-DEFAULT-NEXT: {{^  }}- M2
// MORE-ENTRIES-THAN-DEFAULT-NEXT: {{^  }}- M3
// MORE-ENTRIES-THAN-DEFAULT-NEXT: {{^  }}- M4
// CHECK-NEXT:    {{^[F-Z]}}
