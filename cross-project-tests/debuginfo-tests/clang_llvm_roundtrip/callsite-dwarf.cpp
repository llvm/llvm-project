// RUN: %clang --target=x86_64-unknown-linux -c -g -O1 %s -o - | \
// RUN: llvm-dwarfdump --debug-info - | FileCheck %s --check-prefix=CHECK

// Simple base and derived class with virtual:
// We check for a generated 'DW_AT_LLVM_virtual_call_origin' for 'foo', that
// corresponds to the 'call_target' metadata added to the indirect call
// instruction.

// Note: We should add a test case inside LLDB that make use of the
//       virtuality call-site target information in DWARF.

struct CBaseOne {
  virtual void foo(int &);
};

struct CDerivedOne : CBaseOne {
  void foo(int &);
};

void CDerivedOne::foo(int &) {}

struct CBaseTwo {
  CDerivedOne *DerivedOne;
};

struct CDerivedTwo : CBaseTwo {
  void bar(int &);
};

void CDerivedTwo::bar(int &j) { DerivedOne->foo(j); }

// The IR generated looks like:
//
// define dso_local void @_ZN11CDerivedTwo3barERi(...) !dbg !40 {
// entry:
//   ..
//   %vtable = load ptr, ptr %0, align 8
//   %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
//   %2 = load ptr, ptr %vfn, align 8
//   call void %2(...), !dbg !65, !call_target !25
//   ret void
// }
//
// !25 = !DISubprogram(name: "foo", linkageName: "_ZN11CDerivedOne3fooERi", ...)
// !40 = !DISubprogram(name: "bar", linkageName: "_ZN11CDerivedTwo3barERi", ...)
// !65 = !DILocation(line: 25, column: 15, scope: !40)

// CHECK: DW_TAG_compile_unit
// CHECK:   DW_TAG_structure_type
// CHECK:     DW_AT_name	("CDerivedOne")
// CHECK: [[FOO_DCL:0x[a-f0-9]+]]:    DW_TAG_subprogram
// CHECK:       DW_AT_name	("foo")
// CHECK:   DW_TAG_structure_type
// CHECK:     DW_AT_name	("CBaseOne")
// CHECK: [[FOO_DEF:0x[a-f0-9]+]]:  DW_TAG_subprogram
// CHECK:     DW_AT_call_all_calls	(true)
// CHECK:     DW_AT_specification	([[FOO_DCL]] "{{.*}}foo{{.*}}")
// CHECK:   DW_TAG_structure_type
// CHECK:     DW_AT_name	("CDerivedTwo")
// CHECK:     DW_TAG_subprogram
// CHECK:       DW_AT_name	("bar")
// CHECK:   DW_TAG_structure_type
// CHECK:     DW_AT_name	("CBaseTwo")
// CHECK:   DW_TAG_subprogram
// CHECK:     DW_AT_call_all_calls	(true)
// CHECK:     DW_AT_specification	(0x{{.*}} "{{.*}}bar{{.*}}")
// CHECK:     DW_TAG_call_site
// CHECK:       DW_AT_call_target_clobbered	(DW_OP_reg0 RAX)
// CHECK:       DW_AT_call_tail_call	(true)
// CHECK:       DW_AT_call_pc	(0x{{.*}})
// CHECK:       DW_AT_LLVM_virtual_call_origin	([[FOO_DCL]] "{{.*}}foo{{.*}}")
