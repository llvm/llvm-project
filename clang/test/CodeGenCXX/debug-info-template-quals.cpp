// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

template<typename _CharT>
struct basic_string {

  basic_string&
  assign(const _CharT* __s)
  {
    return *this;
  }
};

void foo (const char *c) {
  basic_string<char> str;
  str.assign(c);
}

// CHECK: [[P:.*]] = metadata !{i32 {{.*}}, metadata [[CON:.*]]} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
// CHECK: [[CON]] = metadata !{i32 {{.*}}, metadata [[CH:.*]]} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
// CHECK: [[CH]] = metadata !{i32 {{.*}}, metadata !"char", {{.*}}} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
// CHECK: metadata !{i32 {{.*}}, metadata !"_ZN12basic_stringIcE6assignEPKc", metadata !6, i32 7, metadata [[TYPE:.*]], i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, %struct.basic_string* (%struct.basic_string*, i8*)* @_ZN12basic_stringIcE6assignEPKc, null, metadata !18, metadata !1, i32 8} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [assign]
// CHECK: [[TYPE]] = metadata !{i32 {{.*}}, null, metadata [[ARGS:.*]], i32 0, i32 0}
// CHECK: [[ARGS]] = metadata !{metadata !15, metadata !24, metadata [[P]]}
