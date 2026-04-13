// RUN: %clang_cc1 -dump-tokens %s 2>&1 | FileCheck %s --strict-whitespace

// To make location reporting in the test more robust, provide line number and file name explicitly.
#line 2 "dump-tokens.cpp"

// Different kinds of identifiers with different spelling lengths.
->                                  // CHECK:      arrow            '->'                            Loc=<{{.*}}:4:1>     [StartOfLine]
5                                   // CHECK-NEXT: numeric_constant '5'                             Loc=<{{.*}}:5:1>     [StartOfLine]
id                                  // CHECK-NEXT: identifier       'id'                            Loc=<{{.*}}:6:1>     [StartOfLine]
&                                   // CHECK-NEXT: amp              '&'                             Loc=<{{.*}}:7:1>     [StartOfLine]
)                                   // CHECK-NEXT: r_paren          ')'                             Loc=<{{.*}}:8:1>     [StartOfLine]
unsigned                            // CHECK-NEXT: unsigned         'unsigned'                      Loc=<{{.*}}:9:1>     [StartOfLine]
~                                   // CHECK-NEXT: tilde            '~'                             Loc=<{{.*}}:10:1>    [StartOfLine]
long_variable_name_very_long        // CHECK-NEXT: identifier       'long_variable_name_very_long'  Loc=<{{.*}}:11:1>    [StartOfLine]
union                               // CHECK-NEXT: union            'union'                         Loc=<{{.*}}:12:1>    [StartOfLine]
42                                  // CHECK-NEXT: numeric_constant '42'                            Loc=<{{.*}}:13:1>    [StartOfLine]
j                                   // CHECK-NEXT: identifier       'j'                             Loc=<{{.*}}:14:1>    [StartOfLine]
&=                                  // CHECK-NEXT: ampequal         '&='                            Loc=<{{.*}}:15:1>    [StartOfLine]
15                                  // CHECK-NEXT: numeric_constant '15'                            Loc=<{{.*}}:16:1>    [StartOfLine]

// Different locations in line and trailing markers.
 at different locations= in line    // CHECK-NEXT: identifier       'at'                            Loc=<{{.*}}:19:2>    [StartOfLine] [LeadingSpace]
                                    // CHECK-NEXT: identifier       'different'                     Loc=<{{.*}}:19:5>    [LeadingSpace]
                                    // CHECK-NEXT: identifier       'locations'                     Loc=<{{.*}}:19:15>   [LeadingSpace]
                                    // CHECK-NEXT: equal            '='                             Loc=<{{.*}}:19:24>
                                    // CHECK-NEXT: identifier       'in'                            Loc=<{{.*}}:19:26>   [LeadingSpace]
                                    // CHECK-NEXT: identifier       'line'                          Loc=<{{.*}}:19:29>   [LeadingSpace]

// Tokens that require escaping & annotations.
#pragma clang __debug parser_crash  // CHECK-NEXT: annot_pragma_parser_crash                        Loc=<{{.*}}:27:23>
                                    // CHECK-NEXT: eod              '\n'                            Loc=<{{.*}}:27:119>  [LeadingSpace]
#pragma clang __debug captured      // CHECK-NEXT: annot_pragma_captured                            Loc=<{{.*}}:29:120>
#pragma clang __debug dump X        // CHECK-NEXT: annot_pragma_dump                                Loc=<{{.*}}:30:23>
                                    // CHECK-NEXT: identifier       'X'                             Loc=<{{.*}}:30:28>   [LeadingSpace]
                                    // CHECK-NEXT: eod              '\n'                            Loc=<{{.*}}:30:119>  [LeadingSpace]
                                    // CHECK-NEXT: eof              ''                              Loc=<{{.*}}:34:1>

