// RUN: %clang_cc1 -dump-tokens %s 2>&1 | FileCheck %s

->                           // CHECK: arrow            '->'
5                            // CHECK: numeric_constant '5'
id                           // CHECK: identifier       'id'
&                            // CHECK: amp              '&'
)                            // CHECK: r_paren          ')'
unsigned                     // CHECK: unsigned         'unsigned'
~                            // CHECK: tilde            '~'
long_variable_name_very_long // CHECK: identifier       'long_variable_name_very_long'
union                        // CHECK: union            'union'
42                           // CHECK: numeric_constant '42'
j                            // CHECK: identifier       'j'
&=                           // CHECK: ampequal         '&='
15                           // CHECK: numeric_constant '15'

