

// RUN: %clang -c %s -### 2>&1 | FileCheck %s --check-prefixes=EMPTY
// RUN: %clang -fbounds-safety -c %s -### 2>&1 | FileCheck %s --check-prefixes=EMPTY

// EMPTY-NOT: -fbounds-safety-bringup-missing-checks
// EMPTY-NOT: -fno-bounds-safety-bringup-missing-checks
// EMPTY-NOT: -fbounds-safety-bringup-missing-checks=all
// EMPTY-NOT: -fno-bounds-safety-bringup-missing-checks=all
// EMPTY-NOT: -fbounds-safety-bringup-missing-checks=batch_0
// EMPTY-NOT: -fno-bounds-safety-bringup-missing-checks=batch_0

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks -c %s -### 2>&1 | FileCheck %s --check-prefix=DISABLED
// DISABLED: -fno-bounds-safety-bringup-missing-checks=all
// DISABLED-NOT: -fbounds-safety-bringup-missing-checks=all

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks -c %s -### 2>&1 | FileCheck %s --check-prefix=ENABLED
// ENABLED: -fbounds-safety-bringup-missing-checks=all
// ENABLED-NOT: -fno-bounds-safety-bringup-missing-checks=all

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks -fno-bounds-safety-bringup-missing-checks -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_NEG
// POS_NEG: -fbounds-safety-bringup-missing-checks=all
// POS_NEG: -fno-bounds-safety-bringup-missing-checks=all

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks -fbounds-safety-bringup-missing-checks -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_POS
// NEG_POS: -fno-bounds-safety-bringup-missing-checks=all
// NEG_POS: -fbounds-safety-bringup-missing-checks=all

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=access_size -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_ACCESS
// POS_ACCESS: -fbounds-safety-bringup-missing-checks=access_size

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=indirect_count_update -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_INDIRECT
// POS_INDIRECT: -fbounds-safety-bringup-missing-checks=indirect_count_update

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=return_size -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_RETURN
// POS_RETURN: -fbounds-safety-bringup-missing-checks=return_size

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=ended_by_lower_bound -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_ENDED_BY_LOWER
// POS_ENDED_BY_LOWER: -fbounds-safety-bringup-missing-checks=ended_by_lower_bound

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=compound_literal_init -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_COMPOUND_LITERAL_INIT
// POS_COMPOUND_LITERAL_INIT: -fbounds-safety-bringup-missing-checks=compound_literal_init

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=libc_attributes -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_LIBC_ATTRIBUTES
// POS_LIBC_ATTRIBUTES: -fbounds-safety-bringup-missing-checks=libc_attributes

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=all -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_ALL
// POS_ALL: -fbounds-safety-bringup-missing-checks=all

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=batch_0 -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_batch_0
// POS_batch_0: -fbounds-safety-bringup-missing-checks=batch_0

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=access_size -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_ACCESS
// NEG_ACCESS: -fno-bounds-safety-bringup-missing-checks=access_size

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=indirect_count_update -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_INDIRECT
// NEG_INDIRECT: -fno-bounds-safety-bringup-missing-checks=indirect_count_update

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=return_size -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_RETURN
// NEG_RETURN: -fno-bounds-safety-bringup-missing-checks=return_size

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=ended_by_lower_bound -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_ENDED_BY_LOWER
// NEG_ENDED_BY_LOWER: -fno-bounds-safety-bringup-missing-checks=ended_by_lower_bound

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=compound_literal_init -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_COMPOUND_LITERAL_INIT
// NEG_COMPOUND_LITERAL_INIT: -fno-bounds-safety-bringup-missing-checks=compound_literal_init

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=libc_attributes -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_LIBC_ATTRIBUTES
// NEG_LIBC_ATTRIBUTES: -fno-bounds-safety-bringup-missing-checks=libc_attributes

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_ALL
// NEG_ALL: -fno-bounds-safety-bringup-missing-checks=all

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=batch_0 -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_batch_0
// NEG_batch_0: -fno-bounds-safety-bringup-missing-checks=batch_0

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all -c %s -### 2>&1 | FileCheck %s --check-prefix=POS_COMMA
// POS_COMMA: -fbounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all -c %s -### 2>&1 | FileCheck %s --check-prefix=NEG_COMMA
// NEG_COMMA: -fno-bounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all

// RUN: %clang -fbounds-safety -fbounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all -fno-bounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all -c %s -### 2>&1 | FileCheck %s --check-prefixes=POS_NEG_COMMA
// POS_NEG_COMMA: -fbounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all
// POS_NEG_COMMA: -fno-bounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all

// RUN: %clang -fbounds-safety -fno-bounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all -fbounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all -c %s -### 2>&1 | FileCheck %s --check-prefixes=NEG_POS_COMMA
// NEG_POS_COMMA: -fno-bounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all
// NEG_POS_COMMA: -fbounds-safety-bringup-missing-checks=access_size,indirect_count_update,return_size,ended_by_lower_bound,compound_literal_init,libc_attributes,all

// RUN: %clang -fno-bounds-safety-bringup-missing-checks -c %s -### 2>&1 | FileCheck %s --check-prefixes=UNUSED_NEG
// UNUSED_NEG: warning: argument unused during compilation: '-fno-bounds-safety-bringup-missing-checks'

// RUN: %clang -fbounds-safety-bringup-missing-checks -c %s -### 2>&1 | FileCheck %s --check-prefixes=UNUSED_POS
// UNUSED_POS: warning: argument unused during compilation: '-fbounds-safety-bringup-missing-checks'

// RUN: %clang -fno-bounds-safety-bringup-missing-checks=batch_0 -c %s -### 2>&1 | FileCheck %s --check-prefixes=UNUSED_NEG_BATCH_0
// UNUSED_NEG_BATCH_0: warning: argument unused during compilation: '-fno-bounds-safety-bringup-missing-checks=batch_0'

// RUN: %clang -fbounds-safety-bringup-missing-checks=batch_0 -c %s -### 2>&1 | FileCheck %s --check-prefixes=UNUSED_POS_BATCH_0
// UNUSED_POS_BATCH_0: warning: argument unused during compilation: '-fbounds-safety-bringup-missing-checks=batch_0'
