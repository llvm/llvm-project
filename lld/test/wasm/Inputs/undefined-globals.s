.globl use_undef_global
.globl unused_undef_global
.globl used_undef_global

.globaltype unused_undef_global, i64, immutable
.globaltype used_undef_global, i64, immutable

use_undef_global:
  .functype use_undef_global () -> (i64)
  global.get used_undef_global
  end_function
