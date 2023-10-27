/// Because the addresses of __acle_se_\sym_name and \sym_name are equal,
/// the linker creates a secure gateway in ".gnu.sgstubs".
.macro cmse_veneer sym_name, sym_type, sym_binding, acle_sym_type, acle_sym_binding
.align  2
.\sym_binding  \sym_name
.\acle_sym_binding  __acle_se_\sym_name
.type  \sym_name, %\sym_type
.type  __acle_se_\sym_name, %\acle_sym_type
\sym_name:
__acle_se_\sym_name:
  nop
.size  \sym_name, .-\sym_name
.size  __acle_se_\sym_name, .-__acle_se_\sym_name
.endm

/// Because the addresses of __acle_se_\sym_name and \sym_name are not equal,
/// the linker considers that an inline secure gateway exists and does not
/// create one.
.macro cmse_no_veneer sym_name, sym_type, sym_binding, acle_sym_type, acle_sym_binding
.align  2
.\sym_binding  \sym_name
.\acle_sym_binding  __acle_se_\sym_name
.type  \sym_name, %\sym_type
.type  __acle_se_\sym_name, %\acle_sym_type
\sym_name:
	sg
  nop
__acle_se_\sym_name:
  nop
.size  \sym_name, .-\sym_name
.size  __acle_se_\sym_name, .-__acle_se_\sym_name
.endm
